#!/home/lsy/anaconda3/envs/ovir3d/bin/python

import argparse
import sys
import copy

import numpy as np
import cv2
import clip
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

# ROS library
import rospy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse
from matplotlib import pyplot as plt

from pf_pose_estimation.tsdf_lib import TSDFVolume
from pf_pose_estimation.particle_filter import ParticleFilter



def main():
    rospy.init_node('detic_service')

    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--vocabulary", default="lvis", choices=['lvis', 'custom', 'ycb_video',
                                                                 'scannet200', 'imagenet21k'])
    parser.add_argument("--custom_vocabulary", default="", help="comma separated words")
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--save_vis", action='store_true')
    parser.add_argument("--depth_scale", type=float, default=1000)
    parser.add_argument("--opts", help="'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    detic_service = DeticService(args)
    print("Detic service started. Waiting for requests...")
    rospy.spin()


def get_clip_feature(clip_model, text_label, normalize=True, prompt_fn=lambda x: f"a {x}", device="cpu"):
    print("computing text features...")
    if isinstance(text_label, str):
        text_inputs = clip.tokenize(prompt_fn(text_label)).to(device)
    else:
        text_inputs = torch.cat([clip.tokenize(prompt_fn(c)) for c in text_label]).to(device)

    # in case the vocab is too large
    chunk_size = 100
    chunks = torch.split(text_inputs, chunk_size, dim=0)
    text_features = []
    for i, chunk in enumerate(chunks):
        chunk_feature = clip_model.encode_text(chunk).detach()
        text_features.append(chunk_feature)

    text_features = torch.cat(text_features, dim=0)
    if normalize:
        text_features = F.normalize(text_features, dim=-1).detach()
    return text_features


def create_pcd(depth_im: np.ndarray,
               cam_intr: np.ndarray,
               color_im: np.ndarray = None,
               cam_extr: np.ndarray = np.eye(4)):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(color_im_o3d, depth_im_o3d, depth_scale=1,
                                                                    convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr,
                                                                depth_scale=1)
    return pcd


class DeticService:

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detic_srv_name = 'detic_service'
        self.init_tracker_srv = rospy.Service(self.detic_srv_name, GetDeticResults, self.get_detic_result)

        self.bridge = CvBridge()

        setup_logger(name="fvcore")
        cfg = self.setup_cfg(args)
        self.detic = VisualizationDemo(cfg, args)
        self.clip_model, _ = clip.load('ViT-B/32', self.device)

        target_list = ['006_mustard_bottle']
        self.target_tsdf_dict = dict()
        for target_name in target_list:
            obj_tsdf_path = f'data/{target_name}/tsdf.npz'
            obj_tsdf_vol = TSDFVolume.load(obj_tsdf_path)
            self.target_tsdf_dict[target_name] = obj_tsdf_vol

    def setup_cfg(self, args):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
        cfg.MODEL.WEIGHTS = "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        cfg.DATALOADER.NUM_WORKERS = 2
        if not args.pred_all_class:
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        return cfg

    @staticmethod
    def generate_coordinate_frame(T, scale=0.05):
        mesh = o3d.geometry.TriangleMesh().create_coordinate_frame()
        mesh.scale(scale, center=np.array([0, 0, 0]))
        return mesh.transform(T)

    @staticmethod
    def plane_detection_o3d(pcd: o3d.geometry.PointCloud,
                            inlier_thresh: float,
                            max_iterations: int = 1000,
                            visualize: bool = False,
                            in_cam_frame: bool = True):
        # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=inlier_thresh,
                                                ransac_n=3,
                                                num_iterations=max_iterations)
        [a, b, c, d] = plane_model  # ax + by + cz + d = 0
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # sample the inlier point that is closest to the camera origin as the world origin
        inlier_pts = np.asarray(inlier_cloud.points)
        squared_distances = np.sum(inlier_pts ** 2, axis=1)
        closest_index = np.argmin(squared_distances)
        x, y, z = inlier_pts[closest_index]
        origin = np.array([x, y, (-d - a * x - b * y) / (c + 1e-12)])
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)

        if in_cam_frame:
            if plane_normal @ origin > 0:
                plane_normal *= -1
        elif plane_normal[2] < 0:
            plane_normal *= -1

        # randomly sample x_dir and y_dir given plane normal as z_dir
        x_dir = np.array([-plane_normal[2], 0, plane_normal[0]])
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(plane_normal, x_dir)
        plane_frame = np.eye(4)
        plane_frame[:3, 0] = x_dir
        plane_frame[:3, 1] = y_dir
        plane_frame[:3, 2] = plane_normal
        plane_frame[:3, 3] = origin

        if visualize:
            plane_frame_vis = DeticService.generate_coordinate_frame(plane_frame, scale=0.05)
            cam_frame_vis = DeticService.generate_coordinate_frame(np.eye(4), scale=0.05)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_frame_vis, cam_frame_vis])

        return plane_frame, inliers

    def get_detic_result(self, req: GetDeticResultsRequest):
        target_name = req.target_name.data
        print(f"{target_name = }")

        camera_info = req.cam_info
        cam_intr = np.array(camera_info.K).reshape((3, 3))
        rgb_msg = req.color_img
        depth_msg = req.depth_img
        rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1').astype(np.float32) / self.args.depth_scale

        # for debugging
        #self.args.depth_scale = 4000
        #rgb_im_path = "data/color.png"
        #depth_im_path = "data/depth.png"
        #rgb_im = cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB)  # for debug
        #depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / self.args.depth_scale
        #fx, fy = 459.906, 460.156
        #cx, cy = 347.191, 256.039
        #cam_intr = np.array([
        #    [fx, 0, cx],
        #    [0, fy, cy],
        #    [0, 0, 1]
        #])

        if req.debug_mode:
            plt.imshow(rgb_im)
            plt.show()
            plt.imshow(depth_im)
            plt.show()

        vocab_features = get_clip_feature(clip_model=self.clip_model, text_label=target_name, normalize=True,
                                          device=self.device)
        vocab_feature = vocab_features.squeeze(0).float()

        bgr_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
        instances = self.detic.predict_instances_only(bgr_im)

        if req.debug_mode:
            metadata = self.detic.metadata
            visualizer = Visualizer(rgb_im, metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            vis_im = vis_output.get_image()
            plt.imshow(vis_im)
            plt.show()

        pred_scores = instances.scores.numpy()  # (M,)
        pred_masks = instances.pred_masks.numpy()  # (M, H, W)
        pred_features = instances.pred_box_features.to(self.device)  # (M, 512)
        pred_features = F.normalize(pred_features, dim=1, p=2)

        similarity = pred_features @ vocab_feature
        best_match = similarity.argmax()
        target_mask = pred_masks[best_match]
        if req.debug_mode:
            print(f"{similarity = }")
            print(f"{best_match = }")
            plt.imshow(target_mask)
            plt.show()

        scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
        scene_pts = np.asarray(scene_pcd.points)
        scene_rgb = np.asarray(scene_pcd.colors)
        table_plane, table_indices = self.plane_detection_o3d(scene_pcd, inlier_thresh=0.01, visualize=req.debug_mode)

        masked_depth_im = depth_im * target_mask
        target_pcd = create_pcd(masked_depth_im, cam_intr, color_im=rgb_im)

        scene_kdtree = KDTree(scene_pts)
        target_pts = np.asarray(target_pcd.points)
        _, target_indices = scene_kdtree.query(target_pts)
        target_mask_3d = np.zeros(scene_pts.shape[0], dtype=np.bool_)
        target_mask_3d[target_indices] = True

        background_mask_3d = np.ones(scene_pts.shape[0], dtype=np.bool_)
        background_mask_3d[table_indices] = False
        background_mask_3d[target_indices] = False

        if req.debug_mode:
            pcd_vis = copy.deepcopy(scene_pcd)
            pcd_colors = np.asarray(pcd_vis.colors)
            pcd_colors[target_mask_3d] = (1, 0, 0)
            pcd_colors[background_mask_3d] = (0, 0, 1)
            o3d.visualization.draw_geometries([pcd_vis])

        pose_msg = Pose()
        if target_name in self.target_tsdf_dict:
            obj_tsdf_vol = self.target_tsdf_dict[target_name]
            pf = ParticleFilter(obj_tsdf_vol, num_particles=2000)
            pose, _ = pf.estimate(rgb_im, depth_im, cam_intr, mask=target_mask, num_iters=100, visualize=req.debug_mode)
            print(pose)
            pose_msg.position.x = pose[0, 3]
            pose_msg.position.y = pose[1, 3]
            pose_msg.position.z = pose[2, 3]
            quaternion = R.from_matrix(pose[:3, :3]).as_quat()
            pose_msg.orientation.x = quaternion[0]
            pose_msg.orientation.y = quaternion[1]
            pose_msg.orientation.z = quaternion[2]
            pose_msg.orientation.w = quaternion[3]

        ret = GetDeticResultsResponse()
        ret.success = True
        ret.pose = pose_msg
        ret.points = Float32MultiArray(data=scene_pts.flatten().tolist())
        ret.colors = Float32MultiArray(data=scene_rgb.flatten().tolist())
        ret.target_mask = target_mask_3d.flatten().tolist()
        ret.background_mask = background_mask_3d.flatten().tolist()
        return ret


if __name__ == '__main__':
    main()
