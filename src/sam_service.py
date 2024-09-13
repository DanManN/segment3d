#!/home/lsy/anaconda3/envs/ovir3d/bin/python
import sys
import copy

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

from lang_sam import LangSAM

# ROS library
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse


def create_pcd(
    depth_im: np.ndarray,
    cam_intr: np.ndarray,
    color_im: np.ndarray = None,
    cam_extr: np.ndarray = np.eye(4)
):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color_im_o3d,
            depth_im_o3d,
            depth_scale=1,
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
            rgbd, intrinsic_o3d, extrinsic=cam_extr
        )
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(
            depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr, depth_scale=1
        )
    return pcd


bridge = CvBridge()
model = LangSAM('vit_b')  #,'./sam_vit_b_01ec64.pth')


def get_result(req: GetDeticResultsRequest):
    target_name = req.target_name.data
    print(f"{target_name = }")

    camera_info = req.cam_info
    cam_intr = np.array(camera_info.K).reshape((3, 3))
    rgb_msg = req.color_img
    depth_msg = req.depth_img
    rgb_im = bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
    depth_im = bridge.imgmsg_to_cv2(depth_msg,
                                    '32FC1').astype(np.float32) / 1000
    image_pil = Image.fromarray(rgb_im)

    # for debugging
    #rgb_im_path = "data/color.png"
    #depth_im_path = "data/depth.png"
    #rgb_im = cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB)  # for debug
    #depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
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
        image_pil.show()
        plt.imshow(depth_im)
        plt.show()

    masks, boxes, phrases, logits = model.predict(image_pil, target_name)
    target_mask = np.asarray(masks[0])
    # target_mask = cv2.erode(target_mask.astype(float), np.ones((7, 7))).astype(bool)
    if req.debug_mode:
        print(f"{similarity = }")
        print(f"{best_match = }")
        plt.imshow(target_mask)
        plt.show()

    scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
    scene_pts = np.asarray(scene_pcd.points)
    scene_rgb = np.asarray(scene_pcd.colors)

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

    ret = GetDeticResultsResponse()
    ret.success = True
    ret.points = Float32MultiArray(data=scene_pts.flatten().tolist())
    ret.colors = Float32MultiArray(data=scene_rgb.flatten().tolist())
    ret.target_mask = target_mask_3d.flatten().tolist()
    ret.background_mask = background_mask_3d.flatten().tolist()
    ret.target_image_mask = bridge.cv2_to_imgmsg(
        target_mask.astype(np.uint8), encoding="mono8"
    )
    return ret


if __name__ == '__main__':
    init_tracker_srv = rospy.Service(
        'lang_sam_service', GetDeticResults, get_result
    )
    rospy.spin()
