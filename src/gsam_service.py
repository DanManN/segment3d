#!/usr/bin/env python3
import sys
import copy
import argparse
import io
import json

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import zmq
import time

#from lang_sam import LangSAM

# ROS library
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse



def main():
    print("Service starting")
    rospy.init_node('detic_service', log_level=rospy.DEBUG)
    #rospy.init_node('test', log_level=rospy.DEBUG)
    print("Node initialized")

    parser = argparse.ArgumentParser(description="SAM")
    parser.add_argument("--depth_scale", type=float, default=1000)
    args = parser.parse_known_args()

    SAMService(args[0])
    print("Gsam service started. Waiting for requests...")
    rospy.spin()



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



class SAMService:

    def __init__(self, args):
        self.args = args
        self.detic_srv_name = 'detic_service'
        self.init_tracker_srv = rospy.Service(self.detic_srv_name, GetDeticResults, self.get_result)

        self.bridge = CvBridge()
        # self.model = LangSAM('vit_b')  #,'./sam_vit_b_01ec64.pth')
        #print("before choosing model")
        #self.model = LangSAM()  #,'./sam_vit_b_01ec64.pth')
        
        #Create the zmq socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ) #Make sure to use REQ
        print("trying to connect to server gsam port")
        #socket.connect("tcp://:8091")
        self.socket.connect("tcp://0.0.0.0:8091")
        print("Client connected")
        self.it = 0

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

        #if visualize:
        #    plane_frame_vis = DeticService.generate_coordinate_frame(plane_frame, scale=0.05)
        #    cam_frame_vis = DeticService.generate_coordinate_frame(np.eye(4), scale=0.05)
        #    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_frame_vis, cam_frame_vis])

        return plane_frame, inliers

    def get_result(self, req: GetDeticResultsRequest):
        target_name = req.target_name.data
        print(f"{target_name = }")

        camera_info = req.cam_info
        cam_intr = np.array(camera_info.K).reshape((3, 3))
        rgb_msg = req.color_img
        depth_msg = req.depth_img
        rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1').astype(np.float32) / self.args.depth_scale
        image_pil = Image.fromarray(rgb_im)
        print("Image Info:")
        print(f"Format: {image_pil.format}")          # Image format (e.g., JPEG, PNG)
        print(f"Size: {image_pil.size}")              # Image size (width, height)
        print(f"Mode: {image_pil.mode}")              # Image mode (e.g., RGB, RGBA, L)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        if req.debug_mode:
            plt.imshow(rgb_im)
            plt.show()
            plt.imshow(depth_im)
            plt.show()

        bgr_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        #-------------Here we do the model predict which first sends the inputs from req and gets the results then fixes them up to the right format:--------------------------
        #masks, boxes, phrases, logits = self.model.predict(image_pil, target_name)
        print("target name", target_name)
        send_string = target_name
        image_stream = io.BytesIO()
        image_pil.save(image_stream, format="JPEG")  #Save image in JPEG format
        image_data = image_stream.getvalue()  #Get byte data
        #socket.send(image_data) #Send the serialized image

        t0 = time.time()
        message = [send_string.encode(), image_data]
        self.socket.send_multipart(message) #SENDING text and image
        print("sent text and image data from gsam")

        #To receive it blocks until it receives
        message_parts = self.socket.recv_multipart() #RECEIVING metadata for masks and mask bytes
        metadata = message_parts[0].decode()  # Decode as string
        metadata = json.loads(metadata)
        print(f"Received metadata from gsam server: {metadata}")
        #Bytes
        mask_bytes = message_parts[1]
        if metadata["shape"] == [0]:  # Adjust condition based on your metadata structure
            print("-----------No masks returned-------------")
            ret = GetDeticResultsResponse()
            ret.success = True
            scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
            scene_pts = np.asarray(scene_pcd.points)
            scene_rgb = np.asarray(scene_pcd.colors)
            ret.points = Float32MultiArray(data=scene_pts.flatten().tolist())
            ret.colors = Float32MultiArray(data=scene_rgb.flatten().tolist())
            # ret.target_mask = target_mask_3d.flatten().tolist()
            # ret.background_mask = background_mask_3d.flatten().tolist()
            # ret.target_image_mask = self.bridge.cv2_to_imgmsg(target_mask.astype(np.uint8), encoding="mono8")
            return ret
        else:
            # Deserialize the mask
            masks = np.frombuffer(mask_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])
            print("masks shape", masks.shape) #It is a number of masks by image size np array
        print("time perception took", time.time() - t0)

        image_cv2 = np.array(image_pil)
        print("masks shape", masks.shape)
        #for mask in masks:
        if True:
            mask = masks[0]
            mask = (mask > 0.5).astype(np.uint8)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask = np.zeros_like(image_cv2, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            image_cv2 = cv2.bitwise_and(image_cv2, image_cv2, mask=1-mask)  # Keep original image where mask is 0
            image_cv2 = cv2.add(image_cv2, colored_mask)  # Add colored mask only where mask is 1
        cv2.imwrite(f"/tmp/tmp_joe/gsamservice_output_{self.it}.jpg", image_cv2)
        self.it += 1

        #Choose one mask as the mask we are going to use, or aggregate all the masks together that we found. 
        #Is there a way to see the confidence for it?
        target_mask = masks[0]#np.asarray(masks[select_idx]) #Here is where we use the masks

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

        ret = GetDeticResultsResponse()
        ret.success = True
        # ret.pose = pose_msg  # this one is not used
        ret.points = Float32MultiArray(data=scene_pts.flatten().tolist())
        ret.colors = Float32MultiArray(data=scene_rgb.flatten().tolist())
        ret.target_mask = target_mask_3d.flatten().tolist()
        ret.background_mask = background_mask_3d.flatten().tolist()
        ret.target_image_mask = self.bridge.cv2_to_imgmsg(target_mask.astype(np.uint8), encoding="mono8")
        return ret



if __name__ == '__main__':
    main()
