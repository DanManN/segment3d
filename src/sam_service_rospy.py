#/usr/bin/env python3
import sys
import copy
import argparse

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

# ROS library
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse
import io
import json
import socket
import time

def main():
    server = 'localhost'
    port = '65372'
    using_server = True
    if using_server == True:
        server = '172.16.71.50'
        port = 8080

    print("Service starting JOE JOE JOE1")
    rospy.init_node('detic_service')
    #rospy.init_node('test', log_level=rospy.DEBUG)
    print("Service starting JOE JOE JOE2")

    parser = argparse.ArgumentParser(description="SAM")
    parser.add_argument("--depth_scale", type=float, default=1000)
    args = parser.parse_args()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(360)
    #client_socket.connect(('localhost', 65432))
    while True:
        try:
            # Attempt to connect to the server
            client_socket.connect((server, port))
            print("Connected to the server.")
            break  # Exit the loop if the connection is successful
        except ConnectionRefusedError:
            print("[JOE] This should take like 5 tries or something so just let it run. Connection refused. Server may not be running yet. Retrying in 3 seconds...")
            time.sleep(3)  # Wait before trying again
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(3)  # Wait before trying again

    print("Service starting JOE JOE JOE3")
    detic_service = SAMService(args, client_socket)
    print("Detic service started. Waiting for requests...")
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

    def __init__(self, args, client_socket):
        self.args = args
        self.detic_srv_name = 'detic_service'
        self.init_tracker_srv = rospy.Service(self.detic_srv_name, GetDeticResults, self.get_result)

        self.client_socket = client_socket

        self.bridge = CvBridge()
        # self.model = LangSAM('vit_b')  #,'./sam_vit_b_01ec64.pth')
        #print("before choosing model")
        #self.model = LangSAM()  #,'./sam_vit_b_01ec64.pth')

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

        if req.debug_mode:
            plt.imshow(rgb_im)
            plt.show()
            plt.imshow(depth_im)
            plt.show()

        bgr_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        print("sent name size")
        self.client_socket.sendall(len(target_name).to_bytes(8, 'big'))
        print("sent name")
        self.client_socket.sendall(target_name.encode('utf-8'))
        with io.BytesIO() as output:
            image_pil.save(output, format="PNG")  # You can choose JPEG or PNG
            image_data = output.getvalue()
        image_size = len(image_data)
        print("Sending msg to service of size")
        self.client_socket.sendall(image_size.to_bytes(8, 'big'))
        print("Sending msg to service of image")
        self.client_socket.sendall(image_data)
        
        # Wait for a response
        print("going to receieve response size")
        response_size_raw = self.client_socket.recv(4)
        response_size = int.from_bytes(response_size_raw, 'big')
        print("received response size", response_size)
        if response_size == 9:
            scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
            scene_pts = np.asarray(scene_pcd.points)
            scene_rgb = np.asarray(scene_pcd.colors)
            #_, table_indices = self.plane_detection_o3d(scene_pcd, inlier_thresh=0.01, visualize=req.debug_mode)

            scene_kdtree = KDTree(scene_pts)
            #target_pts = np.asarray(target_pcd.points)
            #_, target_indices = scene_kdtree.query(target_pts)
            target_mask_3d = np.zeros(scene_pts.shape[0], dtype=np.bool_)
            #target_mask_3d[target_indices] = True

            background_mask_3d = np.ones(scene_pts.shape[0], dtype=np.bool_)
            #background_mask_3d[table_indices] = False
            #background_mask_3d[target_indices] = False

            target_mask = np.zeros_like(depth_im)

            ret = GetDeticResultsResponse()
            ret.success = True
            # ret.pose = pose_msg  # this one is not used
            ret.points = Float32MultiArray(data=scene_pts.flatten().tolist())
            ret.colors = Float32MultiArray(data=scene_rgb.flatten().tolist())
            ret.target_mask = target_mask_3d.flatten().tolist() #all zeros
            ret.background_mask = background_mask_3d.flatten().tolist()
            ret.target_image_mask = self.bridge.cv2_to_imgmsg(target_mask.astype(np.uint8), encoding="mono8")
            return ret

        data = bytearray()  # To store the complete data as it arrives
        chunk_size = 4096   # You can adjust the chunk size if needed
        while len(data) < response_size:
            # Receive the next chunk
            chunk = self.client_socket.recv(min(chunk_size, response_size - len(data)))
            if not chunk:
                # If no data is received, it means the connection might be closed
                raise ConnectionError("Connection lost before receiving all data")
            data.extend(chunk)

        #data = self.client_socket.recv(response_size)
        result = json.loads(data.decode())
        print(f"Main: Received result from service")
        masks = np.array(result["masks"], dtype=np.float32)[0] #it is (1, x, y) so [0] turns to (x, y)
        print("number of target points", np.sum(masks))
        #boxes = result["boxes"]
        #logits = result["logits"]
        #masks, boxes, phrases, logits = self.model.predict(image_pil, target_name)

        # check if the returned values are empty
        if len(masks) == 0:
            ret = GetDeticResultsResponse()
            ret.success = False
            return ret
        #select_idx = np.argmax(logits)
        
        #target_mask = np.asarray(masks[select_idx])
        target_mask = masks

        if req.debug_mode:
            plt.imshow(target_mask)
            plt.show()

        #select_idx = np.argmax(logits)
        #mask = masks[select_idx]
        #box = boxes[select_idx]
        #if req.debug_mode:
        #    x1, y1, x2, y2 = [int(c) for c in box]
        #    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
        #    print("mask: ")
        #    print(mask)
        #    mask = np.asarray(mask).astype(int)
        #    print('mask shape:', mask.shape)
        #    print('image shape: ', image_cv.shape)
        #    color = np.array([0,255,0], dtype='uint8')
        #    masked_img = np.where(mask[...,None], color, image_cv)
        #    out = cv2.addWeighted(image_cv, 0.8, masked_img, 0.2,0)
        #    cv2.imshow("Image", out)
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()


        scene_pcd = create_pcd(depth_im, cam_intr, color_im=rgb_im)
        scene_pts = np.asarray(scene_pcd.points)
        scene_rgb = np.asarray(scene_pcd.colors)
        table_plane, table_indices = self.plane_detection_o3d(scene_pcd, inlier_thresh=0.01, visualize=req.debug_mode)

        masked_depth_im = depth_im * target_mask
        #print(type(masked_depth_im), type(depth_im), type(target_mask))
        #print(target_mask.shape, depth_im.shape)
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
