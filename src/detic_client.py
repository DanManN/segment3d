#!/home/lsy/anaconda3/envs/ovir3d/bin/python

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import rospy
import rosgraph
from sensor_msgs.msg import Image
from std_msgs.msg import String
from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse


def main():
    if not rosgraph.is_master_online():
        print("roscore is not running! Run roscore first!")
        exit(0)

    rospy.init_node("tracking_client", anonymous=True)

    service_name = "detic_service"
    rospy.loginfo(f"Waiting for {service_name} service...")
    rospy.wait_for_service(service_name)
    rospy.loginfo(f"Found {service_name} service.")

    try:
        req = GetDeticResultsRequest()
        #req.target_name = String("006_mustard_bottle")
        req.target_name = String("003_cracker_box")
        req.debug_mode = False

        detic_service= rospy.ServiceProxy(service_name, GetDeticResults)
        rospy.loginfo("Request sent. Waiting for response...")
        response: GetDeticResultsResponse = detic_service(req)
        rospy.loginfo(f"Got response. Request success: {response.success}")

        if response.success:
            pose_msg = response.pose
            #position, orientation = pose_msg.position, pose_msg.orientation
            #pose = np.eye(4)
            #pose[:3, 3] = np.array([position.x, position.y, position.z])
            #pose[:3, :3] = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_matrix()
            print(pose_msg)

            pts = np.array(response.points.data, dtype=np.float32).reshape(-1, 3)
            rgb = np.array(response.colors.data, dtype=np.float32).reshape(-1, 3)
            target_mask = np.array(response.target_mask)
            background_mask = np.array(response.background_mask)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            #pcd.paint_uniform_color([0, 0, 0])
            colors = np.asarray(pcd.colors)
            colors[target_mask] = [1, 0, 0]
            colors[background_mask] = [0, 0, 1]
            o3d.visualization.draw_geometries([pcd])

    except rospy.ServiceException as e:
        rospy.loginfo(f"Service call failed: {e}")


if __name__ == "__main__":
    main()
