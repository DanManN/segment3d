#/usr/bin/env python3
import sys
import copy
import argparse

# ROS library
import rospy


def main():
    print("[TEST] Service starting JOE JOE JOE1")
    rospy.init_node('detic_service', log_level=rospy.DEBUG)
    #rospy.init_node('test', log_level=rospy.DEBUG)
    print("Service starting JOE JOE JOE2")

    #parser = argparse.ArgumentParser(description="SAM")
    #parser.add_argument("--depth_scale", type=float, default=1000)
    #args = parser.parse_args()

    #print("Service starting JOE JOE JOE3")
    #detic_service = SAMService(args)
    #print("Detic service started. Waiting for requests...")
    rospy.spin()

if __name__ == '__main__':
    main()
