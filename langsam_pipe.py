#/usr/bin/env python3
import sys
import copy
#import argparse

#import cv2
import numpy as np
#import open3d as o3d
from PIL import Image
#from scipy.spatial import KDTree
#from matplotlib import pyplot as plt

from lang_sam import LangSAM

# ROS library
#from cv_bridge import CvBridge
#from geometry_msgs.msg import Pose
#from std_msgs.msg import Float32MultiArray
#from sensor_msgs.msg import CameraInfo
#from segment3d.srv import GetDeticResults, GetDeticResultsRequest, GetDeticResultsResponse
import io
import json
import socket


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 65432))
    server_socket.listen()
    print("Service is listening on port 65432...")
    model = LangSAM()
    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")

            # Receive the length of the name (8 bytes)
            name_length_data = conn.recv(8)
            name_length = int.from_bytes(name_length_data, 'big')
            print(f"Expecting to receive name of length: {name_length} characters")

            # Receive the name
            name_data = conn.recv(name_length)
            name = name_data.decode()
            print(f"Received name: {name}")

            # Receive the size of the size of the image
            data = conn.recv(8)  # Expecting 8 bytes for the size of the image
            if not data:
                break

            image_size = int.from_bytes(data, 'big')  # Convert bytes to integer
            print(f"Expecting to receive image of size: {image_size} bytes")

            # Receive the actual image data
            image_data = bytearray()
            while len(image_data) < image_size:
                packet = conn.recv(4096)  # Adjust size as necessary
                if not packet:
                    break
                image_data.extend(packet)

            # Convert bytes back to an image
            image = Image.open(io.BytesIO(image_data))
            print(f"Received image with size: {image.size}")

            # Here you can process the image (e.g., run model prediction)
            masks, boxes, phrases, logits = model.predict(image, name)
            response = {
                "masks": masks,
                "boxes": boxes,
                "logits": logits
            }

            json_response = json.dumps(response).encode()
            response_length = len(json_response)

            # Send the length of the response first
            print("send back result full size")
            conn.sendall(response_length.to_bytes(4, 'big'))
            # Send the response back
            print("sent back result")
            conn.sendall(json_response)


if __name__ == '__main__':
    main()
