#!/usr/bin/env python3

# NOTE: please use only standard libraries
import os
import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c", type=str, help="gripper config file"
    # )
    parser.add_argument(
        "-host", type=str, default="localhost", help="host name or ip-address"
    )
    parser.add_argument(
        "launch_args",
        nargs=argparse.REMAINDER,
        help="launch args in ros style e.g. foo:=var",
    )
    args = parser.parse_args()

    # assert args.c is not None

    print("This is just for running locally it opens up the groundedsam from the Grounded-SAM-iterative repo and runs it")
    docker_run_command = """
        docker run \
            --gpus all --rm -it -p 8091:8091 --net=host \
            -e DISPLAY=${{DISPLAY}} -v /tmp:/tmp \
            groundedsam:stream \
            /bin/bash -i -c \
            "source ~/.bashrc; \
            roscd segment3d; \
            export ROS_IP={ip}; export ROS_MASTER={host}; export ROS_MASTER_URI=http://{host}:11311; \
            python server_gsam.py"
            """.format(
        ip=os.environ['ROS_IP'] if 'ROS_IP' in os.environ else '127.0.0.1',
        host=args.host,
    )
    #roslaunch segment3d segment3d_sam.launch TODO:=todo"
    print(docker_run_command)
    subprocess.call(docker_run_command, shell=True)
