#!/usr/bin/env python3

# NOTE: please use only standard libraries
import os
import argparse
import subprocess
from pathlib import Path
import sys

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
    trust_docker = len(sys.argv) > 1

    print("This is just for running locally it opens up the groundedsam from the Grounded-SAM-iterative repo and runs it")
    if trust_docker:
        docker_run_command = """
            docker run \
                --gpus all --rm -it -p 8091:8091 --net=host \
                -e DISPLAY=${{DISPLAY}} -v /tmp:/tmp \
                groundedsam:latest \
                /bin/bash -i -c \
                "source ~/.bashrc; \
                export ROS_IP={ip}; export ROS_MASTER={host}; export ROS_MASTER_URI=http://{host}:11311; \
            python server_gsam.py"
                """.format(
            ip=os.environ['ROS_IP'] if 'ROS_IP' in os.environ else '127.0.0.1',
            host=args.host,
        )
    else:
        docker_run_command = """
            docker run \
                --gpus all --rm -it -p 8091:8091 --net=host \
                -e DISPLAY=${{DISPLAY}} -v /tmp:/tmp \
                -v $(rospack find segment3d)/cache:/root/.cache \
                -v $(rospack find segment3d)/../Grounded-SAM-2-iterative:/home/appuser/Grounded-SAM-2 \
                groundedsam:latest \
                /bin/bash -i -c \
                "source ~/.bashrc; \
                export ROS_IP={ip}; export ROS_MASTER={host}; export ROS_MASTER_URI=http://{host}:11311; \
            python server_gsam.py"
                """.format(
            ip=os.environ['ROS_IP'] if 'ROS_IP' in os.environ else '127.0.0.1',
            host=args.host,
        )
    #roslaunch segment3d segment3d_sam.launch TODO:=todo"
    print(docker_run_command)
    subprocess.call(docker_run_command, shell=True)

#-v $(rospack find segment3d)/Grounded-SAM-2-iterative:/home/appuser/Grounded-SAM-2 \
#-v $(rospack find segment3d)/cache:/root/.cache \
#These are for if you have the working version as the repo you have not on the docker
