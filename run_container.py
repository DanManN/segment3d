#!/usr/bin/env python3

# NOTE: please use only standard libraries
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

    docker_run_command = """
        docker run \
            --gpus all --rm -it --net=host \
            -e DISPLAY=${{DISPLAY}} -v /tmp:/tmp \
            segment3d:latest \
            /bin/bash -i -c \
            "source ~/.bashrc; \
            roscd segment3d; \
            rossetip $ROS_IP; rossetmaster {host}; \
            roslaunch segment3d segment3d.launch TODO:=todo"
            """.format(
        host=args.host,
    )
    print(docker_run_command)
    subprocess.call(docker_run_command, shell=True)
