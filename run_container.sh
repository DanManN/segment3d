#!/usr/bin/env bash
xhost +
docker run --gpus all --rm --net=host -it -e DISPLAY=${DISPLAY} -v /tmp:/tmp -v .:/mnt segment3d bash
