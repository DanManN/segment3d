#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate ovir3d
cd /home/user/workspace/src/perception/src
python detic_service.py
