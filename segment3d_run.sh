#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate ovir3d
cd $(rospack find segment3d)/src
python detic_service.py
