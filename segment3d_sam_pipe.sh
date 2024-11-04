#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
cd $(rospack find segment3d)/src

# launch langsam
conda activate langsam
python langsam_pipe.py &

# launch ros service
conda activate ovir3d
python sam_service_rospy.py
