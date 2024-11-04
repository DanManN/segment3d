#!/usr/bin/env bash

eval "$(micromamba shell hook --shell bash)"

# ros env
micromamba create -n ovir3d python=3.10 open3d
micromamba activate ovir3d
pip install rospkg catkin-pkg scikit-image scikit-learn torchmetrics cupy-cuda11x opencv-python matplotlib
pip uninstall em
pip install empy==3.3.4

# langsam env
micromamba create -n langsam python=3.11
micromamba activate langsam
pip install torch==2.4.1 torchvision==0.19.1
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
