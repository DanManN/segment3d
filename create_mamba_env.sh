#!/usr/bin/env bash

# install script created based on instructions from https://github.com/shiyoung77/OVIR-3D.git
micromamba create -n ovir3d python=3.10 cuda \
	opencv open3d pytorch-gpu torchvision torchmetrics \
	mss timm dataclasses ftfy regex fasttext rospkg cupy \
	scikit-image scikit-learn lvis nltk numba GitPython einops

micromamba activate ovir3d

pip install \
	git+https://github.com/openai/CLIP.git \
	git+https://github.com/facebookresearch/detectron2.git
