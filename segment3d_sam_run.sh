#!/usr/bin/env bash
if which conda
then
	eval "$(conda shell.bash hook)"
	conda activate ovir3d
else
	eval "$(micromamba shell hook --shell bash)"
	micromamba activate ovir3d
fi
cd $(rospack find segment3d)/src
python sam_service.py
