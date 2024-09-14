#!/usr/bin/env bash
if which conda
then
	eval "$(conda shell.bash hook)"
	conda activate seg3d
else
	eval "$(micromamba shell hook --shell bash)"
	micromamba activate seg3d
fi
cd $(rospack find segment3d)/src
python sam_service.py
