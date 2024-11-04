#!/usr/bin/env bash
cd $(rospack find segment3d)/src

if which conda
then
	eval "$(conda shell.bash hook)"
	# launch langsam
	conda activate langsam
	python langsam_pipe.py &
	# launch ros service
	conda activate ovir3d
	python sam_service_rospy.py
else
	eval "$(micromamba shell hook --shell bash)"
	# launch langsam
	micromamba activate langsam
	python langsam_pipe.py &
	# launch ros service
	micromamba activate ovir3d
	python sam_service_rospy.py
fi
