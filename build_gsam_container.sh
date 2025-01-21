#!/usr/bin/env bash
if [ ! -d Grounded-SAM-2-iterative ]
then
	git clone https://github.com/JoeDoerr/Grounded-SAM-2-iterative.git
fi
docker build -t groundedsam:stream ./Grounded-SAM-2-iterative
