#!/usr/bin/env bash
if [ ! -f src/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth ]; then
	mkdir -p src/models
	wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O src/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi
docker build -t segment3d .
