#!/bin/bash

export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

source /home/gabriel/miniconda3/etc/profile.d/conda.sh
conda activate e4e
cd /home/gabriel/Documents/SYNTHETIC/face_generator

sleep 5

# v4l2-ctl -d /dev/video4 --list-ctrls
# v4l2-ctl --list-devices
# v4l2-ctl -d /dev/video0 --set-ctrl exposure_auto=3
# v4l2-ctl -d /dev/video0 --set-ctrl exposure_absolute=256

python main.py

exec bash;
