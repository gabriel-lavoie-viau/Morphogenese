#!/bin/bash
export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source /home/gabriel/miniconda3/etc/profile.d/conda.sh
conda activate e4e
cd /home/gabriel/Documents/SYNTHETIC/face_generator
sleep 10
v4l2-ctl -d /dev/video0 --set-ctrl focus_auto=0
sleep 5
v4l2-ctl -d /dev/video0 --set-ctrl focus_absolute=275
sleep 5
python main.py --monitor 0 --tweak_mode 0
exec bash;
