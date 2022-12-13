#!/bin/bash

export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /home/gabriel/miniconda3/etc/profile.d/conda.sh
conda activate magenta
cd /home/gabriel/Documents/SYNTHETIC/melody_generator

sleep 5

python main.py

exec bash;
