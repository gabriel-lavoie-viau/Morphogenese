#!/bin/bash  

# gnome-terminal --tab -e /home/gabriel/Documents/bash_scripts/start_melody_generator.sh

# conda init bash; 
# eval "$(conda shell.bash hook)"
# source /home/gabriel/anaconda3/bin/activate magenta

source /home/gabriel/anaconda3/etc/profile.d/conda.sh
conda activate magenta
cd /media/gabriel/Extra/SYNTHETIC/melody_generator/python
python3 main.py

exec bash;
