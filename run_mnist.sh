#!/usr/bin/env bash

tmux new-session -d -s mnist '
export CUDA_VISIBLE_DEVICES=0; 
python3 run_image.py --dataset mnist --background black;
python3 run_image.py --dataset mnist --background train_dist;
read
'

tmux new-session -d -s color_mnist '
export CUDA_VISIBLE_DEVICES=1; 
python3 run_image.py --dataset color_mnist --background black;
python3 run_image.py --dataset color_mnist --background train_dist;
read
'