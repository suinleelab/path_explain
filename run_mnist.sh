#!/usr/bin/env bash

tmux new-session -d -s mnist_black '
export CUDA_VISIBLE_DEVICES=0; 
python3 run_image.py --dataset mnist --background black;
read
'

tmux new-session -d -s color_mnist_black '
export CUDA_VISIBLE_DEVICES=1; 
python3 run_image.py --dataset color_mnist --background black;
read
'

tmux new-session -d -s mnist_train_dist '
export CUDA_VISIBLE_DEVICES=2; 
python3 run_image.py --dataset mnist --background train_dist;
read
'

tmux new-session -d -s color_mnist_train_dist '
export CUDA_VISIBLE_DEVICES=3; 
python3 run_image.py --dataset color_mnist --background train_dist;
read
'