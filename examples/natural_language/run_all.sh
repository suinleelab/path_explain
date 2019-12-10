#!/usr/bin/env bash

tmux new-session -d -s stsb '
export CUDA_VISIBLE_DEVICES=0;
python3 retrain_bert.py --task sts-b --epochs 5 --force_train;
read;
'

# tmux new-session -d -s sst2 '
# export CUDA_VISIBLE_DEVICES=1;
# python3 retrain_bert.py --task sst-2 --epochs 1;
# read;
# '