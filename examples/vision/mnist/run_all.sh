#!/usr/bin/env bash

tmux new-session -d -s regular_mnist '
python3 interpret.py --batch_size 2 --visible_device 1;
read;
'

tmux new-session -d -s pca_mnist '
python3 interpret.py --batch_size 2 --visible_device 2 --interpret_pca;
read;
'
