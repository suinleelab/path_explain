#!/usr/bin/env bash

tmux new-session -d -s regular_mnist '
python3 interpret.py --batch_size 1 --visible_device 0;
read;
'

tmux new-session -d -s pca_mnist '
python3 interpret.py --batch_size 1 --visible_device 1 --interpret_pca;
read;
'
