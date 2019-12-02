#!/usr/bin/env bash

tmux new-session -d -s run0 'python3 train.py --index 0; read;'
tmux new-session -d -s run1 'python3 train.py --index 1; read;'
tmux new-session -d -s run2 'python3 train.py --index 2; read;'
tmux new-session -d -s run3 'python3 train.py --index 3; read;'
tmux new-session -d -s run4 'python3 train.py --index 4; read;'
tmux new-session -d -s run5 'python3 train.py --index 5; read;'
tmux new-session -d -s run6 'python3 train.py --index 6; read;'
tmux new-session -d -s run7 'python3 train.py --index 7; read;'
tmux new-session -d -s run8 'python3 train.py --index 8; read;'
tmux new-session -d -s run9 'python3 train.py --index 9; read;'
