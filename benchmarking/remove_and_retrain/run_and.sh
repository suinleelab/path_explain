#!/usr/bin/env bash

tmux new-session -d -s s5and_ih '
python3 get_performance.py --dataset simulated_5_and --interaction_type integrated_hessians --visible_devices 6 --train_interaction_model;
read;
'
tmux new-session -d -s s5and_eh '
python3 get_performance.py --dataset simulated_5_and --interaction_type expected_hessians --visible_devices 6;
read;
'
tmux new-session -d -s s5and_hs '
python3 get_performance.py --dataset simulated_5_and --interaction_type hessians --visible_devices 6;
read;
'
tmux new-session -d -s s5and_hti '
python3 get_performance.py --dataset simulated_5_and --interaction_type hessians_times_inputs --visible_devices 6;
read;
'
tmux new-session -d -s s5and_ss '
python3 get_performance.py --dataset simulated_5_and --interaction_type shapley_sampling --visible_devices 6;
read;
'
tmux new-session -d -s s5and_cd '
python3 get_performance.py --dataset simulated_5_and --interaction_type contextual_decomposition --visible_devices 6;
read;
'
tmux new-session -d -s s5and_nid '
python3 get_performance.py --dataset simulated_5_and --interaction_type neural_interaction_detection --visible_devices 6;
read;
'

