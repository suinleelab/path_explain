#!/usr/bin/env bash

tmux new-session -d -s s5and_ih '
python3 get_performance.py --dataset simulated_5_and --interaction_type integrated_hessians --visible_devices 0 --train_interaction_model;
read;
'
tmux new-session -d -s s5and_eh '
sleep 2;
python3 get_performance.py --dataset simulated_5_and --interaction_type expected_hessians --visible_devices 1;
read;
'
tmux new-session -d -s s5and_hs '
sleep 4;
python3 get_performance.py --dataset simulated_5_and --interaction_type hessians --visible_devices 2;
read;
'
tmux new-session -d -s s5and_hti '
sleep 6;
python3 get_performance.py --dataset simulated_5_and --interaction_type hessians_times_inputs --visible_devices 3;
read;
'
tmux new-session -d -s s5and_ss '
sleep 8;
python3 get_performance.py --dataset simulated_5_and --interaction_type shapley_sampling --visible_devices 4;
read;
'
tmux new-session -d -s s5and_cd '
sleep 10;
python3 get_performance.py --dataset simulated_5_and --interaction_type contextual_decomposition --visible_devices 5;
read;
'
tmux new-session -d -s s5and_nid '
sleep 12;
python3 get_performance.py --dataset simulated_5_and --interaction_type neural_interaction_detection --visible_devices 6;
read;
'

