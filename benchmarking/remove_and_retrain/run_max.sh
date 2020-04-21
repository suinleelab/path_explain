#!/usr/bin/env bash

tmux new-session -d -s s5max_ih '
python3 get_performance.py --dataset simulated_5_max --interaction_type integrated_hessians --visible_devices 5 --train_interaction_model;
read;
'
tmux new-session -d -s s5max_eh '
python3 get_performance.py --dataset simulated_5_max --interaction_type expected_hessians --visible_devices 5;
read;
'
tmux new-session -d -s s5max_hs '
python3 get_performance.py --dataset simulated_5_max --interaction_type hessians --visible_devices 5;
read;
'
tmux new-session -d -s s5max_hti '
python3 get_performance.py --dataset simulated_5_max --interaction_type hessians_times_inputs --visible_devices 5;
read;
'
tmux new-session -d -s s5max_ss '
python3 get_performance.py --dataset simulated_5_max --interaction_type shapley_sampling --visible_devices 5;
read;
'
tmux new-session -d -s s5max_cd '
python3 get_performance.py --dataset simulated_5_max --interaction_type contextual_decomposition --visible_devices 5;
read;
'
tmux new-session -d -s s5max_nid '
python3 get_performance.py --dataset simulated_5_max --interaction_type neural_interaction_detection --visible_devices 5;
read;
'

