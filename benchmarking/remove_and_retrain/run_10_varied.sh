#!/usr/bin/env bash

tmux new-session -d -s s10_var_ih '
python3 get_performance.py --dataset simulated_10_varied --interaction_type integrated_hessians --visible_devices 0 --train_interaction_model;
read;
'
tmux new-session -d -s s10_var_eh '
sleep 2;
python3 get_performance.py --dataset simulated_10_varied --interaction_type expected_hessians --visible_devices 1;
read;
'
tmux new-session -d -s s10_var_hs '
sleep 4;
python3 get_performance.py --dataset simulated_10_varied --interaction_type hessians --visible_devices 2;
read;
'
tmux new-session -d -s s10_var_hti '
sleep 6;
python3 get_performance.py --dataset simulated_10_varied --interaction_type hessians_times_inputs --visible_devices 3;
read;
'
tmux new-session -d -s s10_var_ss '
sleep 8;
python3 get_performance.py --dataset simulated_10_varied --interaction_type shapley_sampling --visible_devices 4;
read;
'
tmux new-session -d -s s10_var_cd '
sleep 10;
python3 get_performance.py --dataset simulated_10_varied --interaction_type contextual_decomposition --visible_devices 5;
read;
'
tmux new-session -d -s s10_var_nid '
sleep 12;
python3 get_performance.py --dataset simulated_10_varied --interaction_type neural_interaction_detection --visible_devices 6;
read;
'
