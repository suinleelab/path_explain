#!/usr/bin/env bash

tmux new-session -d -s s5_ih '
python3 get_performance.py --dataset simulated_5 --interaction_type integrated_hessians --visible_devices 2 --train_interaction_model;
read;
'
tmux new-session -d -s s5_eh '
python3 get_performance.py --dataset simulated_5 --interaction_type expected_hessians --visible_devices 2;
read;
'
tmux new-session -d -s s5_hs '
python3 get_performance.py --dataset simulated_5 --interaction_type hessians --visible_devices 2;
read;
'
tmux new-session -d -s s5_hti '
python3 get_performance.py --dataset simulated_5 --interaction_type hessians_times_inputs --visible_devices 2;
read;
'
tmux new-session -d -s s5_ss '
python3 get_performance.py --dataset simulated_5 --interaction_type shapley_sampling --visible_devices 2;
read;
'
tmux new-session -d -s s5_cd '
python3 get_performance.py --dataset simulated_5 --interaction_type contextual_decomposition --visible_devices 2;
read;
'
tmux new-session -d -s s5_nid '
python3 get_performance.py --dataset simulated_5 --interaction_type neural_interaction_detection --visible_devices 2;
read;
'

tmux new-session -d -s s10_ih '
python3 get_performance.py --dataset simulated_10 --interaction_type integrated_hessians --visible_devices 3 --train_interaction_model;
read;
'
tmux new-session -d -s s10_eh '
python3 get_performance.py --dataset simulated_10 --interaction_type expected_hessians --visible_devices 3;
read;
'
tmux new-session -d -s s10_hs '
python3 get_performance.py --dataset simulated_10 --interaction_type hessians --visible_devices 3;
read;
'
tmux new-session -d -s s10_hti '
python3 get_performance.py --dataset simulated_10 --interaction_type hessians_times_inputs --visible_devices 3;
read;
'
tmux new-session -d -s s10_ss '
python3 get_performance.py --dataset simulated_10 --interaction_type shapley_sampling --visible_devices 3;
read;
'
tmux new-session -d -s s10_cd '
python3 get_performance.py --dataset simulated_10 --interaction_type contextual_decomposition --visible_devices 3;
read;
'
tmux new-session -d -s s10_nid '
python3 get_performance.py --dataset simulated_10 --interaction_type neural_interaction_detection --visible_devices 3;
read;
'

tmux new-session -d -s s2_ih '
python3 get_performance.py --dataset higher_5 --interaction_type integrated_hessians --visible_devices 4 --train_interaction_model;
read;
'
tmux new-session -d -s s2_eh '
python3 get_performance.py --dataset higher_5 --interaction_type expected_hessians --visible_devices 4;
read;
'
tmux new-session -d -s s2_hs '
python3 get_performance.py --dataset higher_5 --interaction_type hessians --visible_devices 4;
read;
'
tmux new-session -d -s s2_hti '
python3 get_performance.py --dataset higher_5 --interaction_type hessians_times_inputs --visible_devices 4;
read;
'
tmux new-session -d -s s2_ss '
python3 get_performance.py --dataset higher_5 --interaction_type shapley_sampling --visible_devices 4;
read;
'
tmux new-session -d -s s2_cd '
python3 get_performance.py --dataset higher_5 --interaction_type contextual_decomposition --visible_devices 4;
read;
'
tmux new-session -d -s s2_nid '
python3 get_performance.py --dataset higher_5 --interaction_type neural_interaction_detection --visible_devices 4;
read;
'