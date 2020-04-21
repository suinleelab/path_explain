#!/usr/bin/env bash

tmux new-session -d -s hd_ih '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type integrated_hessians --visible_devices 0 --train_interaction_model;
read;
'
tmux new-session -d -s hd_eh '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type expected_hessians --visible_devices 0;
read;
'
tmux new-session -d -s hd_hs '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type hessians --visible_devices 0;
read;
'
tmux new-session -d -s hd_hti '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type hessians_times_inputs --visible_devices 0;
read;
'
tmux new-session -d -s hd_ss '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type shapley_sampling --visible_devices 0;
read;
'
tmux new-session -d -s hd_cd '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type contextual_decomposition --visible_devices 0;
read;
'
tmux new-session -d -s hd_nid '
python3 get_performance.py --dataset heart_disease --epochs 10 --interaction_type neural_interaction_detection --visible_devices 0;
read;
'

tmux new-session -d -s pulsar_ih '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type integrated_hessians --visible_devices 1 --train_interaction_model;
read;
'
tmux new-session -d -s pulsar_eh '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type expected_hessians --visible_devices 1;
read;
'
tmux new-session -d -s pulsar_hs '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type hessians --visible_devices 1;
read;
'
tmux new-session -d -s pulsar_hti '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type hessians_times_inputs --visible_devices 1;
read;
'
tmux new-session -d -s pulsar_ss '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type shapley_sampling --visible_devices 1;
read;
'
tmux new-session -d -s pulsar_cd '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type contextual_decomposition --visible_devices 1;
read;
'
tmux new-session -d -s pulsar_nid '
python3 get_performance.py --dataset pulsar --epochs 10 --interaction_type neural_interaction_detection --visible_devices 1;
read;
'

