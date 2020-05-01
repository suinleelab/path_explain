#!/usr/bin/env bash

tmux new-session -d -s simulated_cossum '
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type integrated_hessians --train_interaction_model;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type expected_hessians;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type hessians;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type hessians_times_inputs;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type shapley_sampling;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type contextual_decomposition;
python3 get_performance.py --dataset simulated_cossum --visible_devices 0   --interaction_type neural_interaction_detection;
read;
'

tmux new-session -d -s simulated_maximum '
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type integrated_hessians --train_interaction_model;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type expected_hessians;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type hessians;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type hessians_times_inputs;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type shapley_sampling;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type contextual_decomposition;
python3 get_performance.py --dataset simulated_maximum --visible_devices 1   --interaction_type neural_interaction_detection;
read;
'

tmux new-session -d -s simulated_minimum '
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type integrated_hessians --train_interaction_model;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type expected_hessians;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type hessians;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type hessians_times_inputs;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type shapley_sampling;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type contextual_decomposition;
python3 get_performance.py --dataset simulated_minimum --visible_devices 2   --interaction_type neural_interaction_detection;
read;
'

tmux new-session -d -s simulated_multiply '
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type integrated_hessians --train_interaction_model;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type expected_hessians;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type hessians;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type hessians_times_inputs;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type shapley_sampling;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type contextual_decomposition;
python3 get_performance.py --dataset simulated_multiply --visible_devices 0   --interaction_type neural_interaction_detection;
read;
'

tmux new-session -d -s simulated_tanhsum '
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type integrated_hessians --train_interaction_model;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type expected_hessians;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type hessians;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type hessians_times_inputs;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type shapley_sampling;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type contextual_decomposition;
python3 get_performance.py --dataset simulated_tanhsum --visible_devices 1   --interaction_type neural_interaction_detection;
read;
'
