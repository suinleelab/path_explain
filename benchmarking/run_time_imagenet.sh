#!/usr/bin/env bash

tmux new-session -d -s vgg16 '
python3 time_imagenet.py --model vgg16 --type attribution --visible_devices 0;
python3 time_imagenet.py --model vgg16 --type single_interaction --visible_devices 0;
python3 time_imagenet.py --model vgg16 --type jacobian_interactions --visible_devices 0;
python3 time_imagenet.py --model vgg16 --type loop_interactions --visible_devices 0;
read;
'

tmux new-session -d -s inception_v3 '
python3 time_imagenet.py --model inception_v3 --type attribution --visible_devices 1;
python3 time_imagenet.py --model inception_v3 --type single_interaction --visible_devices 1;
python3 time_imagenet.py --model inception_v3 --type jacobian_interactions --visible_devices 1;
python3 time_imagenet.py --model inception_v3 --type loop_interactions --visible_devices 1;
read;
'

tmux new-session -d -s mobilenet_v2 '
python3 time_imagenet.py --model mobilenet_v2 --type attribution --visible_devices 2;
python3 time_imagenet.py --model mobilenet_v2 --type single_interaction --visible_devices 2;
python3 time_imagenet.py --model mobilenet_v2 --type jacobian_interactions --visible_devices 2;
python3 time_imagenet.py --model mobilenet_v2 --type loop_interactions --visible_devices 2;
read;
'
