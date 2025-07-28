#!/bin/bash

nohup python zero_shot_eval.py --root "~/data/caltech-101" --dataset caltech101 --batch_size 64 --device cuda:2 > "logs/caltech_clip_zero_shot.log" 2>&1 &