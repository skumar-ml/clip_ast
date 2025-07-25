#!/bin/bash

nohup python main.py train --root "~/data/caltech-101" --shots 8 --stage1_epochs 1 --stage2_epochs 30 --k 6 --out "ckpts/ast_caltech.pt" --device cuda:2 --eval_freq 5 > "logs/caltech_clip_ast_LRScheduler.log" 2>&1 &