#!/bin/bash

# Sweep over lmds
# for lmbd in 1.0 2.0 5.0 10.0; do
#     nohup python main.py train --root "~/data/caltech-101" --shots 8 --stage1_epochs 1 --stage2_epochs 30 --k 6 --out "ckpts/ast_caltech.pt" --device cuda:3 --eval_freq 5 --lmbd $lmbd $lmbd $lmbd > "logs/caltech_LRScheduler_lmbd_$lmbd.log" 2>&1 &
#     wait $!
# done

# Sweep over lr
# for lr in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8; do
#     nohup python main.py train --root "~/data/caltech-101" --shots 8 --stage1_epochs 1 --stage2_epochs 30 --k 6 --out "ckpts/ast_caltech.pt" --device cuda:3 --eval_freq 5 --stage2_lr $lr --lmbd 0.0 0.0 0.0 > "logs/caltech_LRScheduler_lmbd_0_lr_$lr.log" 2>&1 &
#     wait $!
# done

# Random selection experiment
for random_seed in 42 43 44; do
    nohup python main.py train --root "~/data/caltech-101" --shots 8 --stage1_epochs 1 --stage2_epochs 30 --k 6 --out "ckpts/ast_caltech.pt" --device cuda:1 --eval_freq 5 --random_selection --random_seed $random_seed --lmbd 0.0 0.0 0.0 > "logs/caltech_RandomSelection_seed_$random_seed.log" 2>&1 &
    wait $!
done