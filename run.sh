#!/usr/bin/env bash
set -e
python train.py --epochs 1 --train_size 1000 --eval_size 200 --train_bs 2 --eval_bs 2
