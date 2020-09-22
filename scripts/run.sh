#!/bin/bash

margin=(1.0 2.0)
dataset_name=WN18
bern=(True False)
epochs=120
batch_size=(512 2048)
learning_rate=0.01
dim=64
margin=1.0
lr_decay=1.7
norm=1

python3 ../main/main.py --dim 64 --bs 2048 --init_lr 0.01 --lr_decay 1.7 --bern False --margin 1.0 --norm 1 
