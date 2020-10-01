#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:2
#SBATCH -p DGXq
#SBATCH -w node18

margin=(1.0 2.0)
dataset_name=WN18
bern=(False True)
epochs=120
batch_size=(512 1024 2048)
learning_rate=0.01
dim=64
lr_decay=1.8
norm=1

#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 1.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 1024 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 1.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 2048 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 1.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 1.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 1024 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 1.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 2048 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 1.0 --norm 1


#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 1024 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 2048 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 2.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 1024 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 2.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 2048 --init_lr 0.01 --lr_decay 1.8 --bern True --margin 2.0 --norm 1

#python3 ../main/main.py --dim 20 --bs 512 --init_lr 0.01 --lr_decay 1.0 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 20 --bs 1024 --init_lr 0.01 --lr_decay 1.0 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 20 --bs 2048 --init_lr 0.01 --lr_decay 1.0 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 20 --bs 4096 --init_lr 0.01 --lr_decay 1.0 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 512 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 1024 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 2048 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 8192 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.0 --norm 1

#python3 ../main/main.py --dim 50 --bs 4096 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 2.5 --norm 1

#python3 ../main/main.py --dim 50 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 2.5 --norm 1

#python3 ../main/main.py --dim 50 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 3.0 --norm 1

#python3 ../main/main.py --dim 64 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 3.0 --norm 1

python3 ../main/main.py --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 3.0 --norm 1

python3 ../main/main.py --dim 64 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 3.0 --norm 1

python3 ../main/main.py --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 4.0 --norm 1

python3 ../main/main.py --dim 64 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 4.0 --norm 1

python3 ../main/main.py --dim 64 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 4.0 --norm 1

python3 ../main/main.py --dim 64 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 4.0 --norm 1


python3 ../main/main.py --dim 96 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 3.0 --norm 1

python3 ../main/main.py --dim 96 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 3.0 --norm 1

python3 ../main/main.py --dim 96 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 4.0 --norm 1

python3 ../main/main.py --dim 96 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern False --margin 4.0 --norm 1

python3 ../main/main.py --dim 96 --bs 4096 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 4.0 --norm 1

python3 ../main/main.py --dim 96 --bs 8192 --init_lr 0.01 --lr_decay 1.4 --bern True --margin 4.0 --norm 1











rm "slurm-${SLURM_JOB_ID}.out"
