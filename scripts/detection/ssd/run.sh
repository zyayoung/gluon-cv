#!/bin/bash
#SBATCH -J ssd
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
. ~/.bashrc
module load cuda/9.2
#1568804161
module load anaconda3/2019.07
#1568804205
conda activate mxnet
python train_ssd_nem.py --gpus 0,1 --network resnet50_v1 --data-shape 512 --val-interval 5 --lr 0.0005 --lr-decay-epoch 100,200

