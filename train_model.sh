#!/bin/bash
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 01:30:00 
#SBATCH -J pcm_final 
#SBATCH --gres=gpu:1

SEED=$1  # the random seed to be used
PRE=$2  # the pre-trained model to be used ("mono" or "multi")
FINE=$3  # the fine-tuning languages to be used ("pcm" or "igh,ha" or "en")
# don't put quotation marks around the command line arguments!

python train_model.py $SEED $PRE $FINE 
