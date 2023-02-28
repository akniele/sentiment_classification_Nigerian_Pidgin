#!/bin/bash
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 0:20:00 
#SBATCH -J pcm_final 
#SBATCH --gres=gpu:1

python eval_model.py
