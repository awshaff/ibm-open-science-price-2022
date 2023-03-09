#!/bin/bash
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 5-00:00:00
#SBATCH -p fat_soil_shared
#SBATCH --mem=200MB
##SBATCH --gres=gpu:1

python3 ~/HVQE/ground_state.py $PWD 2
