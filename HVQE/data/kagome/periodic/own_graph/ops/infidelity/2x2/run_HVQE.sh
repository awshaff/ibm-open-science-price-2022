#!/bin/bash
#SBATCH -N 1
##SBATCH -n 4
#SBATCH -t 5-00:00:00
##SBATCH -p fat_soil_shared
##SBATCH --mem=50MB
##SBATCH --gres=gpu:1

for np in {1..5}
do
python3 ~/HVQE/HVQE.py $PWD 168 2 --cost_fn infidelity
sleep .1
done
wait
