#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH -N1 --ntasks-per-node=1 -o log_run_disp_field.%j.out -e log_run_disp_field.%j.err

module load gcc/10
module load python/3
module load cuda
cd /mnt/home/pgrover/weighted_disp_field_model
python run_disp_field.py
