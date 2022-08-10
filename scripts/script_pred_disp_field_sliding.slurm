#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH -N1 --ntasks-per-node=1 -o log_pred_disp_field_sliding.%j.out -e log_pred_disp_field_sliding.%j.err

module load gcc/10
module load python/3
module load cuda
cd /mnt/home/pgrover/weighted_disp_field_model/
python pred_disp_field_sliding.py
