#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH -N1 --ntasks-per-node=1 -o log_lineage_pred_disp_field_sliding.%j.out -e log_lineage_pred_disp_field_sliding.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/weighted_disp_field_model/
python lineage_pred_disp_field_sliding.py
