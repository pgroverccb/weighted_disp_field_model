#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=1 -o log_compute_disp_field.%j.out -e log_compute_disp_field.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/
python compute_disp_field.py
