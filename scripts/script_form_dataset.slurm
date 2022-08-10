#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=1 -o log_form_dataset.%j.out -e log_form_dataset.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/weighted_disp_field_model/
python form_dataset.py
