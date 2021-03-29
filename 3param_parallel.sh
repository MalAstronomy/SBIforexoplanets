#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=parallel
#SBATCH --array=1-100
#SBATCH --time=10-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2626 # megabytes
#
#SBATCH --output=/home/mvasist/scripts/output1/3param_out_100k_-%A-%a.txt

conda activate petitRT
cd /home/mvasist/scripts/parallel/
srun python 3param_Simulator.py $SLURM_ARRAY_TASK_ID
