#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=parallel
#SBATCH --array=1-100
#SBATCH --time=10-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2626 # megabytes
#
#SBATCH --output=/home/mvasist/scripts/output1/4param_out_1M_-%A-%a.txt

conda activate petitRT
cd /home/mvasist/scripts/parallel/
srun python Simulator.py $SLURM_ARRAY_TASK_ID
