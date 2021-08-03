#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=coverage
#SBATCH --time=5-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=2626 # megabytes
#
#SBATCH --output=/home/mvasist/scripts/covout.txt

conda activate petitRT
cd /home/mvasist/scripts/
srun python coverage_cal.py 1