#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=30thr
#SBATCH --time=5-01:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=2626 # megabytes
#
#SBATCH --output=/home/mvasist/scripts/output1/eval_paul_MCMC_200w500it_30th_TintLkIRLg.txt

conda activate petitRT
cd /home/mvasist/scripts/parallel/
srun python eval_paul_MCMC.py 30
