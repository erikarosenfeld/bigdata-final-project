#!/bin/bash

#SBATCH --job-name=t7_xgb_noaug
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/t7/xgb/%x_%j.out
#SBATCH --error=logs/t7/xgb/%x_%j.err

mkdir -p logs/t7

module load Anaconda3
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate bigdata_lb

python3 "../scripts/t7_xgb_noaug.py" --workers 4 --mem_limit 2048

sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State