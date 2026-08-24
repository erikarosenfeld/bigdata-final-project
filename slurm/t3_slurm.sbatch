#!/bin/bash

#SBATCH --job-name=t3_bigdata-taxi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/t3/%x_%j.out
#SBATCH --error=logs/t3/%x_%j.err

mkdir -p logs/t3

module load Anaconda3
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate bigdata_lb

python3 "../scripts/t3_exploratory_analysis.py"
