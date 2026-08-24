#!/bin/bash

#SBATCH --job-name=t8_figs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/t8/%x_%j.out
#SBATCH --error=logs/t8/%x_%j.err

mkdir -p logs/t8

module load Anaconda3
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate bigdata_lb

python3 "../scripts/t8_figs.py"
