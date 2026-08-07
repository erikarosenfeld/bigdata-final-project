#!/bin/bash
#SBATCH --job-name=bigdata-taxi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

source /cvmfs/sling.si/modules/el7/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate bigdata

python3 "$@"
