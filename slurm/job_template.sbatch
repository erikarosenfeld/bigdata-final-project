#!/bin/bash
#SBATCH --job-name=bigdata-taxi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Standard job template for every script in this project.
# Runs entirely on a compute node — immune to login-node disconnects/limits.
# Scripts use Dask's local scheduler within these resources (no nested
# SLURM job spin-up), so no partition-name guessing or job-submission
# fragility. Bump --mem/--time here if a specific task needs more.

mkdir -p logs

source /cvmfs/sling.si/modules/el7/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate bigdata

python3 "$@"
