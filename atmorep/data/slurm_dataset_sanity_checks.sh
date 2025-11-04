#!/bin/bash
#SBATCH --job-name=climatology_plots
#SBATCH --account=ab1412
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --output=/work/ab1412/atmorep/data/logs/seasonality_%j.out
#SBATCH --error=/work/ab1412/atmorep/data/logs/seasonality_%j.err

source /work/ab1412/atmorep/pyenv/bin/activate

python -u /work/ab1412/atmorep/data/dataset_sanity_checks.py