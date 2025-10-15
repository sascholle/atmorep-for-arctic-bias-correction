#!/bin/bash
#SBATCH --job-name=examine_t2m
#SBATCH --account=ab1412
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/work/ab1412/atmorep/data/logs/examine_t2m_%j.out
#SBATCH --error=/work/ab1412/atmorep/data/logs/examine_t2m_%j.err

# Load environment
source /work/ab1412/atmorep/pyenv/bin/activate

# Run the analysis
python -u /work/ab1412/atmorep/data/examine_corrected_t2m.py