#!/bin/bash
#SBATCH --job-name=cds_download
#SBATCH --account=ab1412
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/cds_download.%j.out
#SBATCH --error=logs/cds_download.%j.err

# Load your Python environment (adjust if you use modules or conda)
source /work/ab1412/atmorep/pyenv/bin/activate

# Make sure .cdsapirc exists in your home directory

# Run your CDS API download script
python /work/ab1412/atmorep/data/cdsapi_t2m_download_script.py
