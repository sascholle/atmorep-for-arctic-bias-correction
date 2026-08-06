#!/bin/bash
#SBATCH --job-name=zarr_convert
#SBATCH --account=ab1412
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=logs/zarr_convert.%A_%a.out
#SBATCH --error=logs/zarr_convert.%A_%a.err
#SBATCH --array=0-11

source /work/ab1412/atmorep/pyenv/bin/activate

years=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021)
year=${years[$SLURM_ARRAY_TASK_ID]}

python /work/ab1412/atmorep/data/create_zarr.py $year