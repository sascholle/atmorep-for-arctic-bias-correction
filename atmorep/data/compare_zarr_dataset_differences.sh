#!/bin/bash
#SBATCH --account=ab1412
#SBATCH --partition=compute
#SBATCH --job-name=cmp_zarr
#SBATCH --output=/scratch/a/a270277/atmorep/logs/compare_zarr_%j.out
#SBATCH --error=/scratch/a/a270277/atmorep/logs/compare_zarr_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=0

source /work/ab1412/atmorep/pyenv/bin/activate

export SRC="/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_copy.zarr"
export DST="/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m_new.zarr"
export OUT="/scratch/a/a270277/atmorep/compare_zarr_diffs_full.csv.gz"

# run (tweak --max-diffs-per-array to limit output during test)
python /work/ab1412/atmorep/data/compare_zarr_dataset_differences.py \
  "$SRC" "$DST" \
  --out "$OUT" \
  --rtol 1e-6 --atol 1e-8 \
  --progress-every 500 \
  --max-diffs-per-array 0

# end