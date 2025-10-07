#!/bin/bash -x
#SBATCH --account=ab1412
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=34 # changed from 34
#SBATCH --gres=gpu:1
#SBATCH --chdir=.
#SBATCH --output=logs/nice_batch_wrapper%j.out
#SBATCH --error=logs/nice_batch_wrapper%j.err

dataset_length=2099  # or detect this in Python and write to a file
chunk_size=72
num_chunks=$(( (dataset_length + chunk_size - 1) / chunk_size ))

for chunk in $(seq 0 $((num_chunks-1))); do
    start_idx=$((chunk * chunk_size))
    end_idx=$(( (chunk+1) * chunk_size ))
    if [ $end_idx -gt $dataset_length ]; then
        end_idx=$dataset_length
    fi
    sbatch slurm_evaluate_nice.sh $start_idx $end_idx
done