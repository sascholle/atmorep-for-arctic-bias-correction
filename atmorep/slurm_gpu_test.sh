#!/bin/bash
#SBATCH --account=ab1412
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --error=logs/gpu-test-%j.err

module load python3/2023.01-gcc-11.2.0 nvhpc/23.9-gcc-11.2.0
source pyenv/bin/activate

echo "MASTER_ADDR: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Minimal GPU test script
cat <<EOF > gpu_test.py
import os
import torch
print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}, "
      f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID')}, "
      f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")
EOF

srun --label --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores python gpu_test.py