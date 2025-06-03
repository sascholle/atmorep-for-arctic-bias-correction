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
