import os
import sys
import time
import random
from atmorep.core.evaluator import Evaluator
import torch.distributed as dist

# Parse command-line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
parser.add_argument("--hour", type=int, required=True)
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

model_id = 'iuw3ce3v_single_gpu'
file_path = '/scratch/a/a270277/atmorep/era5_y2010_2020_res25_corrected_t2m.zarr'
results_folder = "/work/ab1412/atmorep/results/N-ICE_evaluations"
os.makedirs(results_folder, exist_ok=True)

# Set a unique port for DDP
os.environ['MASTER_ADDR'] = 'localhost'
base_port = 1345
port = base_port + random.randint(0, 10000)
os.environ['MASTER_PORT'] = str(port)

options = {
    'dates': [[args.year, args.month, args.day, args.hour]],
    'geo_range_sampling': None,
    'token_overlap': [0, 0],
    'forecast_num_tokens': 1,
    'attention': False,
    'with_pytest': False
}

print(f"Running global_forecast for {args.year}-{args.month:02d}-{args.day:02d} {args.hour:02d}:00 ...")
now = time.time()
Evaluator.evaluate('global_forecast', model_id, file_path, options)
if dist.is_initialized():
    dist.destroy_process_group()
print(f"Evaluation completed in {time.time() - now:.2f} seconds.")

# Save a marker file for each evaluation
with open(os.path.join(results_folder, f"eval_{args.idx}_{args.year}{args.month:02d}{args.day:02d}_{args.hour:02d}.txt"), "w") as f:
    f.write(f"Evaluated {args.year}-{args.month:02d}-{args.day:02d} {args.hour:02d}\n")