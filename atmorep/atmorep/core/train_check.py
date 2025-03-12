import os
import zarr
import torch

# checkpoint_path = '/work/ab1412/atmorep/models/id3cizyl1q/AtmoRep_id3cizyl1q.mod'
# checkpoint = torch.load(checkpoint_path)
# filtered_keys = [key for key in checkpoint.keys() if not (key.startswith('tails') or key.startswith('decoders') or key.startswith('encoders'))]
# for key in filtered_keys:
#     print(key)


file_path = "/work/ab1412/atmorep/data/era5_y2010_2020_res100.zarr"
#file_path = "/work/ab1412/atmorep/data/era5_y2010_2021_res025.zarr"

zarr_file = zarr.open(file_path, mode='r')
print(zarr_file.tree())

import json

with open('/work/ab1412/atmorep/models/id3cizyl1q/model_id3cizyl1q.json', 'r') as f:
    try:
        data = json.load(f)
        print("JSON is valid")
    except json.JSONDecodeError as e:
        print(f"JSON is invalid: {e}")
