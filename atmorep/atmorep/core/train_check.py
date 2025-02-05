import os
import zarr
import torch

checkpoint_path = '/work/ab1412/atmorep/models/id3cizyl1q/AtmoRep_id3cizyl1q.mod'
checkpoint = torch.load(checkpoint_path)
filtered_keys = [key for key in checkpoint.keys() if not (key.startswith('tails') or key.startswith('decoders') or key.startswith('encoders'))]
for key in filtered_keys:
    print(key)