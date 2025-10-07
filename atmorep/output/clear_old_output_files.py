#!/usr/bin/env python3
import os
import re
import shutil

# Set paths
plot_losses_path = "/work/ab1412/atmorep/plotting/plot_losses.py"
output_folder = "/work/ab1412/atmorep/output"
unused_folder = "/work/ab1412/atmorep/output/unused_txt"

# Ensure the unused folder exists
os.makedirs(unused_folder, exist_ok=True)

# 1) Read plot_losses.py and extract numeric IDs from lines starting with '#' or containing 'Run ID:' etc.
pattern = re.compile(r'(\d+)')
valid_ids = set()

with open(plot_losses_path, 'r') as f:
    for line in f:
        # Capture all numeric sequences in comments or text
        found = pattern.findall(line)
        for num in found:
            valid_ids.add(num.strip())

# 2) List all .txt files in /work/ab1412/atmorep/output
for filename in os.listdir(output_folder):
    if filename.endswith(".txt") and filename.startswith("output_"):
        # Extract numeric part (e.g., "output_17413071.txt" -> "17413071")
        match = re.match(r'output_(\d+)\.txt', filename)
        if match:
            file_id = match.group(1)
            # 3) If file_id not in valid_ids, move it
            if file_id not in valid_ids:
                src_path = os.path.join(output_folder, filename)
                dst_path = os.path.join(unused_folder, filename)
                print(f"Moving {filename} to {unused_folder}")
                shutil.move(src_path, dst_path)