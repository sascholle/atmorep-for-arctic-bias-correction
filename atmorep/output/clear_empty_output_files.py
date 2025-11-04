import os

# Directory containing the .txt files
output_dir = "/work/ab1412/atmorep/output"

deleted_count = 0

for root, _, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            try:
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"🗑️ Deleted empty file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not check/delete {file_path}: {e}")

print(f"\n✅ Done. Deleted {deleted_count} empty .txt file(s) from {output_dir}")
