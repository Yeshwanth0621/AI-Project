"""
Simple model downloader - downloads only essential files for PyTorch
"""

from huggingface_hub import hf_hub_download
import os

print("=" * 60)
print("Downloading DistilGPT-2 (Essential Files Only)")
print("=" * 60)
print()

# Model repo
repo_id = "distilgpt2"
local_dir = "./distilgpt2_model"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Essential files for PyTorch
essential_files = [
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "pytorch_model.bin",  # ~353MB - the main model file
]

print("Downloading essential files...")
print()

downloaded_files = []

for filename in essential_files:
    try:
        print(f"Downloading {filename}...")
        filepath = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        downloaded_files.append(filename)
        print(f"  ✓ {filename} downloaded")
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")

print()
print("=" * 60)
print(f"Downloaded {len(downloaded_files)}/{len(essential_files)} files")
print(f"Model location: {os.path.abspath(local_dir)}")
print("=" * 60)

if len(downloaded_files) == len(essential_files):
    print("\n✓ SUCCESS! All files downloaded!")
    print(f"\nUpdate your config.yaml:")
    print(f'  base_model: "{os.path.abspath(local_dir)}"')
else:
    print("\n✗ Some files failed to download")
    exit(1)
