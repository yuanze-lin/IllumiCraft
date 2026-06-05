from pathlib import Path
import tarfile
import shutil

from huggingface_hub import snapshot_download

REPO_ID = "YuanzeLin/IllumiCraft"

# Final layout:
# dataset/
# ├── train/
# └── demo_examples/
DOWNLOAD_DIR = Path("dataset")

# Downloaded shard location from HF
SHARD_DIR = DOWNLOAD_DIR / "train_shards"

# Recovered training dataset
TRAIN_DIR = DOWNLOAD_DIR / "train"

# Downloaded directly from HF
DEMO_DIR = DOWNLOAD_DIR / "demo_examples"


def safe_extract(tar: tarfile.TarFile, dest: Path):
    """Extract tar safely to avoid path traversal attacks."""
    dest = dest.resolve()

    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")

    tar.extractall(dest)


# Download only the folders we need from HF
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        "train_shards/*",
        "demo_examples/*",
    ],
    local_dir=str(DOWNLOAD_DIR),
)

TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Find all downloaded shard files and metadata txt files
tar_files = sorted(SHARD_DIR.glob("*.tar"))
txt_files = sorted(SHARD_DIR.glob("*.txt"))

print(f"Found {len(tar_files)} tar shards")
print(f"Found {len(txt_files)} txt files")

# Recover original dataset structure from shards
for i, tar_path in enumerate(tar_files, 1):
    print(f"[{i}/{len(tar_files)}] Extracting {tar_path.name}")

    with tarfile.open(tar_path, "r") as tar:
        safe_extract(tar, TRAIN_DIR)

# Copy metadata txt files into train/
for txt_path in txt_files:
    shutil.copy2(txt_path, TRAIN_DIR / txt_path.name)

# No longer needed after extraction
shutil.rmtree(SHARD_DIR, ignore_errors=True)

print(f"Recovered train dataset to: {TRAIN_DIR}")
print(f"Downloaded demo_examples to: {DEMO_DIR}")
