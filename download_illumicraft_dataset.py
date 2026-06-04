from pathlib import Path
import tarfile
import shutil

from huggingface_hub import snapshot_download

REPO_ID = "YuanzeLin/IllumiCraft"

DOWNLOAD_DIR = Path("/mnt/data0/yuanze/dataset")
SHARD_DIR = DOWNLOAD_DIR / "train_shards"

TRAIN_DIR = DOWNLOAD_DIR / "train"
DEMO_DIR = DOWNLOAD_DIR / "demo_examples"


def safe_extract(tar: tarfile.TarFile, dest: Path):
    dest = dest.resolve()

    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")

    tar.extractall(dest)


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

# Restore train/
tar_files = sorted(SHARD_DIR.glob("*.tar"))
txt_files = sorted(SHARD_DIR.glob("*.txt"))

print(f"Found {len(tar_files)} tar shards")
print(f"Found {len(txt_files)} txt files")

for i, tar_path in enumerate(tar_files, 1):
    print(f"[{i}/{len(tar_files)}] Extracting {tar_path.name}")

    with tarfile.open(tar_path, "r") as tar:
        safe_extract(tar, TRAIN_DIR)

for txt_path in txt_files:
    shutil.copy2(txt_path, TRAIN_DIR / txt_path.name)

# Remove downloaded train_shards after recovery
shutil.rmtree(SHARD_DIR, ignore_errors=True)

print(f"Recovered train dataset to {TRAIN_DIR}")
print(f"Downloaded demo_examples to {DEMO_DIR}")
