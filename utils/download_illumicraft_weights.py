from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-1.3B-Control",
    local_dir="checkpoints/Wan2.1-Fun-1.3B-Control",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded to checkpoints/Wan2.1-Fun-1.3B-Control")

dst = Path("checkpoints/illumicraft_pretrained_weights")

snapshot_download(
    repo_id="YuanzeLin/Illumicraft-checkpoints",
    local_dir=dst,
    local_dir_use_symlinks=False,
)

print(f"Downloaded to {dst}")

print("Downloaded to checkpoints/illumicraft_weights")
