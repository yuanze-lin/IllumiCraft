from huggingface_hub import snapshot_download

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
    allow_patterns="checkpoint/*",
)

checkpoint_dir = dst / "checkpoint"

for p in checkpoint_dir.iterdir():
    shutil.move(str(p), str(dst / p.name))

checkpoint_dir.rmdir()

print("Downloaded to checkpoints/illumicraft_weights")

