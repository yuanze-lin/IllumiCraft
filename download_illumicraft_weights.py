from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-1.3B-Control",
    local_dir="checkpoints/Wan2.1-Fun-1.3B-Control",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded to checkpoints/Wan2.1-Fun-1.3B-Control")

snapshot_download(
    repo_id="YuanzeLin/Illumicraft-checkpoints",
    local_dir="checkpoints/illumicraft_pretrained_weights",
    allow_patterns="checkpoint/*",
)

print("Downloaded to checkpoints/illumicraft_weights")

