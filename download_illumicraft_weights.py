from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YuanzeLin/Illumicraft-checkpoints",
    local_dir="checkpoints/illumicraft_weights",
    local_dir_use_symlinks=False,
)
