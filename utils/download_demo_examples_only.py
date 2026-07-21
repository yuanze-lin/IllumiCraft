from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YuanzeLin/IllumiCraft",
    repo_type="dataset",
    allow_patterns=["demo_examples/*"],
    local_dir="dataset",
)
print("Downloaded demo_examples to: dataset/demo_examples")
