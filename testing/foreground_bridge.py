"""Bridge to auto-generate a foreground video from a raw input video.

Wraps `utils/prepare_foreground_video.py` so the inference scripts can turn a
raw `--input_video_path` into the gray-background foreground video they expect,
running SAM3 + MatAnyone in-process (single `illumicraft` env).
"""

import os
import sys
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent


def ensure_foreground_video(
    input_video_path=None,
    foreground_video_path=None,
    foreground_prompt=None,
    output_dir=".",
    sam3_model_path=None,
    matanyone_model=None,
    score_threshold=0.4,
):
    """Return a path to a gray-background foreground video.

    If `foreground_video_path` already points at an existing file, it is
    returned unchanged. Otherwise `input_video_path` + `foreground_prompt`
    are used to auto-generate one (SAM3 seed mask + MatAnyone matting +
    gray-background composite), cached under `output_dir/generated_foreground_videos/`.
    """
    if foreground_video_path and os.path.exists(foreground_video_path):
        return foreground_video_path

    if not input_video_path:
        raise ValueError(
            "Provide --foreground_video_path (pointing at an existing file) or "
            "--input_video_path so the foreground video can be auto-generated."
        )
    if not foreground_prompt:
        raise ValueError(
            "--foreground_prompt is required to auto-extract the foreground video "
            "via SAM3 + MatAnyone when only --input_video_path is given."
        )

    cache_dir = Path(output_dir) / "generated_foreground_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{Path(input_video_path).stem}_foreground.mp4"

    if target_path.exists():
        print(f"[foreground_bridge] Using cached foreground video: {target_path}")
        return str(target_path)

    if str(REPO_ROOT / "utils") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "utils"))
    from prepare_foreground_video import prepare_foreground_video

    kwargs = dict(
        video_path=str(input_video_path),
        text_prompt=foreground_prompt,
        output_path=str(target_path),
        score_threshold=score_threshold,
    )
    if sam3_model_path:
        kwargs["sam3_model_path"] = sam3_model_path
    if matanyone_model:
        kwargs["matanyone_model"] = matanyone_model

    print(f"[foreground_bridge] Extracting foreground video -> {target_path}")
    prepare_foreground_video(**kwargs)

    if not target_path.exists():
        raise RuntimeError(f"Foreground extraction did not produce {target_path}")

    return str(target_path)
