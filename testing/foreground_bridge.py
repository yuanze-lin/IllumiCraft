"""Bridge to auto-generate a foreground video from a raw input video.

`utils/prepare_foreground_video.py` needs a recent `transformers` (for SAM3)
and lives in the separate `fgprep` conda env, while the inference scripts run
in the `illumicraft` env (pinned to an older `transformers`). This module
shells out across that env boundary so callers still only need one command.
"""

import os
import subprocess
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent

DEFAULT_FGPREP_PYTHON = str(REPO_ROOT.parent / "conda_envs" / "fgprep" / "bin" / "python")
DEFAULT_FGPREP_SCRIPT = str(REPO_ROOT / "utils" / "prepare_foreground_video.py")


def ensure_foreground_video(
    input_video_path=None,
    foreground_video_path=None,
    foreground_prompt=None,
    output_dir=".",
    fgprep_python=None,
    fgprep_script=None,
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

    fgprep_python = fgprep_python or os.environ.get("FGPREP_PYTHON", DEFAULT_FGPREP_PYTHON)
    fgprep_script = fgprep_script or os.environ.get("FGPREP_SCRIPT", DEFAULT_FGPREP_SCRIPT)

    cache_dir = Path(output_dir) / "generated_foreground_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{Path(input_video_path).stem}_foreground.mp4"

    if target_path.exists():
        print(f"[foreground_bridge] Using cached foreground video: {target_path}")
        return str(target_path)

    cmd = [
        fgprep_python,
        fgprep_script,
        "--video_path", str(input_video_path),
        "--text_prompt", foreground_prompt,
        "--output_path", str(target_path),
        "--score_threshold", str(score_threshold),
    ]
    if sam3_model_path:
        cmd += ["--sam3_model_path", sam3_model_path]
    if matanyone_model:
        cmd += ["--matanyone_model", matanyone_model]

    # The illumicraft env's own imports (torch/diffusers/transformers) mutate
    # os.environ as a side effect (e.g. TORCH_LOGS='', TORCHDYNAMO_VERBOSE='0').
    # The fgprep env runs a different torch version and mishandles those
    # inherited values, so strip anything torch/dynamo-internal before exec.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TORCH_", "TORCHDYNAMO_", "TORCHINDUCTOR_"))
    }

    print(f"[foreground_bridge] Extracting foreground video: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    if not target_path.exists():
        raise RuntimeError(f"Foreground extraction did not produce {target_path}")

    return str(target_path)
