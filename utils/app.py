import argparse
import os
from pathlib import Path

import gradio as gr
import torch

# transformers>=5 dropped Flax support and removed the legacy FLAX_WEIGHTS_NAME
# constant that diffusers==0.34.0 still imports; restore it before importing diffusers.
import transformers.utils as _tu
if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
    _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from models.illumicraft import WanImageToVideoPipelineTracking, WanTransformer3DModelTracking

import testing.inference_single_sample as infer
from testing.gradio_backend import run_gradio_inference, extract_foreground


def parse_args():
    parser = argparse.ArgumentParser(description="IllumiCraft local Gradio demo")

    parser.add_argument("--wan_model_path", type=str, default=None)
    parser.add_argument("--illumicraft_ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/wan.yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument(
        "--frame_buckets",
        nargs="+",
        type=int,
        default=[49],
    )
    parser.add_argument(
        "--height_buckets",
        nargs="+",
        type=int,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument(
        "--width_buckets",
        nargs="+",
        type=int,
        default=[256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float16
    return device, dtype


def load_pipeline(ckpt_path, config_path, device, dtype):
    config = OmegaConf.load(config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            ckpt_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **infer.filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            ckpt_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            ckpt_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            ckpt_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    )

    transformer = WanTransformer3DModelTracking.from_pretrained(
        os.path.join(ckpt_path, "transformer"),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
    )

    text_encoder.to(device, dtype=dtype)
    vae.to(device, dtype=dtype)
    clip_image_encoder.to(device, dtype=dtype)
    transformer = infer.move_model(transformer, device, dtype)

    pipe = WanImageToVideoPipelineTracking(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipe.to(device, dtype=dtype)

    return config, pipe


def build_app():
    args = parse_args()
    device, dtype = get_device_and_dtype()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.illumicraft_ckpt_path is not None:
        model_root = args.illumicraft_ckpt_path
    else:
        model_root = args.wan_model_path
        
    config, pipe = load_pipeline(
        ckpt_path=model_root,
        config_path=args.config_path,
        device=device,
        dtype=dtype,
    )

    def relight_video(foreground_video, foreground_prompt, lighting_prompt, background_image, seed):
        return run_gradio_inference(
            pipe=pipe,
            wan_model_path=args.wan_model_path,
            illumicraft_ckpt_path=args.illumicraft_ckpt_path,
            config_path=args.config_path,
            output_dir=str(output_dir),
            foreground_video=foreground_video,
            foreground_prompt=foreground_prompt,
            lighting_prompt=lighting_prompt,
            background_image=background_image,
            seed=seed,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            height_buckets=args.height_buckets,
            width_buckets=args.width_buckets,
            frame_buckets=args.frame_buckets,
            dtype_str="bfloat16" if dtype == torch.bfloat16 else "float16",
        )

    def generate_foreground(input_video, foreground_prompt):
        return extract_foreground(
            input_video=input_video,
            foreground_prompt=foreground_prompt,
            output_dir=str(output_dir),
        )

    LOGO = (
        "<span class='logo'>"
        "<svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
        "<path d='M13 2 L4 14 h6 l-1 8 9-12 h-6 z' fill='#ffd76a' stroke='#ffb347' stroke-width='1'/>"
        "</svg></span>"
    )

    def badge(state, corner="top"):
        """Return the HTML for a status badge pinned to a corner of a video box.

        corner: "top" -> upper-right, "bottom" -> lower-right.
        """
        pos = "status-bottom" if corner == "bottom" else "status-top"
        if state == "run":
            return (
                f"<div class='status-badge status-run {pos}'>"
                f"{LOGO}<span class='spin'></span>Processing…</div>"
            )
        if state == "done":
            return f"<div class='status-badge status-done {pos}'>{LOGO}Done</div>"
        return f"<div class='status-badge status-idle {pos}'>{LOGO}Idle</div>"

    def fg_status_run():
        return badge("run", "bottom")

    def fg_status_done():
        return badge("done", "bottom")

    def relit_status_run():
        return badge("run", "top")

    def relit_status_done():
        return badge("done", "top")

    custom_css = """
    .gradio-container {
        background: radial-gradient(1200px 600px at 20% -10%, #1e2a4a 0%, transparent 60%),
                    radial-gradient(1000px 500px at 90% 0%, #3a1e4a 0%, transparent 55%),
                    linear-gradient(160deg, #0b1020 0%, #0e1330 55%, #10152e 100%) !important;
        color: #e8ecff !important;
    }
    #hero {
        text-align: center;
        padding: 26px 18px 10px 18px;
    }
    #hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #7cc9ff 0%, #a98bff 45%, #ff8fd0 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }
    #hero p { color: #aab4e0; margin-top: 8px; font-size: 1.02rem; }
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 6px 0 14px 0;
        backdrop-filter: blur(6px);
    }
    .info-card .k { color: #7cc9ff; font-weight: 600; }
    .info-card .v { color: #eaf1ff; font-weight: 500; }
    button.primary, .primary button {
        background: linear-gradient(90deg, #6d5efc 0%, #b15efc 100%) !important;
        border: none !important;
    }
    button.secondary, .secondary button {
        background: linear-gradient(90deg, #1f9ffa 0%, #17c3b2 100%) !important;
        border: none !important;
        color: #fff !important;
    }
    .gr-group, .block { border-radius: 14px !important; }

    /* processing status badge pinned to a corner of a video box */
    .out-wrap { position: relative; }
    /* the badge holder is absolutely positioned so it takes no vertical flow
       space (otherwise it shrinks the video below it and the column ends up
       shorter than its siblings). It spans the full box so a badge can anchor
       to either the top or bottom edge. */
    .out-wrap .badge-holder {
        position: absolute;
        top: 0;
        bottom: 0;
        right: 0;
        left: 0;
        height: auto !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible;
        z-index: 30;
        pointer-events: none;
    }
    .status-badge {
        position: absolute;
        right: 12px;
        z-index: 31;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: Inter, system-ui, sans-serif;
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(6px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
        pointer-events: auto;
    }
    .status-top    { top: 10px; }
    .status-bottom { bottom: 62px; }
    .status-badge .logo { width: 14px; height: 14px; display: inline-block; }
    .status-idle   { background: rgba(120,130,160,0.22); color: #c3cbe6; }
    .status-run    { background: linear-gradient(90deg, rgba(31,159,250,0.30), rgba(23,195,178,0.30)); color: #eafff9; }
    .status-done   { background: linear-gradient(90deg, rgba(109,94,252,0.30), rgba(177,94,252,0.30)); color: #f3edff; }
    .status-run .spin {
        width: 13px; height: 13px;
        border: 2px solid rgba(255,255,255,0.35);
        border-top-color: #fff;
        border-radius: 50%;
        display: inline-block;
        animation: statusspin 0.8s linear infinite;
    }
    @keyframes statusspin { to { transform: rotate(360deg); } }
    """

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=custom_css, title="IllumiCraft Video Relighting") as demo:
        gr.HTML(
            """
            <div id="hero">
              <h1>✨ IllumiCraft — Video Relighting Studio</h1>
              <p>Upload a raw video → extract the foreground with SAM3 → relight it with your prompt.</p>
            </div>
            """
        )
        gr.HTML(
            """
            <div class="info-card" style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.7;">
              <span class="k">Input video</span>                : <span class="v">Raw Input video</span><br>
              <span class="k">Foreground prompt</span>           : <span class="v">Description of the foreground subject.</span><br>
              <span class="k">Lighting prompt</span>             : <span class="v">Description of the desired lighting effect.</span><br>
              <span class="k">Background image (optional)</span> : <span class="v">Reference image for relighting guidance.</span><br>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            input_video = gr.Video(
                label="① Input Video",
                value="demo/eval/00000.mp4",
                visible=True,
            )
            with gr.Column(elem_classes="out-wrap"):
                fg_badge = gr.HTML(badge("idle", "bottom"), elem_classes="badge-holder")
                foreground_video = gr.Video(
                    label="② Foreground Video (auto-generated)",
                    interactive=True,
                    visible=True,
                )
            background_image = gr.Image(
                label="③ Background Image (optional)",
                type="filepath",
                value="demo/eval/custom_background.jpg",
                visible=True,
            )

        with gr.Group():
            foreground_prompt = gr.Textbox(
                label="Foreground prompt",
                lines=3,
                value="A majestic waterfall cascades down a rugged cliff into a serene pool.",
            )
            lighting_prompt = gr.Textbox(
                label="Lighting prompt",
                lines=3,
                value="Cool-blue spotlights beam through mist onto a central pool of light.",
            )
            seed = gr.Number(label="Seed", value=args.seed, precision=0)

        fg_btn = gr.Button("Generate foreground video", variant="secondary")
        run_btn = gr.Button("Generate relit video", variant="primary")

        with gr.Column(elem_classes="out-wrap"):
            relit_badge = gr.HTML(badge("idle", "top"), elem_classes="badge-holder")
            output_video = gr.Video(label="Relit output video")

        fg_btn.click(
            fn=fg_status_run, inputs=None, outputs=[fg_badge],
        ).then(
            fn=generate_foreground,
            inputs=[input_video, foreground_prompt],
            outputs=[foreground_video],
        ).then(
            fn=fg_status_done, inputs=None, outputs=[fg_badge],
        )
        run_btn.click(
            fn=relit_status_run, inputs=None, outputs=[relit_badge],
        ).then(
            fn=relight_video,
            inputs=[foreground_video, foreground_prompt, lighting_prompt, background_image, seed],
            outputs=[output_video],
        ).then(
            fn=relit_status_done, inputs=None, outputs=[relit_badge],
        )

    return demo, args


if __name__ == "__main__":
    demo, args = build_app()
    demo.queue(max_size=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True
    )
