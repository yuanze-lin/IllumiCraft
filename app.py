import argparse
import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from models.illumicraft import WanImageToVideoPipelineTracking, WanTransformer3DModelTracking

import testing.inference_single_sample as infer
from testing.gradio_backend import run_gradio_inference


def parse_args():
    parser = argparse.ArgumentParser(description="IllumiCraft local Gradio demo")

    parser.add_argument("--wan_model_path", type=str, required=True)
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


def load_pipeline(wan_model_path, illumicraft_ckpt_path, config_path, device, dtype):
    config = OmegaConf.load(config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            wan_model_path,
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
            wan_model_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            wan_model_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            wan_model_path,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        )
    )

    transformer = WanTransformer3DModelTracking.from_pretrained(
        os.path.join(illumicraft_ckpt_path, "transformer"),
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

    config, pipe = load_pipeline(
        wan_model_path=args.wan_model_path,
        illumicraft_ckpt_path=args.illumicraft_ckpt_path,
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

    with gr.Blocks() as demo:
        gr.Markdown("# IllumiCraft Video Relighting Demo")
        gr.Markdown("""
           ### Default Example
           Sample inputs are preloaded and can be freely modified or replaced.

           ### Inputs

           <div style="font-family: monospace;">
           Foreground video            : Foreground subject video with a uniform #888B88 gray background.<br>
           Foreground prompt           : Description of the foreground subject.<br>
           Lighting prompt             : Description of the desired lighting effect.
           Background image (optional) : Reference image for relighting guidance.<br>
           </div>
        """)  
        with gr.Row():
            foreground_video = gr.Video(
                label="Foreground video",
                value="demo/eval/foreground_videos_00000.mp4",
                visible=True,
            )
            background_image = gr.Image(
                label="Background image (optional)",
                type="filepath",
                value="demo/eval/custom_background.jpg",
                visible=True,
            ) 

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

        run_btn = gr.Button("Generate relit video", variant="primary")
        output_video = gr.Video(label="Output video")

        run_btn.click(
            fn=relight_video,
            inputs=[foreground_video, foreground_prompt, lighting_prompt, background_image, seed],
            outputs=[output_video],
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
