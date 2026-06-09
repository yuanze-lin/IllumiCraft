"""IllumiCraft inference pipeline."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
import os

import torch
import torch.cuda.amp as amp
from PIL import Image
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import nn
from tqdm import tqdm

from diffusers.utils import is_torch_version, logging
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from models import AutoTokenizer, AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from models.hdr_encoder import HDRILightEncoder
from models.motion_encoder import MotionEncoder
from models.pipeline_wan_fun_control import WanFunControlPipeline, WanPipelineOutput
from models.wan_transformer3d import WanTransformer3DModel, sinusoidal_embedding_1d
from .cache_utils import TeaCache
from .wan_transformer3d import WanAttentionBlock

# Optional CUDA/Triton diagnostics are suppressed for cleaner release behavior.
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["TORCH_LOGS"] = ""

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class WanTransformer3DModelTracking(WanTransformer3DModel, ModelMixin):
    """Tracking-aware Wan transformer with optional HDR and motion conditioning."""

    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        light_token_num=3,
        hdr_drop_prob=0.5,
        tracking_drop_prob=0.3,
        prompt_dim=1536,
        num_tracking_blocks=4,
        **kwargs
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            **kwargs
        )
        inner_dim = 1536
        # inner_dim = num_attention_heads * attention_head_dim
        self.num_tracking_blocks = num_tracking_blocks

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_tracking_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and tracking maps
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim) for _ in range(num_tracking_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        self.transformer_blocks_copy = nn.ModuleList([
            WanAttentionBlock('i2v_cross_attn', dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps).to("cpu")
            for _ in range(num_tracking_blocks)
        ])

        # For initial combination of hidden states and tracking maps
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim)
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()

        # Unfreeze parameters that need to be trained
        for linear in self.combine_linears:
            for param in linear.parameters():
                param.requires_grad = True

        for block in self.transformer_blocks_copy:
            for param in block.parameters():
                param.requires_grad = True

        for param in self.initial_combine_linear.parameters():
            param.requires_grad = True

        self.light_token_num = light_token_num
        self.hdr_encoder = HDRILightEncoder(mlp_dims=(3072, 4096, 4096, 4096, prompt_dim*self.light_token_num),prompt_dim=prompt_dim)
        self.motion_encoder = MotionEncoder()

        for param in self.hdr_encoder.parameters():
            param.requires_grad = True

        for param in self.motion_encoder.parameters():
            param.requires_grad = True

        #self.hdr_default_tokens = nn.Parameter(torch.zeros(1, self.light_token_num, prompt_dim))
        self.hdr_default_tokens = nn.Parameter(torch.zeros(self.light_token_num, prompt_dim))

        nn.init.normal_(self.hdr_default_tokens, mean=0.0, std=1e-2)
        self.hdr_default_tokens.requires_grad = True
        self.gradient_checkpointing = True
        self.hdr_drop_prob = hdr_drop_prob
        self.tracking_drop_prob = tracking_drop_prob
        self.prompt_dim = prompt_dim

    def forward(
        self,
        x, # latents
        t, # timestep
        context, # prompt_embedding
        seq_len, # sequence length
        hdr_maps=None, # hdr maps
        tracking_maps=None, # tracking maps
        clip_fea=None, # clip feature
        y=None, # control_latens (foreground latents)
        cond_flag=True,
        is_training: bool = True
    ):
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]   # video + foreground
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # to bfloat16 for saving memeory
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context]))

        use_real_hdr = hdr_maps is not None and (
                not is_training or torch.rand((), device=x.device) >= self.hdr_drop_prob
        )

        real_hdr = (
            self.hdr_encoder(hdr_maps)
            if use_real_hdr
            else torch.zeros(
                x.shape[0], self.light_token_num, self.prompt_dim, device=x.device
            )
        )

        # Learned default tokens remain trainable even when real HDR is dropped.
        default = self.hdr_default_tokens.unsqueeze(0).expand(x.shape[0], -1, -1) # [B,3,inner_dim]
        hdr_tokens = real_hdr + default  # [B,3,inner_dim]

        context = torch.cat([hdr_tokens, context], dim=1)
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        if tracking_maps is None:
            tracking_maps = torch.zeros_like(y[:, :y.shape[1] // 2, :, :, :])

        drop_tracking = bool(
            is_training
            and tracking_maps is not None
            and self.tracking_drop_prob > 0
            and torch.rand((), device=x.device) < self.tracking_drop_prob
        )
        if drop_tracking:
            tracking_maps = torch.zeros_like(tracking_maps)

        tracking_maps = self.motion_encoder(tracking_maps, seq_len)

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if self.teacache.cnt == 0 or self.teacache.cnt == self.teacache.num_steps - 1 or skip_flag:
                    should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input,
                                                                                modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.cnt += 1
                if self.teacache.cnt == self.teacache.num_steps:
                    self.teacache.reset()
                self.teacache.should_calc = should_calc
            else:
                should_calc = self.teacache.should_calc

        # TeaCache
        if self.teacache is not None:
            if not should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,  # time embedding
                            seq_lens=seq_lens, # sequence length
                            grid_sizes=grid_sizes, # grid size
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype
                        )
                        x = block(x, **kwargs)

                    if cond_flag:
                        self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                    else:
                        self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype
                    )
                    x = block(x, **kwargs)

            for i, block in enumerate(self.transformer_blocks_copy):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    tracking_maps = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        tracking_maps,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype
                    )
                    tracking_maps = block(tracking_maps, **kwargs)
                # Combine hidden states and tracking maps
                tracking_maps = self.combine_linears[i](tracking_maps)
                x = x + tracking_maps

        if self.sp_world_size > 1:
            x = get_sp_group().all_gather(x, dim=1)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        return x

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            print("Loaded IllumiCraft checkpoint directly.")
            
            for param in model.parameters():
                param.requires_grad = True
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True

            for param in model.hdr_encoder.parameters():
                 param.requires_grad = True

            for param in model.motion_encoder.parameters():
                 param.requires_grad = True

            model.hdr_default_tokens.requires_grad = True

            return model
        
        except Exception as e:
            print(f"Failed to load as IllumiCraft: {e}")
            print("Attempting to load the base Wan transformer and convert...")

            base_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            config = dict(base_model.config)
            # config["num_tracking_blocks"] = kwargs.pop("num_tracking_blocks", 18)
            
            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.initial_combine_linear.weight.data.zero_()
            model.initial_combine_linear.bias.data.zero_()
            
            for linear in model.combine_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            
            for i in range(model.num_tracking_blocks):
                model.transformer_blocks_copy[i].load_state_dict(model.transformer_blocks[i].state_dict())

            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "WanTransformer3DModelTracking"
            config_dict["num_tracking_blocks"] = self.num_tracking_blocks
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)

class WanImageToVideoPipelineTracking(WanFunControlPipeline):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, clip_image_encoder, scheduler)

        if not isinstance(self.transformer, WanTransformer3DModelTracking):
            raise ValueError("The transformer in this pipeline must be of type WanTransformer3DModelTracking")

        print(f"Number of transformer blocks: {len(self.transformer.blocks)}")
        print(f"Number of tracking transformer blocks: {len(self.transformer.transformer_blocks_copy)}")
        # self.transformer = torch.compile(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        control_video: Union[torch.FloatTensor] = None,
        ref_image: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Image = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        hdr_maps: Optional[torch.Tensor] = None,
        tracking_maps: Optional[torch.Tensor] = None,
        foreground_maps: Optional[torch.Tensor] = None,
    ) -> Union[WanPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = negative_prompt_embeds + prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps,
                                                                mu=1)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        if comfyui_progressbar:
            pbar.update(1)

        # Prepare mask latent variables
        if control_video is not None:
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"),
                                                            height=height, width=width)
            control_video = control_video.to(dtype=torch.float32)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
            control_latents = (
                torch.cat([control_video_latents] * 2) if do_classifier_free_guidance else control_video_latents
            ).to(device, weight_dtype)
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_latents = (
                torch.cat([control_video_latents] * 2) if do_classifier_free_guidance else control_video_latents
            ).to(device, weight_dtype)

        if ref_image is not None:
            video_length = ref_image.shape[2]
            ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height,
                                                        width=width)
            ref_image = ref_image.to(dtype=torch.float32)
            ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)

            ref_image_latentes = self.prepare_control_latents(
                None,
                ref_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            ref_image_latentes_conv_in = torch.zeros_like(latents)
            if latents.size()[2] != 1:
                ref_image_latentes_conv_in[:, :, :1] = ref_image_latentes
            ref_image_latentes_conv_in = (
                torch.cat(
                    [ref_image_latentes_conv_in] * 2) if do_classifier_free_guidance else ref_image_latentes_conv_in
            ).to(device, weight_dtype)
            control_latents = torch.cat([control_latents, ref_image_latentes_conv_in], dim=1)
        else:
            ref_image_latentes_conv_in = torch.zeros_like(latents)
            ref_image_latentes_conv_in = (
                torch.cat(
                    [ref_image_latentes_conv_in] * 2) if do_classifier_free_guidance else ref_image_latentes_conv_in
            ).to(device, weight_dtype)
            control_latents = torch.cat([control_latents, ref_image_latentes_conv_in], dim=1)

        # Prepare clip latent variables
        if clip_image is not None:
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
            clip_context = (
                torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
            )
        else:
            clip_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
            clip_context = (
                torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
            )
            clip_context = torch.zeros_like(clip_context)
        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
                        width // self.vae.spacial_compression_ratio, height // self.vae.spacial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (
                    self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                try:
                    hdr_maps_input = torch.cat([hdr_maps] * 2) if do_classifier_free_guidance else hdr_maps
                except:
                    hdr_maps_input = None

                try:
                    tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                except:
                    tracking_maps_input = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        hdr_maps=hdr_maps_input,
                        tracking_maps=tracking_maps_input,
                        y=control_latents,
                        clip_fea=clip_context,
                        is_training=False
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)

