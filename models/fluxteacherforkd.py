import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Union

from diffusers import AutoencoderKL
try:
    from diffusers import FluxPipeline
except Exception:
    FluxPipeline = None


@dataclass
class FluxTeacherOutput:
    # Unpacked BCHW latent, aligned with student latent geometry
    teacher_latents: torch.Tensor              # [B, 16, H', W']
    teacher_noisy_latents: torch.Tensor        # [B, 16, H', W']

    # Packed transformer-space tensors
    teacher_model_pred: torch.Tensor           # [B, N, 64] for FLUX.1-dev packed latents
    teacher_x0_latent: torch.Tensor            # [B, 16, H', W']
    teacher_x0_image: torch.Tensor             # [B, 3, H, W]


class FluxTeacherForKD(nn.Module):
    """
    FLUX.1-dev teacher wrapper for Stage-B KD.

    This wrapper:
      - encodes pixel images to teacher VAE latents
      - packs latents into FLUX transformer token format
      - encodes prompts using FluxPipeline.encode_prompt(...)
      - runs teacher transformer forward
      - converts teacher prediction into x0 estimate
      - decodes teacher x0 image
    """

    def __init__(
        self,
        repo_id: str = "black-forest-labs/FLUX.1-dev",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
        latent_scaling_factor: float = 1.0,
        max_sequence_length: int = 512,
        guidance_scale: float = 3.5,
    ):
        super().__init__()
        if FluxPipeline is None:
            raise ImportError("FluxPipeline is not available in your installed diffusers package.")

        self.pipe = FluxPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.text_encoder = getattr(self.pipe, "text_encoder", None)
        self.text_encoder_2 = getattr(self.pipe, "text_encoder_2", None)
        self.tokenizer = getattr(self.pipe, "tokenizer", None)
        self.tokenizer_2 = getattr(self.pipe, "tokenizer_2", None)

        self.latent_scaling_factor = latent_scaling_factor
        self.max_sequence_length = max_sequence_length
        self.guidance_scale = guidance_scale

        if device is not None:
            self.pipe.to(device)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        posterior = self.vae.encode(pixel_values).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return latents * self.latent_scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents / self.latent_scaling_factor).sample

    @torch.no_grad()
    def encode_text(
        self,
        prompts: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: Optional[int] = None,
    ):
        """
        Uses FluxPipeline.encode_prompt(...), which returns:
          prompt_embeds, pooled_prompt_embeds, text_ids
        in the current pipeline implementation. :contentReference[oaicite:4]{index=4}
        """
        if device is None:
            device = self.device
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length

        if isinstance(prompts, str):
            prompts = [prompts]

        # For FluxPipeline, prompt_2 can be None; the pipeline handles it.
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _flow_sigma_from_timestep(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert training timestep -> sigma for flow/noising.

        The FlowMatch scheduler stores sigmas as shifted versions of t / num_train_timesteps
        when dynamic shifting is off. The scheduler's scale_noise formula is:
            x_t = sigma * noise + (1 - sigma) * x_0. :contentReference[oaicite:5]{index=5}

        This helper reproduces that logic approximately for arbitrary training timesteps.
        """
        t = timesteps.float() / float(self.scheduler.config.num_train_timesteps)

        use_dynamic = getattr(self.scheduler.config, "use_dynamic_shifting", False)
        if use_dynamic:
            # Dynamic shifting depends on image sequence length and pipeline-level mu computation.
            # For training, unless you re-implement full dynamic shift, fall back to raw normalized t.
            sigma = t
        else:
            shift = float(getattr(self.scheduler.config, "shift", 1.0))
            sigma = shift * t / (1.0 + (shift - 1.0) * t)

        return sigma

    def _pack_teacher_latents(self, latents_bchw: torch.Tensor):
        """
        Pack [B, C, H, W] -> [B, N, 4C] as Flux transformer expects. :contentReference[oaicite:6]{index=6}
        """
        b, c, h, w = latents_bchw.shape
        packed = self.pipe._pack_latents(latents_bchw, b, c, h, w)
        latent_image_ids = self.pipe._prepare_latent_image_ids(
            b,
            h // 2,
            w // 2,
            latents_bchw.device,
            latents_bchw.dtype,
        )
        return packed, latent_image_ids

    def _unpack_teacher_latents(self, packed_latents: torch.Tensor, latent_h: int, latent_w: int):
        """
        Unpack [B, N, 4C] -> [B, C, H, W]. :contentReference[oaicite:7]{index=7}
        """
        unpacked = self.pipe._unpack_latents(
            packed_latents,
            latent_h * self.pipe.vae_scale_factor,
            latent_w * self.pipe.vae_scale_factor,
            self.pipe.vae_scale_factor,
        )
        return unpacked

    @torch.no_grad()
    def forward_kd(
        self,
        pixel_values: torch.Tensor,
        prompts: Union[str, List[str]],
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        sample_posterior: bool = False,
        guidance_scale: Optional[float] = None,
    ) -> FluxTeacherOutput:
        """
        Args:
            pixel_values: [B, 3, H, W] in [-1, 1]
            prompts: list[str] of length B
            timesteps: [B] integer-like training timesteps in [0, num_train_timesteps)
            noise: [B, 16, H', W'] student/teacher-aligned latent noise in BCHW
        Returns:
            teacher outputs usable for Stage-B KD.
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # 1) Teacher clean latent in BCHW
        teacher_latents_bchw = self.encode_image(
            pixel_values,
            sample_posterior=sample_posterior,
        )  # [B,16,H',W']

        b, c, latent_h, latent_w = teacher_latents_bchw.shape

        # 2) Pack clean latent for transformer-space processing
        teacher_latents_packed, latent_image_ids = self._pack_teacher_latents(teacher_latents_bchw)

        # 3) Pack noise into transformer token space too
        if noise.dim() != 4:
            raise ValueError(f"`noise` is expected in BCHW latent format, got shape={tuple(noise.shape)}")
        if noise.shape != teacher_latents_bchw.shape:
            raise ValueError(
                f"`noise` shape {tuple(noise.shape)} must match teacher latent shape {tuple(teacher_latents_bchw.shape)}"
            )
        noise_packed, _ = self._pack_teacher_latents(noise)

        # 4) Build noised teacher latent using flow-match-style sigma
        sigma = self._flow_sigma_from_timestep(timesteps).to(device=self.device, dtype=teacher_latents_packed.dtype)
        while sigma.ndim < teacher_latents_packed.ndim:
            sigma = sigma.unsqueeze(-1)

        teacher_noisy_packed = sigma * noise_packed + (1.0 - sigma) * teacher_latents_packed
        teacher_noisy_bchw = self._unpack_teacher_latents(teacher_noisy_packed, latent_h, latent_w)

        # 5) Text conditioning
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_text(
            prompts=prompts,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.max_sequence_length,
        )

        # 6) Guidance embedding, if enabled by transformer config
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.full(
                (teacher_noisy_packed.shape[0],),
                float(guidance_scale),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            guidance = None

        # 7) Transformer forward
        # Current FluxPipeline uses:
        # hidden_states=latents,
        # timestep=timestep / 1000,
        # guidance=guidance,
        # pooled_projections=pooled_prompt_embeds,
        # encoder_hidden_states=prompt_embeds,
        # txt_ids=text_ids,
        # img_ids=latent_image_ids,
        # joint_attention_kwargs=...,
        # return_dict=False. :contentReference[oaicite:8]{index=8}
        timestep_for_transformer = timesteps.to(device=self.device, dtype=teacher_noisy_packed.dtype) / 1000.0

        teacher_model_pred = self.transformer(
            hidden_states=teacher_noisy_packed,
            timestep=timestep_for_transformer,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # 8) Convert teacher prediction to x0 estimate
        # For FlowMatchEulerDiscreteScheduler:
        # scale_noise uses x_t = sigma * noise + (1 - sigma) * x0
        # and deterministic step uses prev = sample + dt * model_output,
        # consistent with velocity-style parameterization. A common x0 estimate is:
        # x0 ≈ x_t - sigma * v_theta. :contentReference[oaicite:9]{index=9}
        teacher_x0_packed = teacher_noisy_packed - sigma * teacher_model_pred

        # 9) Unpack x0 back to BCHW latent and decode
        teacher_x0_bchw = self._unpack_teacher_latents(teacher_x0_packed, latent_h, latent_w)
        teacher_x0_image = self.decode_latents(teacher_x0_bchw)

        return FluxTeacherOutput(
            teacher_latents=teacher_latents_bchw,
            teacher_noisy_latents=teacher_noisy_bchw,
            teacher_model_pred=teacher_model_pred,
            teacher_x0_latent=teacher_x0_bchw,
            teacher_x0_image=teacher_x0_image,
        )