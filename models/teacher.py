from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn

from diffusers import AutoencoderKL
try:
    from diffusers import FluxPipeline
except Exception:
    FluxPipeline = None


@dataclass
class TeacherKDOutput:
    teacher_latents: torch.Tensor
    teacher_recon: torch.Tensor
    teacher_mean: Optional[torch.Tensor] = None
    teacher_logvar: Optional[torch.Tensor] = None


class TeacherWrapper(nn.Module):
    """
    Teacher wrapper for black-forest-labs/FLUX.1-dev.

    Recommended for Stage 1:
        load_mode="vae_only"

    Notes:
        - FLUX.1-dev uses a 16-channel sampled latent space.
        - The encoder internally predicts Gaussian posterior moments
          (mean + logvar), but the denoiser-compatible latent is 16-channel.
    """

    def __init__(
        self,
        teacher_repo_id: str = "black-forest-labs/FLUX.1-dev",
        load_mode: Literal["vae_only", "pipeline"] = "vae_only",
        subfolder: str = "vae",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        latent_scaling_factor: float = 1.0,
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        self.teacher_repo_id = teacher_repo_id
        self.load_mode = load_mode
        self.latent_scaling_factor = latent_scaling_factor
        self.hf_token = hf_token

        self.pipeline = None

        if load_mode == "pipeline":
            if FluxPipeline is None:
                raise ImportError(
                    "FluxPipeline is not available in your installed diffusers. "
                    "Install a diffusers version with FLUX support."
                )

            self.pipeline = FluxPipeline.from_pretrained(
                teacher_repo_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
            self.vae = self.pipeline.vae

        elif load_mode == "vae_only":
            self.vae = AutoencoderKL.from_pretrained(
                teacher_repo_id,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        else:
            raise ValueError(f"Unknown load_mode: {load_mode}")

        if device is not None:
            self.vae.to(device=device, dtype=torch_dtype)
            if self.pipeline is not None:
                self.pipeline.to(device)

        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        if not hasattr(self.vae, "config") or not hasattr(self.vae.config, "latent_channels"):
            raise ValueError("Loaded teacher VAE does not expose `config.latent_channels`.")

        self.latent_channels = int(self.vae.config.latent_channels)

        # For FLUX.1-dev this is expected to be 16
        if self.latent_channels != 16:
            print(
                f"[TeacherWrapper] Warning: expected FLUX.1-dev latent_channels=16, "
                f"but got {self.latent_channels}."
            )

    @property
    def device(self):
        return next(self.vae.parameters()).device

    @torch.no_grad()
    def encode(
        self,
        pixel_values: torch.Tensor,
        sample_posterior: bool = False,
    ):
        """
        Returns:
            latents: [B, 16, H', W'] for FLUX.1-dev
            mean:    [B, 16, H', W']
            logvar:  [B, 16, H', W']
        """
        enc_out = self.vae.encode(pixel_values)
        latent_dist = enc_out.latent_dist

        mean = latent_dist.mean
        logvar = latent_dist.logvar

        latents = latent_dist.sample() if sample_posterior else latent_dist.mode()
        latents = latents * self.latent_scaling_factor

        return latents, mean * self.latent_scaling_factor, logvar

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        recon = self.vae.decode(latents).sample
        return recon

    @torch.no_grad()
    def forward_for_kd(
        self,
        pixel_values: torch.Tensor,
        sample_posterior: bool = False,
    ) -> TeacherKDOutput:
        teacher_latents, teacher_mean, teacher_logvar = self.encode(
            pixel_values,
            sample_posterior=sample_posterior,
        )
        teacher_recon = self.decode(teacher_latents)

        return TeacherKDOutput(
            teacher_latents=teacher_latents,
            teacher_recon=teacher_recon,
            teacher_mean=teacher_mean,
            teacher_logvar=teacher_logvar,
        )
    
'''
teacher = TeacherWrapper(
    teacher_repo_id="black-forest-labs/FLUX.1-dev",
    load_mode="vae_only",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    device=torch.device("cuda"),
    latent_scaling_factor=1.0,  # replace after calibration
    hf_token="YOUR_HF_TOKEN",
)
'''