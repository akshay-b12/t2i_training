from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

def create_student_flow_scheduler(cfg: StudentFlowConfig) -> FlowMatchEulerDiscreteScheduler:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        shift=cfg.shift,
        use_dynamic_shifting=cfg.use_dynamic_shifting,
    )
    # For training we want a valid timestep table to sample from.
    scheduler.set_timesteps(cfg.num_train_timesteps)
    return scheduler

@dataclass
class StudentFlowForwardOutput:
    latents: torch.Tensor
    noisy_latents: torch.Tensor
    timesteps: torch.Tensor              # actual scheduler timesteps, float-like
    sigmas: torch.Tensor                 # [B, 1, 1, 1]
    noise: torch.Tensor                  # sampled noise
    target: torch.Tensor                 # flow target = noise - x0
    model_pred: torch.Tensor             # student predicted flow
    pred_x0_latent: torch.Tensor         # x0 = xt - sigma * pred
    pred_image: torch.Tensor
    clean_recon: Optional[torch.Tensor] = None


class StudentModelFlow(nn.Module):
    """
    Student U-Net trained with FLUX-style flow-matching semantics.

    Key equations:
      noisy_latents = scheduler.scale_noise(x0, t, noise)
      target        = noise - x0
      x0_hat        = x_t - sigma * model_pred

    This keeps the student architecture as U-Net, but changes the training objective
    to be closer to the FLUX teacher.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: nn.Module,
        scheduler: FlowMatchEulerDiscreteScheduler,
        latent_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.latent_scaling_factor = latent_scaling_factor

        # Ensure a timestep schedule exists for training-time sampling.
        if not hasattr(self.scheduler, "timesteps") or self.scheduler.timesteps is None or len(self.scheduler.timesteps) == 0:
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

    def encode_image(self, pixel_values: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor:
        posterior = self.vae.encode(pixel_values).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        latents = latents * self.latent_scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        image = self.vae.decode(latents).sample
        return image

    def sample_training_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample actual scheduler timesteps, not integer indices.

        FlowMatchEulerDiscreteScheduler expects timesteps drawn from scheduler.timesteps,
        and its step() explicitly rejects raw integer indices. :contentReference[oaicite:4]{index=4}
        """
        schedule_timesteps = self.scheduler.timesteps.to(device=device)
        idx = torch.randint(
            low=0,
            high=schedule_timesteps.shape[0],
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        timesteps = schedule_timesteps[idx]
        return timesteps

    def sigma_from_timesteps(self, timesteps: torch.Tensor, sample_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Map each sampled timestep to the scheduler sigma used by scale_noise().
        """
        schedule_timesteps = self.scheduler.timesteps.to(device=device)
        schedule_sigmas = self.scheduler.sigmas.to(device=device, dtype=sample_dtype)

        # scheduler.index_for_timestep(...) works one timestep at a time
        indices = [self.scheduler.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        sigmas = schedule_sigmas[torch.tensor(indices, device=device, dtype=torch.long)]
        return sigmas.view(-1, 1, 1, 1)

    def compute_target(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Flow-matching / velocity-style target consistent with:
            x_t = sigma * noise + (1 - sigma) * x0
        so:
            x_t = x0 + sigma * (noise - x0)
        which gives target:
            v* = noise - x0
        """
        return noise - latents

    def predict_x0_from_model_pred(
        self,
        noisy_latents: torch.Tensor,
        model_pred: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistent with FlowMatchEulerDiscreteScheduler step logic:
            x0 = sample - current_sigma * model_output
        in the scheduler's stochastic branch. :contentReference[oaicite:5]{index=5}
        """
        x0 = noisy_latents - sigmas * model_pred
        return x0

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        decode_clean_latents: bool = False,
    ) -> StudentFlowForwardOutput:
        encoder_hidden_states = self.encode_text(input_ids, attention_mask=attention_mask)
        latents = self.encode_image(pixel_values, sample_posterior=sample_posterior)

        if noise is None:
            noise = torch.randn_like(latents)

        bsz = latents.shape[0]
        if timesteps is None:
            timesteps = self.sample_training_timesteps(bsz, latents.device)

        # Flow forward/noising:
        # x_t = sigma * noise + (1 - sigma) * x0
        noisy_latents = self.scheduler.scale_noise(
            sample=latents,
            timestep=timesteps,
            noise=noise,
        )

        sigmas = self.sigma_from_timesteps(
            timesteps=timesteps,
            sample_dtype=latents.dtype,
            device=latents.device,
        )

        # Student U-Net predicts flow / velocity
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        target = self.compute_target(latents, noise)
        pred_x0_latent = self.predict_x0_from_model_pred(
            noisy_latents=noisy_latents,
            model_pred=model_pred,
            sigmas=sigmas,
        )
        pred_image = self.decode_latents(pred_x0_latent)

        clean_recon = None
        if decode_clean_latents:
            clean_recon = self.decode_latents(latents)

        return StudentFlowForwardOutput(
            latents=latents,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            sigmas=sigmas,
            noise=noise,
            target=target,
            model_pred=model_pred,
            pred_x0_latent=pred_x0_latent,
            pred_image=pred_image,
            clean_recon=clean_recon,
        )