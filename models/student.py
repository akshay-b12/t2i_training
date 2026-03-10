import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


# -------------------------
# Student build config
# -------------------------

@dataclass
class StudentConfig:
    pretrained_unet_id: Optional[str] = None
    pretrained_unet_subfolder: str = "unet"

    pretrained_vae_id: Optional[str] = None
    pretrained_vae_subfolder: Optional[str] = None  # set if needed

    sample_size: Optional[int] = None
    cross_attention_dim: int = 768
    latent_channels: int = 16

    # If building natively from config
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Tuple[int, ...] = (5, 10, 20, 20)
    norm_num_groups: int = 32
    use_linear_projection: bool = False

    # scheduler
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"   # or "v_prediction"

    # latent scaling
    latent_scaling_factor: float = 1.0

    # if loading a pretrained 4ch unet and inflating it
    inflate_method: str = "tile_div"   # "tile_div" | "first4_zero_rest"


# -------------------------
# Helpers for 4ch -> 16ch inflation
# -------------------------

def _inflate_conv_in_weight(old_w: torch.Tensor, new_in_ch: int, method: str = "tile_div") -> torch.Tensor:
    """
    old_w: [out_ch, old_in_ch, k, k]
    returns new_w: [out_ch, new_in_ch, k, k]
    """
    out_ch, old_in_ch, kh, kw = old_w.shape
    new_w = torch.zeros((out_ch, new_in_ch, kh, kw), dtype=old_w.dtype, device=old_w.device)

    if method == "first4_zero_rest":
        new_w[:, :old_in_ch] = old_w
        return new_w

    if method == "tile_div":
        reps = (new_in_ch + old_in_ch - 1) // old_in_ch
        tiled = old_w.repeat(1, reps, 1, 1)[:, :new_in_ch]
        # preserve activation scale approximately
        scale = old_in_ch / float(new_in_ch)
        new_w.copy_(tiled * scale)
        return new_w

    raise ValueError(f"Unknown inflate method: {method}")


def _inflate_conv_out_weight(old_w: torch.Tensor, new_out_ch: int, method: str = "tile_div") -> torch.Tensor:
    """
    old_w: [old_out_ch, in_ch, k, k]
    returns new_w: [new_out_ch, in_ch, k, k]
    """
    old_out_ch, in_ch, kh, kw = old_w.shape
    new_w = torch.zeros((new_out_ch, in_ch, kh, kw), dtype=old_w.dtype, device=old_w.device)

    if method == "first4_zero_rest":
        new_w[:old_out_ch] = old_w
        return new_w

    if method == "tile_div":
        reps = (new_out_ch + old_out_ch - 1) // old_out_ch
        tiled = old_w.repeat(reps, 1, 1, 1)[:new_out_ch]
        scale = old_out_ch / float(new_out_ch)
        new_w.copy_(tiled * scale)
        return new_w

    raise ValueError(f"Unknown inflate method: {method}")


def _inflate_bias(old_b: torch.Tensor, new_ch: int, method: str = "tile_div") -> torch.Tensor:
    old_ch = old_b.shape[0]
    new_b = torch.zeros((new_ch,), dtype=old_b.dtype, device=old_b.device)

    if method == "first4_zero_rest":
        new_b[:old_ch] = old_b
        return new_b

    if method == "tile_div":
        reps = (new_ch + old_ch - 1) // old_ch
        tiled = old_b.repeat(reps)[:new_ch]
        scale = old_ch / float(new_ch)
        new_b.copy_(tiled * scale)
        return new_b

    raise ValueError(f"Unknown inflate method: {method}")


# -------------------------
# Student UNet creation
# -------------------------

def create_native_16ch_unet(cfg: StudentConfig) -> UNet2DConditionModel:
    model = UNet2DConditionModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.latent_channels,
        out_channels=cfg.latent_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
        cross_attention_dim=cfg.cross_attention_dim,
        attention_head_dim=cfg.attention_head_dim,
        norm_num_groups=cfg.norm_num_groups,
        use_linear_projection=cfg.use_linear_projection,
    )
    return model


def inflate_pretrained_unet_to_16ch(cfg: StudentConfig, torch_dtype=torch.float32) -> UNet2DConditionModel:
    if cfg.pretrained_unet_id is None:
        raise ValueError("cfg.pretrained_unet_id must be set for inflate_pretrained_unet_to_16ch.")

    old_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_unet_id,
        subfolder=cfg.pretrained_unet_subfolder,
        torch_dtype=torch_dtype,
    )

    new_config = old_unet.config.copy()
    new_config["in_channels"] = cfg.latent_channels
    new_config["out_channels"] = cfg.latent_channels

    new_unet = UNet2DConditionModel(**new_config)
    new_state = new_unet.state_dict()
    old_state = old_unet.state_dict()

    for k, v in old_state.items():
        if k == "conv_in.weight":
            new_state[k] = _inflate_conv_in_weight(v, cfg.latent_channels, method=cfg.inflate_method)
        elif k == "conv_in.bias":
            new_state[k] = v.clone()
        elif k == "conv_out.weight":
            new_state[k] = _inflate_conv_out_weight(v, cfg.latent_channels, method=cfg.inflate_method)
        elif k == "conv_out.bias":
            new_state[k] = _inflate_bias(v, cfg.latent_channels, method=cfg.inflate_method)
        elif k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
        # else: silently skip incompatible tensors

    missing, unexpected = new_unet.load_state_dict(new_state, strict=False)
    print("[inflate_pretrained_unet_to_16ch] missing:", len(missing), "unexpected:", len(unexpected))
    return new_unet


def create_student_scheduler(cfg: StudentConfig) -> DDPMScheduler:
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    return scheduler


# -------------------------
# Student wrapper
# -------------------------

@dataclass
class StudentForwardOutput:
    latents: torch.Tensor
    noisy_latents: torch.Tensor
    timesteps: torch.Tensor
    target: torch.Tensor
    model_pred: torch.Tensor
    pred_x0_latent: torch.Tensor
    pred_image: torch.Tensor
    clean_recon: Optional[torch.Tensor] = None


class StudentModel(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: nn.Module,
        scheduler: DDPMScheduler,
        latent_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.latent_scaling_factor = latent_scaling_factor

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

    def compute_target(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.scheduler.config.prediction_type == "epsilon":
            return noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.scheduler.config.prediction_type}")

    def predict_x0_from_model_pred(
        self,
        noisy_latents: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.scheduler.alphas_cumprod.to(noisy_latents.device)
        a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)

        if self.scheduler.config.prediction_type == "epsilon":
            x0 = (noisy_latents - sqrt_one_minus_a_t * model_pred) / sqrt_a_t.clamp(min=1e-6)
            return x0

        elif self.scheduler.config.prediction_type == "v_prediction":
            # x0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
            x0 = sqrt_a_t * noisy_latents - sqrt_one_minus_a_t * model_pred
            return x0

        raise ValueError(f"Unsupported prediction_type: {self.scheduler.config.prediction_type}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        decode_clean_latents: bool = False,
    ) -> StudentForwardOutput:
        encoder_hidden_states = self.encode_text(input_ids, attention_mask=attention_mask)
        latents = self.encode_image(pixel_values, sample_posterior=sample_posterior)

        if noise is None:
            noise = torch.randn_like(latents)

        if timesteps is None:
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.long,
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        target = self.compute_target(latents, noise, timesteps)
        pred_x0_latent = self.predict_x0_from_model_pred(noisy_latents, model_pred, timesteps, noise)
        pred_image = self.decode_latents(pred_x0_latent)

        clean_recon = None
        if decode_clean_latents:
            clean_recon = self.decode_latents(latents)

        return StudentForwardOutput(
            latents=latents,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            target=target,
            model_pred=model_pred,
            pred_x0_latent=pred_x0_latent,
            pred_image=pred_image,
            clean_recon=clean_recon,
        )
    
'''
from transformers import CLIPTextModel, CLIPTokenizer

# --- tokenizer / text encoder ---
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")

# --- student config ---
student_cfg = StudentConfig(
    pretrained_unet_id="runwayml/stable-diffusion-v1-5",  # or your NanoSD source
    pretrained_unet_subfolder="unet",
    latent_channels=16,
    cross_attention_dim=text_encoder.config.hidden_size,
    prediction_type="epsilon",
    latent_scaling_factor=1.0,  # replace after calibration
)

# Option A: native 16ch from scratch-like config
# student_unet = create_native_16ch_unet(student_cfg)

# Option B: inflate pretrained 4ch unet to 16ch
student_unet = inflate_pretrained_unet_to_16ch(student_cfg, torch_dtype=torch.float32)

# Student VAE placeholder: replace with your distilled small 16ch VAE repo/local path
student_vae = AutoencoderKL.from_pretrained(
    "YOUR_SMALL_16CH_VAE_REPO_OR_LOCAL_PATH",
    subfolder=None,
)

student_scheduler = create_student_scheduler(student_cfg)

student_model = StudentModel(
    unet=student_unet,
    vae=student_vae,
    text_encoder=text_encoder,
    scheduler=student_scheduler,
    latent_scaling_factor=student_cfg.latent_scaling_factor,
)
'''