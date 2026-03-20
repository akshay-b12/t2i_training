import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm

@dataclass
class Stage2TrainConfig:
    # training length
    stage2_steps: int = 100_000
    grad_accum_steps: int = 1

    # optimizer
    lr_unet: float = 1e-4
    lr_vae_decoder: float = 5e-5
    lr_vae_encoder: float = 1e-5
    lr_bridge: float = 1e-4

    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # precision
    mixed_precision: str = "fp16"   # "fp16", "bf16", or "no"

    # logging/checkpointing
    output_dir: str = "./outputs_stage2"
    project_name: str = "toc_sr_stage2"
    log_every: int = 50
    save_every: int = 5000

    # teacher usage
    teacher_every_n_steps: int = 1
    sample_posterior_student: bool = True
    sample_posterior_teacher: bool = False

    # Stage-2 weights
    w_diff: float = 1.0
    w_rec: float = 0.25
    w_clean_recon: float = 0.15
    w_teacher_recon: float = 0.75
    w_teacher_latent: float = 0.10
    w_lpips_kd: float = 0.20

    # optional warmup schedule for early stage-2
    use_warmup_weights: bool = True
    warmup_steps: int = 5000

    # warmup weights
    warm_w_rec: float = 0.30
    warm_w_clean_recon: float = 0.20
    warm_w_teacher_recon: float = 0.40
    warm_w_teacher_latent: float = 0.15
    warm_w_lpips_kd: float = 0.15

    # resume
    resume_ckpt: Optional[str] = None


# bridges
bridge_in = ResidualLatentBridge(channels=16, hidden_channels=32)
bridge_out = ResidualLatentBridge(channels=16, hidden_channels=32)

# optimizer
cfg = Stage2TrainConfig(
    stage2_steps=100_000,
    grad_accum_steps=2,
    mixed_precision="fp16",   # or "bf16"
    output_dir="./outputs_stage2_flux1_16ch",
    project_name="toc_sr_stage2",
    log_every=50,
    save_every=5000,

    lr_unet=1e-4,
    lr_vae_decoder=5e-5,
    lr_vae_encoder=1e-5,
    lr_bridge=1e-4,

    w_diff=1.0,
    w_rec=0.25,
    w_clean_recon=0.15,
    w_teacher_recon=0.75,
    w_teacher_latent=0.10,
    w_lpips_kd=0.20,

    use_warmup_weights=True,
    warmup_steps=5000,
)

optimizer = make_stage2_optimizer(
    student=student,
    cfg=cfg,
    bridge_in=bridge_in,
    bridge_out=bridge_out,
)

lpips_model = maybe_build_lpips(use_lpips=True)

trained_student = train_stage2_full(
    cfg=cfg,
    student=student,
    teacher=teacher,
    dataloader=train_loader,
    optimizer=optimizer,
    bridge_in=bridge_in,
    bridge_out=bridge_out,
    lpips_model=lpips_model,
)