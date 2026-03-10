# main_train.py
import torch
from contextlib import nullcontext

from data.streaming_laion_pop import make_train_dataloader
from models.student import StudentModel
from models.teacher import TeacherWrapper
from models.bridges import LatentBridge
from train.utils import make_optimizer

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def run_training(cfg, tokenizer, student_unet, student_vae, text_encoder, scheduler, teacher_modules, lpips_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = StudentModel(
        unet=student_unet,
        vae=student_vae,
        text_encoder=text_encoder,
        scheduler=scheduler,
        latent_scaling_factor=cfg.latent_scaling_factor,
    ).to(device)

    teacher = TeacherWrapper(teacher_modules).to(device)

    # Optional latent bridges
    bridge_in = None
    bridge_out = None

    dataloader = make_train_dataloader(cfg, tokenizer)

    # -------- Stage 1 --------
    optimizer = make_optimizer(cfg, student, stage=1, bridge_in=bridge_in, bridge_out=bridge_out)
    train_stage1(cfg, student, dataloader, optimizer, lpips_model=lpips_model, accelerator=None)

    # -------- Stage 2 --------
    optimizer = make_optimizer(cfg, student, stage=2, bridge_in=bridge_in, bridge_out=bridge_out)
    train_stage2(cfg, student, teacher, dataloader, optimizer, lpips_model=lpips_model,
                 bridge_in=bridge_in, bridge_out=bridge_out, accelerator=None)

    # -------- Stage 3 --------
    optimizer = make_optimizer(cfg, student, stage=3, bridge_in=bridge_in, bridge_out=bridge_out)
    train_stage3(cfg, student, teacher, dataloader, optimizer, lpips_model=lpips_model,
                 bridge_in=bridge_in, bridge_out=bridge_out, accelerator=None)

    return student