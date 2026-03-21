from dataclasses import dataclass
from typing import Optional

@dataclass
class StageBConfig:
    stageB_steps: int = 100_000
    grad_accum_steps: int = 1
    mixed_precision: str = "bf16"
    output_dir: str = "./outputs_stageB"
    project_name: str = "toc_sr_stageB"
    log_every: int = 50

    num_train_timesteps: int = 1000
    max_grad_norm: float = 1.0

    lr_unet: float = 1e-4
    lr_vae_decoder: float = 5e-5
    lr_vae_encoder: float = 1e-5
    lr_bridge: float = 1e-4
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    w_teacher_pred: float = 1.0
    w_task: float = 0.5
    w_x0_img: float = 0.5
    w_lpips: float = 0.1
    w_latent: float = 0.05
    w_clean: float = 0.05

    sample_posterior_student: bool = True
    sample_posterior_teacher: bool = False

import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm


def mse_loss(x, y):
    return F.mse_loss(x.float(), y.float(), reduction="mean")


def l1_loss(x, y):
    return F.l1_loss(x.float(), y.float(), reduction="mean")


def train_stageB_flux_kd(
    cfg,
    student,
    teacher,          # FluxTeacherForKD
    dataloader,
    optimizer,
    bridge_pred,      # BCHW bridge
    bridge_lat=None,  # optional BCHW bridge
    lpips_model=None,
):
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with=LoggerType.TENSORBOARD,
        project_dir=cfg.output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name=cfg.project_name,
        config={
            "stageB_steps": cfg.stageB_steps,
            "grad_accum_steps": cfg.grad_accum_steps,
            "lr_unet": cfg.lr_unet,
            "lr_vae_decoder": cfg.lr_vae_decoder,
            "lr_vae_encoder": cfg.lr_vae_encoder,
            "lr_bridge": cfg.lr_bridge,
            "w_teacher_pred": cfg.w_teacher_pred,
            "w_task": cfg.w_task,
            "w_x0_img": cfg.w_x0_img,
            "w_lpips": cfg.w_lpips,
            "w_latent": cfg.w_latent,
            "w_clean": cfg.w_clean,
        },
    )

    teacher.to(accelerator.device)
    teacher.eval()

    if lpips_model is not None:
        lpips_model.to(accelerator.device)
        lpips_model.eval()

    modules_to_prepare = [student, bridge_pred, optimizer, dataloader]
    if bridge_lat is not None:
        modules_to_prepare.insert(2, bridge_lat)

    prepared = accelerator.prepare(*modules_to_prepare)

    if bridge_lat is not None:
        student, bridge_pred, bridge_lat, optimizer, dataloader = prepared
    else:
        student, bridge_pred, optimizer, dataloader = prepared

    student.train()
    bridge_pred.train()
    if bridge_lat is not None:
        bridge_lat.train()

    global_step = 0
    progress_bar = tqdm(
        total=cfg.stageB_steps,
        disable=not accelerator.is_local_main_process,
        desc="Stage-B",
    )

    while global_step < cfg.stageB_steps:
        for batch in dataloader:
            with accelerator.accumulate(student):
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device, non_blocking=True)

                prompts = batch.get("captions_for_model", batch.get("captions", None))
                if prompts is None:
                    prompts = [""] * pixel_values.shape[0]

                # --------------------------------------------------
                # Build shared timesteps + noise in BCHW latent space
                # --------------------------------------------------
                with torch.no_grad():
                    teacher_latents_bchw = teacher.encode_image(
                        pixel_values,
                        sample_posterior=getattr(cfg, "sample_posterior_teacher", False),
                    )

                noise_bchw = torch.randn_like(teacher_latents_bchw)
                bsz = teacher_latents_bchw.shape[0]
                timesteps = torch.randint(
                    low=0,
                    high=cfg.num_train_timesteps,
                    size=(bsz,),
                    device=accelerator.device,
                    dtype=torch.long,
                )

                # --------------------------------------------------
                # Teacher forward
                # --------------------------------------------------
                with torch.no_grad():
                    teacher_outputs = teacher.forward_kd(
                        pixel_values=pixel_values,
                        prompts=prompts,
                        timesteps=timesteps,
                        noise=noise_bchw,
                        sample_posterior=getattr(cfg, "sample_posterior_teacher", False),
                    )

                # teacher_outputs.teacher_model_pred is PACKED [B, N, 64]
                # Convert it back to BCHW latent space so we can compare with student U-Net output.
                teacher_model_pred_bchw = teacher._unpack_teacher_latents(
                    teacher_outputs.teacher_model_pred,
                    latent_h=teacher_outputs.teacher_x0_latent.shape[-2],
                    latent_w=teacher_outputs.teacher_x0_latent.shape[-1],
                )

                # --------------------------------------------------
                # Student forward
                # --------------------------------------------------
                student_outputs = student(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timesteps=timesteps,
                    noise=noise_bchw,
                    sample_posterior=getattr(cfg, "sample_posterior_student", True),
                    decode_clean_latents=True,
                )

                # --------------------------------------------------
                # 1) Main teacher prediction KD
                # teacher pred -> bridge -> student pred domain
                # --------------------------------------------------
                teacher_pred_for_student = bridge_pred(teacher_model_pred_bchw)
                loss_teacher_pred = mse_loss(student_outputs.model_pred, teacher_pred_for_student)

                # --------------------------------------------------
                # 2) Student task loss
                # Keep a smaller native task anchor
                # This works whether target is DDPM epsilon or flow target,
                # depending on how StudentModel is implemented.
                # --------------------------------------------------
                loss_task = mse_loss(student_outputs.model_pred, student_outputs.target)

                # --------------------------------------------------
                # 3) Teacher x0 image KD
                # --------------------------------------------------
                loss_x0_img = l1_loss(student_outputs.pred_image, teacher_outputs.teacher_x0_image)

                # --------------------------------------------------
                # 4) LPIPS on teacher x0 image
                # --------------------------------------------------
                loss_lpips = torch.tensor(0.0, device=accelerator.device)
                if lpips_model is not None:
                    loss_lpips = lpips_model(
                        student_outputs.pred_image,
                        teacher_outputs.teacher_x0_image,
                    ).mean()

                # --------------------------------------------------
                # 5) Optional latent consistency
                # teacher clean latent -> bridge -> student latent domain
                # --------------------------------------------------
                loss_latent = torch.tensor(0.0, device=accelerator.device)
                if bridge_lat is not None:
                    teacher_lat_for_student = bridge_lat(teacher_outputs.teacher_latents)
                    loss_latent = mse_loss(student_outputs.latents, teacher_lat_for_student)

                # --------------------------------------------------
                # 6) Optional light clean reconstruction regularizer
                # --------------------------------------------------
                loss_clean = torch.tensor(0.0, device=accelerator.device)
                if student_outputs.clean_recon is not None:
                    loss_clean = l1_loss(student_outputs.clean_recon, pixel_values)

                # --------------------------------------------------
                # Total loss
                # --------------------------------------------------
                loss = (
                    cfg.w_teacher_pred * loss_teacher_pred
                    + cfg.w_task * loss_task
                    + cfg.w_x0_img * loss_x0_img
                    + cfg.w_lpips * loss_lpips
                    + cfg.w_latent * loss_latent
                    + cfg.w_clean * loss_clean
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grads = list(student.parameters()) + list(bridge_pred.parameters())
                    if bridge_lat is not None:
                        grads += list(bridge_lat.parameters())
                    accelerator.clip_grad_norm_(grads, cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % cfg.log_every == 0:
                        accelerator.log(
                            {
                                "stageB/loss": loss.detach().float().item(),
                                "stageB/loss_teacher_pred": loss_teacher_pred.detach().float().item(),
                                "stageB/loss_task": loss_task.detach().float().item(),
                                "stageB/loss_x0_img": loss_x0_img.detach().float().item(),
                                "stageB/loss_lpips": loss_lpips.detach().float().item(),
                                "stageB/loss_latent": loss_latent.detach().float().item(),
                                "stageB/loss_clean": loss_clean.detach().float().item(),
                            },
                            step=global_step,
                        )

                    if global_step >= cfg.stageB_steps:
                        break

            if global_step >= cfg.stageB_steps:
                break

    accelerator.end_training()
    return accelerator.unwrap_model(student)