from accelerate import Accelerator
from accelerate.utils import LoggerType
from contextlib import nullcontext
from tqdm import tqdm
import os
import torch

from torch.optim import AdamW

def make_stageB_optimizer(student, cfg, bridge_pred, bridge_lat=None):
    param_groups = [
        {"params": [p for p in student.unet.parameters() if p.requires_grad], "lr": cfg.lr_unet, "weight_decay": cfg.weight_decay},
    ]

    if hasattr(student.vae, "decoder"):
        param_groups.append(
            {"params": [p for p in student.vae.decoder.parameters() if p.requires_grad], "lr": cfg.lr_vae_decoder, "weight_decay": cfg.weight_decay}
        )
    if hasattr(student.vae, "encoder"):
        param_groups.append(
            {"params": [p for p in student.vae.encoder.parameters() if p.requires_grad], "lr": cfg.lr_vae_encoder, "weight_decay": cfg.weight_decay}
        )

    param_groups.append(
        {"params": [p for p in bridge_pred.parameters() if p.requires_grad], "lr": cfg.lr_bridge, "weight_decay": cfg.weight_decay}
    )

    if bridge_lat is not None:
        param_groups.append(
            {"params": [p for p in bridge_lat.parameters() if p.requires_grad], "lr": cfg.lr_bridge, "weight_decay": cfg.weight_decay}
        )

    return AdamW(
        param_groups,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

def train_stageB_flux_kd(
    cfg,
    student,
    teacher,          # FluxTeacherForKD
    dataloader,
    optimizer,
    bridge_pred,
    bridge_lat=None,
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

    student, bridge_pred, optimizer, dataloader = accelerator.prepare(
        student, bridge_pred, optimizer, dataloader
    )

    if bridge_lat is not None:
        bridge_lat = accelerator.prepare(bridge_lat)

    student.train()
    bridge_pred.train()
    if bridge_lat is not None:
        bridge_lat.train()

    global_step = 0
    pbar = tqdm(total=cfg.stageB_steps, disable=not accelerator.is_local_main_process, desc="Stage-B")

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

                # Use the same timesteps/noise for teacher and student
                with torch.no_grad():
                    teacher_latents = teacher.encode_image(
                        pixel_values,
                        sample_posterior=getattr(cfg, "sample_posterior_teacher", False),
                    )

                noise = torch.randn_like(teacher_latents)
                bsz = teacher_latents.shape[0]
                timesteps = torch.randint(
                    0,
                    cfg.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                    dtype=torch.long,
                )

                # teacher forward
                with torch.no_grad():
                    teacher_outputs = teacher.forward_kd(
                        pixel_values=pixel_values,
                        prompts=prompts,
                        timesteps=timesteps,
                        noise=noise,
                        sample_posterior=getattr(cfg, "sample_posterior_teacher", False),
                    )

                # student forward with same t/noise
                student_outputs = student(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timesteps=timesteps,
                    noise=noise,
                    sample_posterior=getattr(cfg, "sample_posterior_student", True),
                    decode_clean_latents=True,
                )

                # 1) main teacher prediction KD
                teacher_pred_for_student = bridge_pred(teacher_outputs.teacher_model_pred)
                loss_teacher_pred = mse_loss(student_outputs.model_pred, teacher_pred_for_student)

                # 2) student task loss
                loss_task = mse_loss(student_outputs.model_pred, student_outputs.target)

                # 3) teacher x0 image KD
                loss_x0_img = l1_loss(student_outputs.pred_image, teacher_outputs.teacher_x0_image)

                # 4) LPIPS
                loss_lpips = torch.tensor(0.0, device=accelerator.device)
                if lpips_model is not None:
                    loss_lpips = lpips_model(
                        student_outputs.pred_image,
                        teacher_outputs.teacher_x0_image,
                    ).mean()

                # 5) optional latent consistency
                loss_latent = torch.tensor(0.0, device=accelerator.device)
                if bridge_lat is not None:
                    teacher_lat_for_student = bridge_lat(teacher_outputs.teacher_latents)
                    loss_latent = mse_loss(student_outputs.latents, teacher_lat_for_student)

                # 6) optional light clean reconstruction
                loss_clean = torch.tensor(0.0, device=accelerator.device)
                if student_outputs.clean_recon is not None:
                    loss_clean = l1_loss(student_outputs.clean_recon, pixel_values)

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
                    pbar.update(1)

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