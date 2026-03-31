from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm
import torch
import torch.nn.functional as F

import os
import torch

from torch.optim import AdamW


def make_t2i_unet_distill_optimizer(student, feature_adapters, cfg: T2IDistillConfig):
    params = [
        {
            "params": [p for p in student.unet.parameters() if p.requires_grad],
            "lr": cfg.lr_unet,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for p in feature_adapters.parameters() if p.requires_grad],
            "lr": cfg.lr_adapter,
            "weight_decay": cfg.weight_decay,
        },
    ]

    return AdamW(
        params,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

def save_t2i_distill_checkpoint(
    accelerator,
    student,
    feature_adapters,
    optimizer,
    global_step: int,
    output_dir: str,
):
    if not accelerator.is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "global_step": global_step,
        "student_unet": accelerator.unwrap_model(student).unet.state_dict(),
        "feature_adapters": accelerator.unwrap_model(feature_adapters).state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    path = os.path.join(output_dir, f"distill_step_{global_step}.pt")
    torch.save(ckpt, path)
    print(f"[distill] saved checkpoint: {path}")


def load_t2i_distill_checkpoint(
    ckpt_path: str,
    student,
    feature_adapters,
    optimizer=None,
    map_location="cpu",
):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    student.unet.load_state_dict(ckpt["student_unet"], strict=True)
    feature_adapters.load_state_dict(ckpt["feature_adapters"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("global_step", 0)

def maybe_build_lpips(use_lpips: bool = True):
    if not use_lpips:
        return None
    try:
        import lpips
        return lpips.LPIPS(net="vgg")
    except Exception:
        return None


def train_t2i_unet_distill(
    cfg: T2IDistillConfig,
    student,                # StudentModelFlow
    teacher_unet,           # frozen 700M teacher UNet
    dataloader,
    feature_adapters,       # FeatureAdapterBank
    feature_keys: list[str],
    optimizer,
    teacher_text_encoder=None,
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
            "train_steps": cfg.train_steps,
            "lr_unet": cfg.lr_unet,
            "lr_adapter": cfg.lr_adapter,
            "w_teacher_pred": cfg.w_teacher_pred,
            "w_feature": cfg.w_feature,
            "w_task": cfg.w_task,
            "w_x0_img": cfg.w_x0_img,
            "w_lpips": cfg.w_lpips,
            "w_gt_img": cfg.w_gt_img,
        },
    )

    # Freeze everything except student.unet + adapters
    student.vae.requires_grad_(False)
    student.text_encoder.requires_grad_(False)
    student.vae.eval()
    student.text_encoder.eval()

    teacher_unet.requires_grad_(False)
    teacher_unet.eval()
    teacher_unet.to(accelerator.device)

    if teacher_text_encoder is not None:
        teacher_text_encoder.requires_grad_(False)
        teacher_text_encoder.eval()
        teacher_text_encoder.to(accelerator.device)

    if lpips_model is None:
        lpips_model = maybe_build_lpips(use_lpips=(cfg.w_lpips > 0))
    if lpips_model is not None:
        lpips_model.to(accelerator.device)
        lpips_model.eval()

    student, feature_adapters, optimizer, dataloader = accelerator.prepare(
        student, feature_adapters, optimizer, dataloader
    )

    global_step = 0
    if cfg.resume_ckpt is not None:
        accelerator.wait_for_everyone()
        global_step = load_t2i_distill_checkpoint(
            cfg.resume_ckpt,
            student=accelerator.unwrap_model(student),
            feature_adapters=accelerator.unwrap_model(feature_adapters),
            optimizer=optimizer,
            map_location="cpu",
        )
        accelerator.wait_for_everyone()

    # Register hooks AFTER prepare so they attach to the wrapped modules' underlying module via DDP forward path.
    student_store = ActivationStore()
    teacher_store = ActivationStore()

    student_handles = register_unet_feature_hooks(accelerator.unwrap_model(student).unet, student_store)
    teacher_handles = register_unet_feature_hooks(teacher_unet, teacher_store)

    progress_bar = tqdm(
        total=cfg.train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="T2I-UNet-Distill",
    )

    while global_step < cfg.train_steps:
        for batch in dataloader:
            with accelerator.accumulate(student):
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device, non_blocking=True)

                # Clear feature caches every iteration
                student_store.clear_all()
                teacher_store.clear_all()

                # --------------------------------------------------
                # Shared fixed latent + text conditioning
                # --------------------------------------------------
                with torch.no_grad():
                    latents = accelerator.unwrap_model(student).encode_image(
                        pixel_values,
                        sample_posterior=cfg.sample_posterior_latents,
                    )

                    # Flow timesteps MUST come from scheduler.timesteps
                    sched = accelerator.unwrap_model(student).scheduler
                    if not hasattr(sched, "timesteps") or sched.timesteps is None or len(sched.timesteps) == 0:
                        sched.set_timesteps(sched.config.num_train_timesteps, device=accelerator.device)

                    schedule_timesteps = sched.timesteps.to(device=accelerator.device)
                    idx = torch.randint(
                        low=0,
                        high=schedule_timesteps.shape[0],
                        size=(latents.shape[0],),
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    timesteps = schedule_timesteps[idx]

                    noise = torch.randn_like(latents)
                    noisy_latents = sched.scale_noise(
                        sample=latents,
                        timestep=timesteps,
                        noise=noise,
                    )

                    if cfg.reuse_student_text_condition or teacher_text_encoder is None:
                        teacher_hidden = None
                        encoder_hidden_states = accelerator.unwrap_model(student).encode_text(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                    else:
                        encoder_hidden_states = accelerator.unwrap_model(student).encode_text(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_hidden_out = teacher_text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_hidden = (
                            teacher_hidden_out.last_hidden_state
                            if hasattr(teacher_hidden_out, "last_hidden_state")
                            else teacher_hidden_out[0]
                        )

                if teacher_hidden is None:
                    teacher_hidden = encoder_hidden_states.detach()

                # --------------------------------------------------
                # Teacher forward (frozen)
                # --------------------------------------------------
                teacher_dtype = next(teacher_unet.parameters()).dtype
                with torch.no_grad():
                    teacher_model_pred = teacher_unet(
                        noisy_latents.to(dtype=teacher_dtype),
                        timesteps.to(device=accelerator.device),
                        encoder_hidden_states=teacher_hidden.to(dtype=teacher_dtype),
                        return_dict=True,
                    ).sample

                # --------------------------------------------------
                # Student forward (only UNet trainable)
                # --------------------------------------------------
                student_model_pred = student.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True,
                ).sample

                sigmas = accelerator.unwrap_model(student).sigma_from_timesteps(
                    timesteps=timesteps,
                    sample_dtype=noisy_latents.dtype,
                    device=accelerator.device,
                )

                target = accelerator.unwrap_model(student).compute_target(latents, noise)

                student_x0 = accelerator.unwrap_model(student).predict_x0_from_model_pred(
                    noisy_latents=noisy_latents,
                    model_pred=student_model_pred,
                    sigmas=sigmas,
                )
                teacher_x0 = accelerator.unwrap_model(student).predict_x0_from_model_pred(
                    noisy_latents=noisy_latents,
                    model_pred=teacher_model_pred.to(dtype=noisy_latents.dtype),
                    sigmas=sigmas,
                )

                # Student decoded x0 keeps gradient through fixed VAE
                student_img = accelerator.unwrap_model(student).decode_latents(student_x0)

                # Teacher decoded x0 is target only
                with torch.no_grad():
                    teacher_img = accelerator.unwrap_model(student).decode_latents(teacher_x0)

                # --------------------------------------------------
                # Losses
                # --------------------------------------------------
                loss_teacher_pred = F.mse_loss(
                    student_model_pred.float(),
                    teacher_model_pred.float(),
                    reduction="mean",
                )

                loss_feature = feature_kd_loss(
                    student_store=student_store,
                    teacher_store=teacher_store,
                    adapters=feature_adapters,
                    feature_keys=feature_keys,
                )

                loss_task = F.mse_loss(
                    student_model_pred.float(),
                    target.float(),
                    reduction="mean",
                )

                loss_x0_img = F.l1_loss(
                    student_img.float(),
                    teacher_img.float(),
                    reduction="mean",
                )

                loss_lpips = torch.tensor(0.0, device=accelerator.device)
                if lpips_model is not None and cfg.w_lpips > 0:
                    loss_lpips = lpips_model(student_img, teacher_img).mean()

                loss_gt_img = F.l1_loss(
                    student_img.float(),
                    pixel_values.float(),
                    reduction="mean",
                )

                loss = (
                    cfg.w_teacher_pred * loss_teacher_pred
                    + cfg.w_feature * loss_feature
                    + cfg.w_task * loss_task
                    + cfg.w_x0_img * loss_x0_img
                    + cfg.w_lpips * loss_lpips
                    + cfg.w_gt_img * loss_gt_img
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grads = list(student.unet.parameters()) + list(feature_adapters.parameters())
                    accelerator.clip_grad_norm_(grads, cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % cfg.log_every == 0:
                        accelerator.log(
                            {
                                "distill/loss": loss.detach().float().item(),
                                "distill/loss_teacher_pred": loss_teacher_pred.detach().float().item(),
                                "distill/loss_feature": loss_feature.detach().float().item(),
                                "distill/loss_task": loss_task.detach().float().item(),
                                "distill/loss_x0_img": loss_x0_img.detach().float().item(),
                                "distill/loss_lpips": loss_lpips.detach().float().item(),
                                "distill/loss_gt_img": loss_gt_img.detach().float().item(),
                            },
                            step=global_step,
                        )

                    if global_step % cfg.save_every == 0:
                        save_t2i_distill_checkpoint(
                            accelerator=accelerator,
                            student=student,
                            feature_adapters=feature_adapters,
                            optimizer=optimizer,
                            global_step=global_step,
                            output_dir=cfg.output_dir,
                        )

                    if global_step >= cfg.train_steps:
                        break

            if global_step >= cfg.train_steps:
                break

    accelerator.wait_for_everyone()

    save_t2i_distill_checkpoint(
        accelerator=accelerator,
        student=student,
        feature_adapters=feature_adapters,
        optimizer=optimizer,
        global_step=global_step,
        output_dir=cfg.output_dir,
    )

    remove_hooks(student_handles)
    remove_hooks(teacher_handles)

    accelerator.end_training()
    return accelerator.unwrap_model(student)