import torch
from torch.optim import AdamW
import os

def save_stage2_checkpoint(
    accelerator: Accelerator,
    student,
    optimizer,
    global_step: int,
    output_dir: str,
    bridge_in=None,
    bridge_out=None,
):
    if not accelerator.is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    unwrapped_student = accelerator.unwrap_model(student)
    ckpt = {
        "global_step": global_step,
        "student_unet": unwrapped_student.unet.state_dict(),
        "student_vae": unwrapped_student.vae.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if bridge_in is not None:
        ckpt["bridge_in"] = accelerator.unwrap_model(bridge_in).state_dict()
    if bridge_out is not None:
        ckpt["bridge_out"] = accelerator.unwrap_model(bridge_out).state_dict()

    path = os.path.join(output_dir, f"stage2_step_{global_step}.pt")
    torch.save(ckpt, path)
    print(f"[stage2] saved checkpoint: {path}")


def load_stage2_checkpoint(
    ckpt_path: str,
    student,
    optimizer=None,
    bridge_in=None,
    bridge_out=None,
    map_location="cpu",
):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    student.unet.load_state_dict(ckpt["student_unet"], strict=True)
    student.vae.load_state_dict(ckpt["student_vae"], strict=True)

    if bridge_in is not None and "bridge_in" in ckpt:
        bridge_in.load_state_dict(ckpt["bridge_in"], strict=True)
    if bridge_out is not None and "bridge_out" in ckpt:
        bridge_out.load_state_dict(ckpt["bridge_out"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    global_step = ckpt.get("global_step", 0)
    return global_step

def make_stage2_optimizer(
    student,
    cfg,
    bridge_in=None,
    bridge_out=None,
):
    """
    Stage-2 optimizer:
      - U-Net: main trainable block
      - VAE decoder: moderate LR
      - VAE encoder: lower LR
      - Bridges: same LR as U-Net by default
    """

    param_groups = []

    # 1) U-Net
    unet_params = [p for p in student.unet.parameters() if p.requires_grad]
    if len(unet_params) > 0:
        param_groups.append(
            {
                "params": unet_params,
                "lr": cfg.lr_unet,
                "weight_decay": cfg.weight_decay,
            }
        )

    # 2) VAE decoder
    if hasattr(student.vae, "decoder"):
        vae_decoder_params = [p for p in student.vae.decoder.parameters() if p.requires_grad]
        if len(vae_decoder_params) > 0:
            param_groups.append(
                {
                    "params": vae_decoder_params,
                    "lr": cfg.lr_vae_decoder,
                    "weight_decay": cfg.weight_decay,
                }
            )

    # 3) VAE encoder
    if hasattr(student.vae, "encoder"):
        vae_encoder_params = [p for p in student.vae.encoder.parameters() if p.requires_grad]
        if len(vae_encoder_params) > 0:
            param_groups.append(
                {
                    "params": vae_encoder_params,
                    "lr": cfg.lr_vae_encoder,
                    "weight_decay": cfg.weight_decay,
                }
            )

    # 4) bridge_in
    if bridge_in is not None:
        bridge_in_params = [p for p in bridge_in.parameters() if p.requires_grad]
        if len(bridge_in_params) > 0:
            param_groups.append(
                {
                    "params": bridge_in_params,
                    "lr": getattr(cfg, "lr_bridge", cfg.lr_unet),
                    "weight_decay": cfg.weight_decay,
                }
            )

    # 5) bridge_out
    if bridge_out is not None:
        bridge_out_params = [p for p in bridge_out.parameters() if p.requires_grad]
        if len(bridge_out_params) > 0:
            param_groups.append(
                {
                    "params": bridge_out_params,
                    "lr": getattr(cfg, "lr_bridge", cfg.lr_unet),
                    "weight_decay": cfg.weight_decay,
                }
            )

    optimizer = AdamW(
        param_groups,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    return optimizer

def train_stage2_full(
    cfg: Stage2TrainConfig,
    student,
    teacher,
    dataloader,
    optimizer,
    bridge_in=None,
    bridge_out=None,
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
            "stage2_steps": cfg.stage2_steps,
            "grad_accum_steps": cfg.grad_accum_steps,
            "lr_unet": cfg.lr_unet,
            "lr_vae_decoder": cfg.lr_vae_decoder,
            "lr_vae_encoder": cfg.lr_vae_encoder,
            "lr_bridge": cfg.lr_bridge,
            "w_diff": cfg.w_diff,
            "w_rec": cfg.w_rec,
            "w_clean_recon": cfg.w_clean_recon,
            "w_teacher_recon": cfg.w_teacher_recon,
            "w_teacher_latent": cfg.w_teacher_latent,
            "w_lpips_kd": cfg.w_lpips_kd,
        },
    )

    if lpips_model is None:
        lpips_model = maybe_build_lpips(use_lpips=True)

    if lpips_model is not None:
        lpips_model.eval()

    # teacher is frozen; keep it outside optimizer
    teacher.to(accelerator.device)
    teacher.eval()

    modules_to_prepare = [student, optimizer, dataloader]
    if bridge_in is not None:
        modules_to_prepare.insert(1, bridge_in)
    if bridge_out is not None:
        modules_to_prepare.insert(2 if bridge_in is not None else 1, bridge_out)

    prepared = accelerator.prepare(*modules_to_prepare)

    # unpack prepared modules
    if bridge_in is not None and bridge_out is not None:
        student, bridge_in, bridge_out, optimizer, dataloader = prepared
    elif bridge_in is not None and bridge_out is None:
        student, bridge_in, optimizer, dataloader = prepared
    elif bridge_in is None and bridge_out is not None:
        student, bridge_out, optimizer, dataloader = prepared
    else:
        student, optimizer, dataloader = prepared

    if lpips_model is not None:
        lpips_model.to(accelerator.device)

    student.train()
    if bridge_in is not None:
        bridge_in.train()
    if bridge_out is not None:
        bridge_out.train()

    global_step = 0

    # resume
    if cfg.resume_ckpt is not None:
        # load before/after prepare is always a bit delicate; here we load into unwrapped modules
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"[stage2] loading checkpoint: {cfg.resume_ckpt}")
        global_step = load_stage2_checkpoint(
            cfg.resume_ckpt,
            student=accelerator.unwrap_model(student),
            optimizer=optimizer,
            bridge_in=accelerator.unwrap_model(bridge_in) if bridge_in is not None else None,
            bridge_out=accelerator.unwrap_model(bridge_out) if bridge_out is not None else None,
            map_location="cpu",
        )
        accelerator.wait_for_everyone()

    progress_bar = tqdm(
        total=cfg.stage2_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Stage2",
    )

    while global_step < cfg.stage2_steps:
        for batch in dataloader:
            with accelerator.accumulate(student):
                pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device, non_blocking=True)

                # teacher forward
                use_teacher = (global_step % cfg.teacher_every_n_steps == 0)
                teacher_outputs = None
                if use_teacher:
                    with torch.no_grad():
                        teacher_outputs = teacher.forward_for_kd(
                            pixel_values=pixel_values,
                            sample_posterior=cfg.sample_posterior_teacher,
                        )

                # student forward
                student_outputs = student(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sample_posterior=cfg.sample_posterior_student,
                    decode_clean_latents=True,
                )

                weights = get_stage2_weights(cfg, global_step)

                # 1) diffusion
                loss_diff = diffusion_loss(student_outputs.model_pred, student_outputs.target)

                # 2) GT reconstruction anchor
                loss_rec = reconstruction_loss(student_outputs.pred_image, pixel_values)

                # 3) clean reconstruction
                if student_outputs.clean_recon is not None:
                    loss_clean_recon = reconstruction_loss(student_outputs.clean_recon, pixel_values)
                else:
                    loss_clean_recon = torch.tensor(0.0, device=accelerator.device)

                # 4) teacher image KD
                loss_teacher_recon = torch.tensor(0.0, device=accelerator.device)
                loss_lpips_kd = torch.tensor(0.0, device=accelerator.device)

                if teacher_outputs is not None:
                    loss_teacher_recon = image_kd_loss(
                        student_outputs.pred_image,
                        teacher_outputs.teacher_recon,
                    )

                    if lpips_model is not None:
                        loss_lpips_kd = lpips_loss(
                            lpips_model,
                            student_outputs.pred_image,
                            teacher_outputs.teacher_recon,
                        )

                # 5) bridged latent KD
                loss_teacher_latent = torch.tensor(0.0, device=accelerator.device)

                if teacher_outputs is not None and bridge_in is not None and bridge_out is not None:
                    teacher_latents_for_student = bridge_out(bridge_in(teacher_outputs.teacher_latents))
                    loss_teacher_latent = latent_kd_loss(
                        student_outputs.latents,
                        teacher_latents_for_student,
                    )

                # total loss
                loss = (
                    weights["w_diff"] * loss_diff
                    + weights["w_rec"] * loss_rec
                    + weights["w_clean_recon"] * loss_clean_recon
                    + weights["w_teacher_recon"] * loss_teacher_recon
                    + weights["w_teacher_latent"] * loss_teacher_latent
                    + weights["w_lpips_kd"] * loss_lpips_kd
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grads = list(student.parameters())
                    if bridge_in is not None:
                        grads += list(bridge_in.parameters())
                    if bridge_out is not None:
                        grads += list(bridge_out.parameters())

                    accelerator.clip_grad_norm_(grads, cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % cfg.log_every == 0:
                        accelerator.log(
                            {
                                "stage2/loss": loss.detach().float().item(),
                                "stage2/loss_diff": loss_diff.detach().float().item(),
                                "stage2/loss_rec": loss_rec.detach().float().item(),
                                "stage2/loss_clean_recon": loss_clean_recon.detach().float().item(),
                                "stage2/loss_teacher_recon": loss_teacher_recon.detach().float().item(),
                                "stage2/loss_teacher_latent": loss_teacher_latent.detach().float().item(),
                                "stage2/loss_lpips_kd": loss_lpips_kd.detach().float().item(),
                                "stage2/lr": optimizer.param_groups[0]["lr"],
                                "stage2/w_rec": weights["w_rec"],
                                "stage2/w_clean_recon": weights["w_clean_recon"],
                                "stage2/w_teacher_recon": weights["w_teacher_recon"],
                                "stage2/w_teacher_latent": weights["w_teacher_latent"],
                                "stage2/w_lpips_kd": weights["w_lpips_kd"],
                            },
                            step=global_step,
                        )

                    if global_step % cfg.save_every == 0:
                        save_stage2_checkpoint(
                            accelerator=accelerator,
                            student=student,
                            optimizer=optimizer,
                            global_step=global_step,
                            output_dir=cfg.output_dir,
                            bridge_in=bridge_in,
                            bridge_out=bridge_out,
                        )

                    if accelerator.is_main_process and global_step % cfg.log_every == 0:
                        progress_bar.set_postfix(
                            loss=f"{loss.detach().float().item():.4f}",
                            diff=f"{loss_diff.detach().float().item():.4f}",
                            t_rec=f"{loss_teacher_recon.detach().float().item():.4f}",
                            t_lat=f"{loss_teacher_latent.detach().float().item():.4f}",
                        )

                    if global_step >= cfg.stage2_steps:
                        break

        if global_step >= cfg.stage2_steps:
            break

    accelerator.wait_for_everyone()

    save_stage2_checkpoint(
        accelerator=accelerator,
        student=student,
        optimizer=optimizer,
        global_step=global_step,
        output_dir=cfg.output_dir,
        bridge_in=bridge_in,
        bridge_out=bridge_out,
    )

    accelerator.end_training()
    return accelerator.unwrap_model(student)

'''
cfg.lr_unet = 1e-4
cfg.lr_vae_decoder = 5e-5
cfg.lr_vae_encoder = 1e-5
cfg.lr_bridge = 1e-4

cfg.weight_decay = 1e-2
cfg.adam_beta1 = 0.9
cfg.adam_beta2 = 0.999
cfg.adam_eps = 1e-8
'''