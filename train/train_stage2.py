def train_stage2(
    cfg,
    student,
    teacher,
    dataloader,
    optimizer,
    lpips_model=None,
    bridge_in=None,
    bridge_out=None,
    accelerator=None,
):
    student.train()
    teacher.eval()

    global_step = 0
    while global_step < cfg.stage2_steps:
        for batch in dataloader:
            with accelerator.accumulate(student) if accelerator else nullcontext():
                pixel_values = batch["pixel_values"].to(student.unet.device)
                input_ids = batch["input_ids"].to(student.unet.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(student.unet.device)

                student_outputs, noise = student(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = 0.0

                # 1) diffusion task loss
                loss_diff = diffusion_loss(student_outputs.model_pred, noise)
                loss = loss + cfg.w_diff * loss_diff

                # 2) teacher image-space KD
                if global_step % cfg.teacher_every_n_steps == 0:
                    teacher_outputs = teacher.forward_for_kd(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss_img_kd = image_kd_loss(student_outputs.pred_image, teacher_outputs.pred_image)
                    loss = loss + cfg.w_img_kd * loss_img_kd

                    if lpips_model is not None:
                        loss_lpips_kd = lpips_loss(lpips_model, student_outputs.pred_image, teacher_outputs.pred_image)
                        loss = loss + cfg.w_lpips_kd * loss_lpips_kd

                # 3) optional bridge latent KD
                if bridge_in is not None and bridge_out is not None:
                    with torch.no_grad():
                        teacher_pred_16 = None
                        # Example placeholder:
                        # teacher_pred_native = teacher.forward_latent(...)
                        # teacher_pred_16 = bridge_out(teacher_pred_native)
                    if teacher_pred_16 is not None:
                        loss_lat_kd = latent_kd_loss(student_outputs.model_pred, teacher_pred_16)
                        loss = loss + cfg.w_lat_kd * loss_lat_kd

                # 4) light reconstruction anchor
                loss_rec = reconstruction_loss(student_outputs.pred_image, pixel_values)
                loss = loss + 0.25 * cfg.w_rec * loss_rec

                if accelerator:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grads = list(student.parameters())
                        if bridge_in is not None:
                            grads += list(bridge_in.parameters())
                        if bridge_out is not None:
                            grads += list(bridge_out.parameters())
                        accelerator.clip_grad_norm_(grads, cfg.max_grad_norm)
                else:
                    loss.backward()
                    grads = list(student.parameters())
                    if bridge_in is not None:
                        grads += list(bridge_in.parameters())
                    if bridge_out is not None:
                        grads += list(bridge_out.parameters())
                    torch.nn.utils.clip_grad_norm_(grads, cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step >= cfg.stage2_steps:
                break