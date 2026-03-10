def train_stage3(
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
    while global_step < cfg.stage3_steps:
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

                teacher_outputs = teacher.forward_for_kd(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss_diff = diffusion_loss(student_outputs.model_pred, noise)
                loss_img_kd = image_kd_loss(student_outputs.pred_image, teacher_outputs.pred_image)
                loss_rec = reconstruction_loss(student_outputs.pred_image, pixel_values)

                loss = (
                    0.75 * cfg.w_diff * loss_diff
                    + 0.75 * cfg.w_img_kd * loss_img_kd
                    + 0.5 * cfg.w_rec * loss_rec
                )

                if lpips_model is not None:
                    loss = loss + 0.75 * cfg.w_lpips_kd * lpips_loss(
                        lpips_model, student_outputs.pred_image, teacher_outputs.pred_image
                    )

                if accelerator:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step >= cfg.stage3_steps:
                break