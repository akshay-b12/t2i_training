def train_stage1(cfg, student, dataloader, optimizer, lpips_model=None, accelerator=None):
    student.train()

    global_step = 0
    while global_step < cfg.stage1_steps:
        for batch in dataloader:
            with accelerator.accumulate(student) if accelerator else nullcontext():
                pixel_values = batch["pixel_values"].to(student.unet.device)
                input_ids = batch["input_ids"].to(student.unet.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(student.unet.device)

                outputs, noise = student(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                target = noise  # epsilon prediction
                loss_diff = diffusion_loss(outputs.model_pred, target)
                loss_rec = reconstruction_loss(outputs.pred_image, pixel_values)

                loss = cfg.w_diff * loss_diff + cfg.w_rec * loss_rec

                if lpips_model is not None:
                    loss_lpips = lpips_loss(lpips_model, outputs.pred_image, pixel_values)
                    loss = loss + cfg.w_lpips * loss_lpips

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
            if global_step >= cfg.stage1_steps:
                break