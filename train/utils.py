# train/utils.py
def make_optimizer(cfg, student, stage: int, bridge_in=None, bridge_out=None):
    params = []

    # U-Net always trainable in all three stages
    params.append({
        "params": [p for p in student.unet.parameters() if p.requires_grad],
        "lr": cfg.lr_unet,
        "weight_decay": cfg.wd_unet,
    })

    # VAE policy by stage
    decoder_params = list(student.vae.decoder.parameters()) if hasattr(student.vae, "decoder") else []
    encoder_params = list(student.vae.encoder.parameters()) if hasattr(student.vae, "encoder") else []

    if stage == 1:
        # keep encoder slow, decoder moderate
        if decoder_params:
            params.append({
                "params": [p for p in decoder_params if p.requires_grad],
                "lr": cfg.lr_vae_decoder,
                "weight_decay": cfg.wd_vae,
            })
        if encoder_params:
            params.append({
                "params": [p for p in encoder_params if p.requires_grad],
                "lr": cfg.lr_vae_encoder,
                "weight_decay": cfg.wd_vae,
            })

    elif stage == 2:
        if decoder_params:
            params.append({
                "params": [p for p in decoder_params if p.requires_grad],
                "lr": cfg.lr_vae_decoder,
                "weight_decay": cfg.wd_vae,
            })
        if encoder_params:
            params.append({
                "params": [p for p in encoder_params if p.requires_grad],
                "lr": cfg.lr_vae_encoder,
                "weight_decay": cfg.wd_vae,
            })

    elif stage == 3:
        # either full refinement or lower LR globally
        if decoder_params:
            params.append({
                "params": [p for p in decoder_params if p.requires_grad],
                "lr": cfg.lr_vae_decoder * 0.5,
                "weight_decay": cfg.wd_vae,
            })
        if encoder_params:
            params.append({
                "params": [p for p in encoder_params if p.requires_grad],
                "lr": cfg.lr_vae_encoder * 0.5,
                "weight_decay": cfg.wd_vae,
            })

    if bridge_in is not None:
        params.append({"params": bridge_in.parameters(), "lr": cfg.lr_unet, "weight_decay": 0.0})
    if bridge_out is not None:
        params.append({"params": bridge_out.parameters(), "lr": cfg.lr_unet, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        params,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    return optimizer