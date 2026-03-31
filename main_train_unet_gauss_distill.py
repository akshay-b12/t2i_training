# Build adapters first, before optimizer
feature_adapters, feature_keys, channel_map = build_feature_adapters_from_dummy(
    student=student,
    teacher_unet=teacher_unet,
    image_size=512,
    batch_size=1,
    seq_len=77,
    device="cuda",
)

cfg = T2IDistillConfig(
    train_steps=100_000,
    grad_accum_steps=2,
    mixed_precision="bf16",
    output_dir="./outputs_t2i_unet_distill",
    project_name="nano16_teacher700m_distill",
    log_every=50,
    save_every=5000,
    lr_unet=1e-4,
    lr_adapter=1e-4,
    w_teacher_pred=1.0,
    w_feature=1.0,
    w_task=0.5,
    w_x0_img=0.25,
    w_lpips=0.05,
    w_gt_img=0.05,
    reuse_student_text_condition=True,
)

optimizer = make_t2i_unet_distill_optimizer(
    student=student,
    feature_adapters=feature_adapters,
    cfg=cfg,
)

feature_keys = [f"d{i}" for i in range(len(student.unet.down_blocks))] + \
               [f"u{i}" for i in range(len(student.unet.up_blocks))]

trained_student = train_t2i_unet_distill(
    cfg=cfg,
    student=student,
    teacher_unet=teacher_unet,
    dataloader=train_loader,
    feature_adapters=feature_adapters,
    feature_keys=feature_keys,
    optimizer=optimizer,
    teacher_text_encoder=None,   # set if separate
    lpips_model=maybe_build_lpips(use_lpips=(cfg.w_lpips > 0)),
)