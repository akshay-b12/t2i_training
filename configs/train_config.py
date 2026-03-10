from dataclasses import dataclass

@dataclass
class TrainConfig:
    dataset_name: str = "laion/relaion-pop"   # replace if your actual HF id differs
    dataset_split: str = "train"
    image_column: str = "image"
    caption_column: str = "text"
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = True

    train_batch_size: int = 8
    num_workers: int = 4
    grad_accum_steps: int = 2
    max_train_steps: int = 300_000
    mixed_precision: str = "bf16"  # or "fp16"
    gradient_checkpointing: bool = True

    prediction_type: str = "epsilon"  # or "v_prediction"
    num_train_timesteps: int = 1000

    lr_unet: float = 1e-4
    lr_vae_decoder: float = 5e-5
    lr_vae_encoder: float = 1e-5
    lr_text_proj: float = 0.0

    wd_unet: float = 1e-2
    wd_vae: float = 1e-2

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    latent_scaling_factor: float = 1.0  # replace with your calibrated 16ch value

    # stage steps
    stage1_steps: int = 50_000
    stage2_steps: int = 200_000
    stage3_steps: int = 50_000

    # loss weights
    w_diff: float = 1.0
    w_rec: float = 0.5
    w_lpips: float = 0.1
    w_img_kd: float = 0.5
    w_lpips_kd: float = 0.1
    w_lat_kd: float = 0.05

    # optional teacher frequency
    teacher_every_n_steps: int = 1