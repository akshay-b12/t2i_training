# losses/kd.py
import torch
import torch.nn.functional as F

def reconstruction_loss(pred_image, gt_image):
    return F.l1_loss(pred_image.float(), gt_image.float(), reduction="mean")

def image_kd_loss(student_image, teacher_image):
    return F.l1_loss(student_image.float(), teacher_image.float(), reduction="mean")

def latent_kd_loss(student_pred, teacher_pred_16ch):
    return F.mse_loss(student_pred.float(), teacher_pred_16ch.float(), reduction="mean")
def normalize_feature(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Normalize per-sample globally across C,H,W to make KD less scale-sensitive
    denom = torch.sqrt(torch.mean(x.float() ** 2, dim=(1, 2, 3), keepdim=True) + eps)
    return x.float() / denom


def feature_kd_loss(
    student_store: dict,
    teacher_store: dict,
    adapters: FeatureAdapterBank,
    feature_keys: list[str],
    feature_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    loss = 0.0
    total_w = 0.0

    for key in feature_keys:
        if key not in student_store or key not in teacher_store:
            continue

        s_feat = student_store[key]
        t_feat = teacher_store[key].detach()

        s_feat = adapters.forward_one(key, s_feat)

        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.interpolate(
                s_feat,
                size=t_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        s_feat = normalize_feature(s_feat)
        t_feat = normalize_feature(t_feat)

        w = 1.0 if feature_weights is None else feature_weights.get(key, 1.0)
        loss = loss + w * F.mse_loss(s_feat, t_feat)
        total_w += w

    if total_w == 0:
        # safe zero on current device
        any_tensor = next(iter(student_store.values()))
        return torch.tensor(0.0, device=any_tensor.device)

    return loss / total_w