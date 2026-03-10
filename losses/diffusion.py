# losses/diffusion.py
import torch
import torch.nn.functional as F

def diffusion_loss(model_pred, target, loss_type: str = "mse"):
    if loss_type == "mse":
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    if loss_type == "l1":
        return F.l1_loss(model_pred.float(), target.float(), reduction="mean")
    raise ValueError(loss_type)