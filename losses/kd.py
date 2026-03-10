# losses/kd.py
import torch
import torch.nn.functional as F

def reconstruction_loss(pred_image, gt_image):
    return F.l1_loss(pred_image.float(), gt_image.float(), reduction="mean")

def image_kd_loss(student_image, teacher_image):
    return F.l1_loss(student_image.float(), teacher_image.float(), reduction="mean")

def latent_kd_loss(student_pred, teacher_pred_16ch):
    return F.mse_loss(student_pred.float(), teacher_pred_16ch.float(), reduction="mean")