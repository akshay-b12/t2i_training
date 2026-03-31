class FeatureAdapterBank(nn.Module):
    """
    One adapter per hooked feature key.
    Adapts student feature channels to teacher feature channels.
    """

    def __init__(self, channel_map: dict):
        """
        channel_map: {
            "d0": (student_channels, teacher_channels),
            ...
        }
        """
        super().__init__()
        self.adapters = nn.ModuleDict()

        for key, (c_s, c_t) in channel_map.items():
            if c_s == c_t:
                self.adapters[key] = nn.Identity()
            else:
                self.adapters[key] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True)

    def forward_one(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return self.adapters[key](x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ActivationStore(dict):
    def clear_all(self):
        self.clear()


def _make_hook(store: dict, name: str, take_first: bool):
    def hook(module, inputs, output):
        # In diffusers UNet, down blocks often return tuples like (hidden_states, res_samples)
        if take_first and isinstance(output, (tuple, list)):
            store[name] = output[0]
        else:
            store[name] = output
    return hook


def register_unet_feature_hooks(unet: nn.Module, store: dict):
    """
    Registers hooks on all down_blocks and up_blocks.
    No mid_block hooks because you said both teacher/student do not have a middle block.
    Returns a list of removable handles.
    """
    handles = []

    for i in range(len(unet.down_blocks)):
        handles.append(
            unet.down_blocks[i].register_forward_hook(
                _make_hook(store, f"d{i}", take_first=True)
            )
        )

    for i in range(len(unet.up_blocks)):
        handles.append(
            unet.up_blocks[i].register_forward_hook(
                _make_hook(store, f"u{i}", take_first=False)
            )
        )

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()

@torch.no_grad()
def build_feature_adapters_from_dummy(
    student,
    teacher_unet,
    image_size: int,
    batch_size: int = 1,
    seq_len: int = 77,
    device: str = "cuda",
):
    """
    Runs one synthetic forward on both teacher and student U-Nets to infer block feature channels.
    Assumes:
      - student is StudentModelFlow
      - teacher_unet is a UNet2DConditionModel-like teacher
      - no middle block distillation
    """
    device = torch.device(device)

    student_store = ActivationStore()
    teacher_store = ActivationStore()

    student_handles = register_unet_feature_hooks(student.unet, student_store)
    teacher_handles = register_unet_feature_hooks(teacher_unet, teacher_store)

    latent_h = image_size // student.vae_scale_factor
    latent_w = image_size // student.vae_scale_factor

    # Dummy inputs
    x = torch.randn(
        batch_size,
        student.unet.config.in_channels,
        latent_h,
        latent_w,
        device=device,
        dtype=next(student.unet.parameters()).dtype,
    )
    t = student.scheduler.timesteps[:batch_size].to(device=device)
    text_dim = student.unet.config.cross_attention_dim
    enc = torch.randn(
        batch_size,
        seq_len,
        text_dim,
        device=device,
        dtype=next(student.unet.parameters()).dtype,
    )

    _ = student.unet(x, t, encoder_hidden_states=enc).sample
    _ = teacher_unet(
        x.to(dtype=next(teacher_unet.parameters()).dtype),
        t.to(device=device),
        encoder_hidden_states=enc.to(dtype=next(teacher_unet.parameters()).dtype),
    ).sample

    channel_map = {}
    feature_keys = sorted(set(student_store.keys()).intersection(set(teacher_store.keys())))

    for key in feature_keys:
        s = student_store[key]
        t_ = teacher_store[key]
        channel_map[key] = (s.shape[1], t_.shape[1])

    remove_hooks(student_handles)
    remove_hooks(teacher_handles)

    adapters = FeatureAdapterBank(channel_map).to(device)
    return adapters, feature_keys, channel_map