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