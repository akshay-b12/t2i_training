import torch
import torch.nn as nn


class ResidualLatentBridge(nn.Module):
    """
    Lightweight latent-domain alignment bridge for 16ch teacher/student spaces.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]

    Default:
        channels = 16
        hidden_channels = 32
    """
    def __init__(
        self,
        channels: int = 16,
        hidden_channels: int = 32,
        use_residual: bool = True,
        use_norm: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_norm = use_norm

        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(),
        ]

        if use_norm:
            layers.append(nn.GroupNorm(num_groups=1, num_channels=hidden_channels))

        layers.extend([
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        ])

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # near-identity start: last layer initialized small
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last = None
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.use_residual:
            return x + y
        return y
    
'''
bridge_in = ResidualLatentBridge(channels=16, hidden_channels=32)
bridge_out = ResidualLatentBridge(channels=16, hidden_channels=32)
'''