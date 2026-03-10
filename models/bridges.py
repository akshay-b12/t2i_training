# models/bridges.py
import torch
import torch.nn as nn

class LatentBridge(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)