import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)               # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


class InvertedResidualDilated(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1, use_eca=False):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, 3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.eca = ECABlock(oup) if use_eca else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.eca(out)
        return x + out if self.use_res_connect else out


class MobileMonoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_ch_enc = [16, 24, 32, 96, 320]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = InvertedResidualDilated(32, 16, 1, 1)
        self.stage2 = InvertedResidualDilated(16, 24, 2, 6)
        self.stage3 = InvertedResidualDilated(24, 32, 2, 6)
        self.stage4 = InvertedResidualDilated(32, 64, 2, 6)

        # 🔥 Dilated + Attention stage
        self.stage5 = InvertedResidualDilated(
            64, 160,
            stride=1,
            expand_ratio=6,
            dilation=2,
            use_eca=True
        )

    def forward(self, x):
        features = []

        x = self.stem(x)
        x = self.stage1(x); features.append(x)
        x = self.stage2(x); features.append(x)
        x = self.stage3(x); features.append(x)
        x = self.stage4(x); features.append(x)
        x = self.stage5(x); features.append(x)

        return features
