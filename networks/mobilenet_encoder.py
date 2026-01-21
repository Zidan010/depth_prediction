import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import numpy as np


class MobileNetEncoder(nn.Module):
    def __init__(self, weights=MobileNet_V2_Weights, num_input_images=1):
        super().__init__()

        self.encoder = models.mobilenet_v2(weights=weights).features

        if num_input_images > 1:
            first_conv = self.encoder[0][0]
            new_conv = nn.Conv2d(
                num_input_images * 3,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )

            new_conv.weight.data = (
                first_conv.weight.data.repeat(1, num_input_images, 1, 1)
                / num_input_images
            )

            self.encoder[0][0] = new_conv

        self.num_ch_enc = np.array([16, 24, 32, 64, 160])

        self.feature_idxs = [0, 2, 4, 7, 14]

    def forward(self, x):
        features = []

        x = (x - 0.45) / 0.225

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.feature_idxs:
                features.append(x)

        return features

mobile_enc = MobileNetEncoder()
x = torch.randn(1, 3, 192, 640)
features = mobile_enc(x)

for idx, f in enumerate(features):
    print(f"{idx}: shape {features[idx].shape}")

