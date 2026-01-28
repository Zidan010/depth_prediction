import torch
import torch.nn as nn
from collections import OrderedDict

class DWUpsample(nn.Module):
    """
    Depthwise transposed convolution for upsampling.
    Depthwise is chosen to reduce compute and memory.
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=channels,  # Depthwise
            bias=False
        )

    def forward(self, x):
        return self.up(x)


class MobileDepthDecoder(nn.Module):
    """
    Mobile-Aware Depth Decoder (MAD)
    - additive skip fusion (memory-efficient)
    - depthwise upsampling
    - single-scale output (scale 0)
    """

    def __init__(self, num_ch_enc):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_scales = 1  # We only need the main output

        # Conservatively chosen decoder widths (small memory footprint)
        self.num_ch_dec = [16, 24, 32, 64, 160]

        self.convs = OrderedDict()

        # Project encoder skips to decoder dims with 1x1 conv
        for i in range(5):
            self.convs[f"skip_proj_{i}"] = nn.Conv2d(
                num_ch_enc[i],
                self.num_ch_dec[i],
                kernel_size=1,
                bias=False
            )

        # Build decoder blocks with depthwise operations
        for i in range(4, -1, -1):
            self.convs[f"up_{i}"] = DWUpsample(self.num_ch_dec[i])

            self.convs[f"conv_{i}"] = nn.Sequential(
                nn.Conv2d(
                    self.num_ch_dec[i],
                    self.num_ch_dec[i],
                    kernel_size=3,
                    padding=1,
                    groups=self.num_ch_dec[i],  # Depthwise
                    bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    self.num_ch_dec[i],
                    self.num_ch_dec[i],
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),
            )

        # Depth head for final output
        self.disp_head = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.net = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        x = self.convs["skip_proj_4"](input_features[4])
        for i in range(4, -1, -1):
            x = self.convs[f"up_{i}"](x)

            skip = self.convs[f"skip_proj_{i}"](input_features[i])

            # Additive fusion (no channel explosion)
            x = x + skip

            x = self.convs[f"conv_{i}"](x)

        disp = self.disp_head(x)
        return {("disp", 0): disp}
