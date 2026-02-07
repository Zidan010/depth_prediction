import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import ActivationMemoryTracker


# --------------------------------------------------
# Depthwise upsampling
# --------------------------------------------------

class DWUpsample(nn.Module):
    """Depthwise transposed convolution (spatial upsample, channel preserving)"""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.up(x)


# --------------------------------------------------
# Optional blocks
# --------------------------------------------------

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
        y = self.avg_pool(x)                # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2) # [B,1,C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


class ScaleModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale


class EdgeAwareRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat, image):
        # grayscale edge cue
        gray = image.mean(1, keepdim=True)
        edge = torch.abs(gray - F.avg_pool2d(gray, 3, 1, 1))
        edge = F.interpolate(
            edge, size=feat.shape[-2:],
            mode="bilinear", align_corners=False
        )
        gate = self.sigmoid(self.pointwise(self.depthwise(feat)))
        return feat + gate * edge


# --------------------------------------------------
# Mobile Depth Decoder (Extended)
# --------------------------------------------------

class MobileDepthDecoderV2(nn.Module):
    """
    Mobile-Aware Depth Decoder (MAD++)
    - Additive skip fusion
    - Depthwise upsampling
    - Optional ECA / Scale / Edge awareness
    - Activation memory tracking
    """

    def __init__(
        self,
        num_ch_enc,
        num_scales=1,
        use_eca=False,
        use_scale_modulation=False,
        use_edge_aware=False,
    ):
        super().__init__()

        assert num_scales in [1, 2, 3, 4]

        self.num_ch_enc = num_ch_enc
        self.num_scales = num_scales
        self.use_eca = use_eca
        self.use_scale_modulation = use_scale_modulation
        self.use_edge_aware = use_edge_aware

        # Mobile-safe decoder widths
        self.num_ch_dec = [16, 24, 32, 64, 160]

        self.convs = OrderedDict()
        self.mem_tracker = None

        # --------------------------------------------------
        # Skip projections
        # --------------------------------------------------
        for i in range(5):
            self.convs[f"skip_proj_{i}"] = nn.Conv2d(
                num_ch_enc[i],
                self.num_ch_dec[i],
                kernel_size=1,
                bias=False
            )

        # --------------------------------------------------
        # Decoder stages
        # --------------------------------------------------
        for i in range(4, -1, -1):

            in_ch = self.num_ch_dec[i + 1] if i < 4 else self.num_ch_dec[4]

            self.convs[f"up_{i}"] = DWUpsample(in_ch)
            self.convs[f"up_proj_{i}"] = nn.Conv2d(
                in_ch, self.num_ch_dec[i], kernel_size=1, bias=False
            )

            self.convs[f"conv_{i}"] = nn.Sequential(
                nn.Conv2d(
                    self.num_ch_dec[i], self.num_ch_dec[i],
                    kernel_size=3, padding=1,
                    groups=self.num_ch_dec[i], bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.num_ch_dec[i], self.num_ch_dec[i],
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),
            )

            if use_eca:
                self.convs[f"eca_{i}"] = ECABlock(self.num_ch_dec[i])

            if use_scale_modulation:
                self.convs[f"scale_{i}"] = ScaleModulation(self.num_ch_dec[i])

            if use_edge_aware:
                self.convs[f"edge_{i}"] = EdgeAwareRefinement(self.num_ch_dec[i])

        # --------------------------------------------------
        # Depth heads
        # --------------------------------------------------
        self.disp_heads = nn.ModuleDict()
        for s in range(num_scales):
            self.disp_heads[str(s)] = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[s], 1, 3, padding=1),
                nn.Sigmoid()
            )

        self.net = nn.ModuleList(self.convs.values())

    def forward(self, input_features, image=None):
        if self.mem_tracker:
            self.mem_tracker.records.clear()

        self.outputs = {}

        x = self.convs["skip_proj_4"](input_features[4])
        if self.mem_tracker:
            self.mem_tracker.record("encoder_out", x)

        for i in range(4, -1, -1):

            x = self.convs[f"up_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"up_{i}", x)

            x = self.convs[f"up_proj_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"up_proj_{i}", x)

            skip = self.convs[f"skip_proj_{i}"](input_features[i])
            if self.mem_tracker:
                self.mem_tracker.record(f"skip_proj_{i}", skip)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            x = x + skip
            if self.mem_tracker:
                self.mem_tracker.record(f"add_{i}", x)

            x = self.convs[f"conv_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"conv_{i}", x)

            if self.use_eca:
                x = self.convs[f"eca_{i}"](x)
                if self.mem_tracker:
                    self.mem_tracker.record(f"eca_{i}", x)

            if self.use_scale_modulation:
                x = self.convs[f"scale_{i}"](x)
                if self.mem_tracker:
                    self.mem_tracker.record(f"scale_{i}", x)

            if self.use_edge_aware:
                if image is None:
                    raise ValueError("image required when use_edge_aware=True")
                x = self.convs[f"edge_{i}"](x, image)
                if self.mem_tracker:
                    self.mem_tracker.record(f"edge_{i}", x)

            if self.mem_tracker:
                self.mem_tracker.record(f"dec_{i}", x)

            if i < self.num_scales:
                disp = self.disp_heads[str(i)](x)
                self.outputs[("disp", i)] = disp
                if self.mem_tracker:
                    self.mem_tracker.record(f"disp_{i}", disp)

        if self.mem_tracker:
            self.mem_tracker.export_csv()

        return self.outputs
