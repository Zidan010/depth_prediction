import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import ActivationMemoryTracker


class DWUpsample(nn.Module):
    """
    Depthwise transposed convolution for spatial upsampling.
    NOTE: This preserves channel count.
    """
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=channels,   # depthwise
            bias=False
        )

    def forward(self, x):
        return self.up(x)


class MobileDepthDecoder(nn.Module):
    """
    Mobile-Aware Depth Decoder (MAD)

    Key properties:
    - Additive skip fusion (no concat)
    - Depthwise upsampling
    - Explicit channel transitions (1x1 convs)
    - Single-scale depth output (scale 0)
    """

    def __init__(self, num_ch_enc, num_scales=1):
        super().__init__()

        assert num_scales in [1, 2, 3, 4], "num_scales must be 1-4"

        self.num_ch_enc = num_ch_enc
        self.num_scales = num_scales

        # Decoder channel widths (low-memory, mobile-safe)
        # Must align with skip projections
        self.num_ch_dec = [16, 24, 32, 64, 160]

        self.convs = OrderedDict()
        # self.mem_tracker = ActivationMemoryTracker()
        self.mem_tracker = None
        
        # --------------------------------------------------
        # 1. Project encoder features → decoder channels
        # --------------------------------------------------
        for i in range(5):
            self.convs[f"skip_proj_{i}"] = nn.Conv2d(
                self.num_ch_enc[i],
                self.num_ch_dec[i],
                kernel_size=1,
                bias=False
            )

        # --------------------------------------------------
        # 2. Decoder blocks (top-down)
        # --------------------------------------------------
        for i in range(4, -1, -1):

            # a) Upsample (preserves channels)
            in_ch = self.num_ch_dec[i + 1] if i < 4 else self.num_ch_dec[4]
            self.convs[f"up_{i}"] = DWUpsample(in_ch)

            # b) Channel reduction AFTER upsampling
            self.convs[f"up_proj_{i}"] = nn.Conv2d(
                in_ch,
                self.num_ch_dec[i],
                kernel_size=1,
                bias=False
            )

            # c) Depthwise-separable refinement
            self.convs[f"conv_{i}"] = nn.Sequential(
                # depthwise
                nn.Conv2d(
                    self.num_ch_dec[i],
                    self.num_ch_dec[i],
                    kernel_size=3,
                    padding=1,
                    groups=self.num_ch_dec[i],
                    bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),

                # pointwise
                nn.Conv2d(
                    self.num_ch_dec[i],
                    self.num_ch_dec[i],
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(self.num_ch_dec[i]),
                nn.ReLU(inplace=True),
            ) 

        # --------------------------------------------------
        # 3. Final depth head (single-scale)
        # --------------------------------------------------
        self.disp_heads = nn.ModuleDict()
        for s in range(self.num_scales):
            self.disp_heads[str(s)] = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[s], 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        # for name, module in self.convs.items():
        #     module.register_forward_hook(
        #         lambda m, i, o, n=name: (
        #             self.mem_tracker.hook(n)(m, i, o)
        #             if self.mem_tracker is not None else None
        #         )
        #     )

        # self.disp_head.register_forward_hook(
        #     lambda m, i, o: (
        #         self.mem_tracker.hook("disp_head")(m, i, o)
        #         if self.mem_tracker is not None else None
        #     )
        # )


        # Register all layers
        self.net = nn.ModuleList(self.convs.values())

    def forward(self, input_features):
        """
        input_features:
            list of 5 tensors from encoder
            resolutions: [1/2, 1/4, 1/8, 1/16, 1/32]
        """
        if self.mem_tracker is not None:
            self.mem_tracker.records.clear()
        # Start from deepest feature

        self.outputs = {}

        x = self.convs["skip_proj_4"](input_features[4])
        if self.mem_tracker:
            self.mem_tracker.record("encoder_out", x)

        for i in range(4, -1, -1):

            # Upsample spatially
            x = self.convs[f"up_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"up_{i}", x)

            # Reduce channels
            x = self.convs[f"up_proj_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"up_proj_{i}", x)

            # Project skip
            skip = self.convs[f"skip_proj_{i}"](input_features[i])
            if self.mem_tracker:
                self.mem_tracker.record(f"skip_proj_{i}", skip)

            # Ensure spatial alignment (KITTI odd sizes)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Additive fusion (memory-efficient)
            x = x + skip
            if self.mem_tracker:
                self.mem_tracker.record(f"add_{i}", x)
            
            # Refinement
            x = self.convs[f"conv_{i}"](x)
            if self.mem_tracker:
                self.mem_tracker.record(f"conv_{i}", x)

            if i < self.num_scales:
                disp = self.disp_heads[str(i)](x)
                self.outputs[('disp', i)] = disp
                if self.mem_tracker:
                    self.mem_tracker.record(f"disp_{i}", disp)
            
        # disp = self.disp_head(x)

        return {("disp", 0): disp}

