# Mobile Depth Decdoer

1. Depthwise Upsampling Block (DWUpsample)
    * Doubles spatial resolution (H×W → 2H×2W)

    * Preserves channel count

    * Operates independently per channel 

2. Skip Projections
    * Align encoder feature channels with decoder channels

    * Enable additive skip fusion

3. Channel Reduction after Upsampling
    * Reduces channels after spatial upsampling


4. Depthwise-Separable Refinement Block \
    Depthwise Conv (3×3, groups=C) \
    → BN → ReLU \
    → Pointwise Conv (1×1) \
    → BN → ReLU

    * Refines fused features
    * Enables limited channel interaction at low cost

5. Additive Skip Fusion: x = x + skip \
    Concatenation:
    * Doubles channels
    * Increases memory and compute
    * Requires heavy fusion convs

    Additive fusion:
    * Preserves channel count
    * Encourages residual refinement
    * Is stable and memory-efficient



| Stage  | Resolution | Channels | Operations                       |
| ------ | ---------- | -------- | -------------------------------- |
| Input  | 1/32       | 160      | skip_proj_4                      |
| D4     | 1/16       | 64       | DWUpsample → 1×1 → Add → DW-Conv |
| D3     | 1/8        | 32       | DWUpsample → 1×1 → Add → DW-Conv |
| D2     | 1/4        | 24       | DWUpsample → 1×1 → Add → DW-Conv |
| D1     | 1/2        | 16       | DWUpsample → 1×1 → Add → DW-Conv |
| Output | 1/2        | 1        | 3×3 Conv                         |

**Top-down decoding loop (i = 4 → 0)** \
For each scale:
* Depthwise upsample
* Channel projection
* Skip projection
* Spatial alignment
* Additive fusion
* Depthwise-separable refinement
* depth prediction