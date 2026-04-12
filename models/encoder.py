import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class SharedEncoder(nn.Module):
    """
    MobileNetV2 encoder with a 1280 → 128 bottleneck at the deepest stage.

    WHY: The original architecture passed raw 1280-ch features into two
    DoGAttentionGates per branch: Conv2d(1280+6, 1280, 1). With two branches
    that is 2 × 1.65M = 3.3M params just in gating. Bottlenecking to 128ch
    cuts this to 2 × 17k = 34k — a ~97% reduction in gate cost with
    negligible accuracy loss (MBV2 already compresses to 160-1280 via
    its own bottleneck structure).

    Output channels per level : [16, 24, 32, 96, 128]
    Output strides             : [½,  ¼,  ⅛,  1/16, 1/32]
    """
    SKIP_CHS = [16, 24, 32, 96, 128]

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mv2 = models.mobilenet_v2(weights=weights).features
        self.s0 = mv2[:2]    # stride 2,   16 ch
        self.s1 = mv2[2:4]   # stride 4,   24 ch
        self.s2 = mv2[4:7]   # stride 8,   32 ch
        self.s3 = mv2[7:14]  # stride 16,  96 ch
        self.s4 = mv2[14:]   # stride 32, 1280 ch

        # 1×1 bottleneck: 1280 → 128
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

        self.adapt = nn.Conv2d(in_channels, 3, 1, bias=False) \
            if in_channels != 3 else None

    def forward(self, x: torch.Tensor):
        if self.adapt is not None:
            x = self.adapt(x)
        f0 = self.s0(x)                      # (B,  16, H/2,  W/2 )
        f1 = self.s1(f0)                     # (B,  24, H/4,  W/4 )
        f2 = self.s2(f1)                     # (B,  32, H/8,  W/8 )
        f3 = self.s3(f2)                     # (B,  96, H/16, W/16)
        f4 = self.bottleneck(self.s4(f3))    # (B, 128, H/32, W/32)
        return f0, f1, f2, f3, f4