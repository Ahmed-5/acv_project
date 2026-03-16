import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class SharedEncoder(nn.Module):
    """
    Lightweight MobileNetV2 encoder shared across both decoder branches.
    Returns 5-level feature pyramid for skip connections.

    Output channels per level: 16, 24, 32, 96, 1280
    Output strides:            1/2, 1/4, 1/8, 1/16, 1/32
    """
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mv2 = models.mobilenet_v2(weights=weights).features
        self.s0 = mv2[:2]    # 1/2,   16ch
        self.s1 = mv2[2:4]   # 1/4,   24ch
        self.s2 = mv2[4:7]   # 1/8,   32ch
        self.s3 = mv2[7:14]  # 1/16,  96ch
        self.s4 = mv2[14:]   # 1/32, 1280ch

        self.adapt = nn.Conv2d(in_channels, 3, 1, bias=False) \
                     if in_channels != 3 else None

    def forward(self, x):
        if self.adapt is not None:
            x = self.adapt(x)
        f0 = self.s0(x)
        f1 = self.s1(f0)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f0, f1, f2, f3, f4   # 5-level pyramid
