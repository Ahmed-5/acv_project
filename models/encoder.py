import torch
import torch.nn as nn

import timm


class SharedEncoder(nn.Module):
    """
    EfficientNet-B0 encoder with FixEfficientNet-style pretrained weights.

    timm `tf_efficientnet_b0.ns_jft_in1k` (Noisy-Student) provides the strongest
    publicly-available B0 ImageNet weights and is the closest open analogue to
    the FixEfficientNet recipe (train low-res, fine-tune at higher res).

    Output channels per level: [16, 24, 40, 112, 128] (last after bottleneck)
    Output strides:            [1/2, 1/4, 1/8, 1/16, 1/32]
    """
    SKIP_CHS = [16, 24, 40, 112, 128]

    BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            self.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_channels,
        )
        last = self.backbone.feature_info.channels()[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(last, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f0, f1, f2, f3, f4 = self.backbone(x)
        f4 = self.bottleneck(f4)
        return f0, f1, f2, f3, f4
