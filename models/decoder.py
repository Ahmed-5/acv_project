import torch
import torch.nn as nn
import torch.nn.functional as F


class DoGAttentionGate(nn.Module):
    """
    Modulates encoder skip features using the DoG signal at the same resolution.
    The gate learns WHICH features matter for depth at each scale.
    Novel: uses DoG (not learned attention query) as the modulation signal.
    """
    def __init__(self, feat_ch, dog_ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + dog_ch, feat_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat, dog):
        dog_r = F.interpolate(dog, size=feat.shape[-2:],
                              mode='bilinear', align_corners=False)
        return feat * self.gate(torch.cat([feat, dog_r], dim=1))


class DecoderBranch(nn.Module):
    """
    Single decoder branch with DoG-gated skip connections.
    Coarse branch uses coarse-scale DoG; fine branch uses fine-scale DoG.
    """
    SKIP_CHS = [1280, 96, 32, 24, 16]   # MobileNetV2 channels per level

    def __init__(self, dog_fine_ch, dog_coarse_ch, out_ch=64):
        super().__init__()
        # Assign DoG frequency per decoder stage
        dog_chs = [dog_coarse_ch]*2 + [dog_fine_ch]*3
        self.gates = nn.ModuleList([
            DoGAttentionGate(self.SKIP_CHS[i], dog_chs[i])
            for i in range(5)
        ])
        in_chs = [1280, 96+out_ch, 32+out_ch, 24+out_ch, 16+out_ch]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for c in in_chs
        ])

    def forward(self, feats, dog_fine, dog_coarse):
        f0, f1, f2, f3, f4 = feats
        dogs = [dog_coarse]*2 + [dog_fine]*3
        enc  = [f4, f3, f2, f1, f0]

        x = self.gates[0](enc[0], dogs[0])
        x = self.convs[0](x)
        for i in range(1, 5):
            x = F.interpolate(x, size=enc[i].shape[-2:],
                              mode='bilinear', align_corners=False)
            skip = self.gates[i](enc[i], dogs[i])
            x = self.convs[i](torch.cat([x, skip], dim=1))
        return x   # (B, out_ch, H/2, W/2)
