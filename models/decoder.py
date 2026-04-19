import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class DoGAttentionGate(nn.Module):
    def __init__(self, feat_ch: int, dog_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + dog_ch, feat_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor, dog: torch.Tensor) -> torch.Tensor:
        dog_r = F.interpolate(dog, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        return feat * self.gate(torch.cat([feat, dog_r], dim=1))


class DecoderBranch(nn.Module):
    SKIP_CHS = [128, 96, 32, 24, 16]

    def __init__(self, dog_fine_ch: int, dog_coarse_ch: int, out_ch: int = 64):
        super().__init__()
        dog_chs = [dog_coarse_ch] * 2 + [dog_fine_ch] * 3
        self.gates = nn.ModuleList([
            DoGAttentionGate(self.SKIP_CHS[i], dog_chs[i])
            for i in range(5)
        ])
        in_chs = [128, 96 + out_ch, 32 + out_ch, 24 + out_ch, 16 + out_ch]
        self.convs = nn.ModuleList([
            nn.Sequential(
                DSConvBlock(c, out_ch),
                DSConvBlock(out_ch, out_ch)
            ) for c in in_chs
        ])

    def forward(self, feats, dog_fine, dog_coarse):
        f0, f1, f2, f3, f4 = feats
        dogs = [dog_coarse] * 2 + [dog_fine] * 3
        enc = [f4, f3, f2, f1, f0]
        x = self.gates[0](enc[0], dogs[0])
        x = self.convs[0](x)
        for i in range(1, 5):
            x = F.interpolate(x, size=enc[i].shape[-2:], mode='bilinear', align_corners=False)
            skip = self.gates[i](enc[i], dogs[i])
            x = self.convs[i](torch.cat([x, skip], dim=1))
        return x
