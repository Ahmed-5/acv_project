import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Efficient conv block ────────────────────────────────────────────────────
class DSConvBlock(nn.Module):
    """
    Depthwise-separable conv block.
    Params vs standard 3×3 Conv2d(C, C):
        Standard  : C × C × 9      (e.g. 48×48×9 = 20 736)
        DS        : C×9 + C×C      (e.g. 48×9 + 48×48 = 2 736)  — 7.6× fewer
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch,  in_ch,  3, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch,  out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ── DoG-gated skip connection ───────────────────────────────────────────────
class DoGAttentionGate(nn.Module):
    """
    Modulates encoder skip features using the DoG signal at the same spatial
    resolution. The gate learns which features matter for depth at each scale.

    Novel: uses DoG (not a learned attention query from the decoder) as the
    modulation signal — directly linking boundary detection to skip weighting.
    """
    def __init__(self, feat_ch: int, dog_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + dog_ch, feat_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, dog: torch.Tensor) -> torch.Tensor:
        dog_r = F.interpolate(dog, size=feat.shape[-2:],
                              mode='bilinear', align_corners=False)
        return feat * self.gate(torch.cat([feat, dog_r], dim=1))


# ── Single decoder branch ───────────────────────────────────────────────────
class DecoderBranch(nn.Module):
    """
    U-Net-style decoder with DoG-gated skip connections and DS-conv blocks.

    v2 changes vs v1:
    - SKIP_CHS[0] is 128 (was 1280) — matches SharedEncoder bottleneck
    - All 3×3 conv pairs replaced by DSConvBlock pairs (~7× param reduction
      per block)
    - Single branch (was dual) — fusion head handles frequency split

    SKIP_CHS must match SharedEncoder.SKIP_CHS exactly.
    """
    SKIP_CHS = [128, 96, 32, 24, 16]   # ← updated for bottlenecked encoder

    def __init__(self, dog_fine_ch: int, dog_coarse_ch: int, out_ch: int = 48):
        super().__init__()
        # Assign DoG frequency band per decoder stage
        # Stages 0-1 (coarse, 1/32–1/16): use coarse DoG
        # Stages 2-4 (fine,   1/8–1/2):   use fine DoG
        dog_chs = [dog_coarse_ch] * 2 + [dog_fine_ch] * 3

        self.gates = nn.ModuleList([
            DoGAttentionGate(self.SKIP_CHS[i], dog_chs[i])
            for i in range(5)
        ])

        # Stage 0: input = gated f4 only (128ch)
        # Stages 1-4: input = upsampled prev + gated skip
        in_chs = [128,
                  96  + out_ch,
                  32  + out_ch,
                  24  + out_ch,
                  16  + out_ch]

        self.convs = nn.ModuleList([
            nn.Sequential(
                DSConvBlock(c, out_ch),
                DSConvBlock(out_ch, out_ch),
            ) for c in in_chs
        ])

    def forward(self,
                feats:      tuple,
                dog_fine:   torch.Tensor,
                dog_coarse: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3, f4 = feats
        dogs = [dog_coarse] * 2 + [dog_fine] * 3
        enc  = [f4, f3, f2, f1, f0]            # top-down order

        x = self.gates[0](enc[0], dogs[0])     # gate f4
        x = self.convs[0](x)                   # (B, out_ch, H/32, W/32)

        for i in range(1, 5):
            x    = F.interpolate(x, size=enc[i].shape[-2:],
                                 mode='bilinear', align_corners=False)
            skip = self.gates[i](enc[i], dogs[i])
            x    = self.convs[i](torch.cat([x, skip], dim=1))

        return x  # (B, out_ch, H/2, W/2)