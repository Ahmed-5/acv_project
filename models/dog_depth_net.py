import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dog_module import AdaptiveDoGModule
from models.encoder   import SharedEncoder
from models.decoder   import DecoderBranch


class DoGDepthNet(nn.Module):
    """
    Adaptive Multi-Scale DoG Depth Network — v2 (efficient).

    Architecture
    ────────────
    RGB ──► AdaptiveDoGModule ──► dog_fine, dog_coarse
     │
    RGB ──► SharedEncoder (MBV2 + 1280→128 bottleneck) ──► 5-level pyramid
     │
    DecoderBranch(pyramid, dog_fine, dog_coarse)  ──► features (B, ch, H/2, W/2)
     │
    FusionHead (1×1 → DS 3×3 → 1×1 → Softplus) ──► depth (B, 1, H, W)

    v2 improvements over v1
    ───────────────────────
    • Single decoder branch (v1 had two) ............. −~2.0M params
    • 1280→128 encoder bottleneck .................... −~3.3M params
    • All 3×3 pairs → DSConvBlock pairs .............. −~40% decoder params
    • Lighter fusion head (no 128-ch 3×3 pair) ....... −~0.3M params
    Estimated total: ~3.5–4M  (v1 was ~8M)

    Novelty axes (unchanged from v1)
    ─────────────────────────────────
    1. Learnable sigma pairs (adaptive DoG)
    2. DoG-gated skip connections (boundary-aware attention)
    3. Frequency-split decoder injection
    4. DoG-weighted distillation loss
    5. Cross-scale consistency loss
    """

    def __init__(self,
                 sigma_pairs: list = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
                 branch_ch:   int  = 48,
                 pretrained:  bool = True):
        super().__init__()
        self.dog     = AdaptiveDoGModule(sigma_pairs)
        self.encoder = SharedEncoder(pretrained=pretrained)

        n   = len(sigma_pairs)
        mid = n // 2
        fch = (mid + 1) * 3   # fine DoG channels   e.g. 3×(1+1)=6
        cch = (n - mid) * 3   # coarse DoG channels  e.g. 3×(3-1)=6

        self.decoder = DecoderBranch(fch, cch, branch_ch)

        # Lightweight fusion head: 1×1 pointwise → DS 3×3 → 1×1 → Softplus
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_ch, branch_ch,      kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(branch_ch, branch_ch,      kernel_size=3,
                      padding=1, groups=branch_ch, bias=False),  # DW
            nn.Conv2d(branch_ch, branch_ch // 2, kernel_size=1, bias=False),  # PW
            nn.BatchNorm2d(branch_ch // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(branch_ch // 2, 1, kernel_size=1),
            nn.Softplus(),   # enforces positive depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dog_fine, dog_coarse = self.dog(x, return_split=True)
        feats = self.encoder(x)
        out   = self.decoder(feats, dog_fine, dog_coarse)
        depth = self.fusion(out)
        return F.interpolate(depth, size=x.shape[-2:],
                             mode='bilinear', align_corners=False)

    def boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.dog.boundary_map(x)

    def get_learned_sigmas(self) -> list:
        s1 = torch.exp(self.dog.log_sigma1).detach().tolist()
        s2 = torch.exp(self.dog.log_sigma2).detach().tolist()
        return [(round(a, 3), round(b, 3)) for a, b in zip(s1, s2)]

    def count_params(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f'{n/1e6:.2f}M'