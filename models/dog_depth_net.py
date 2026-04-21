import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dog_module import AdaptiveDoGModule
from models.encoder import SharedEncoder
from models.decoder import DecoderBranch


class AdaBinsLiteHead(nn.Module):
    """
    Lightweight adaptive-bin depth head.

    - Per-pixel softmax over `n_bins` bin logits (1x1 conv).
    - Per-image bin widths from global-pooled fused features (small MLP).
    - Final depth = sum_n softmax(logits)_n * center_n.

    Centers are guaranteed to lie in [min_depth, max_depth] by construction.
    """

    def __init__(self, in_ch: int, n_bins: int = 64,
                 min_depth: float = 1e-3, max_depth: float = 10.0):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_logits = nn.Conv2d(in_ch, n_bins, 1)
        self.center_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, in_ch),
            nn.ReLU6(inplace=True),
            nn.Linear(in_ch, n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.bin_logits(x), dim=1)
        w = torch.softmax(self.center_mlp(x), dim=1) * (self.max_depth - self.min_depth)
        edges = torch.cumsum(w, dim=1)
        centers = self.min_depth + edges - w / 2.0
        centers = centers.unsqueeze(-1).unsqueeze(-1)
        return (p * centers).sum(dim=1, keepdim=True)


class DoGDepthNet(nn.Module):
    """
    v4: EfficientNet-B0 encoder (FixEffNet weights) + dual DSConv decoders +
    AdaBins-lite bin-regression head.
    """
    def __init__(self,
                 sigma_pairs=[(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
                 branch_ch: int = 64,
                 n_bins: int = 64,
                 min_depth: float = 1e-3,
                 max_depth: float = 10.0,
                 pretrained: bool = True):
        super().__init__()
        self.dog = AdaptiveDoGModule(sigma_pairs)
        self.encoder = SharedEncoder(pretrained=pretrained)

        n = len(sigma_pairs)
        mid = n // 2
        fch = (mid + 1) * 3
        cch = (n - mid) * 3

        self.coarse_dec = DecoderBranch(fch, cch, branch_ch)
        self.fine_dec = DecoderBranch(fch, cch, branch_ch)

        self.fusion = nn.Sequential(
            nn.Conv2d(branch_ch * 2, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, groups=branch_ch, bias=False),
            nn.Conv2d(branch_ch, branch_ch // 2, 1, bias=False),
            nn.BatchNorm2d(branch_ch // 2),
            nn.ReLU6(inplace=True),
        )
        self.head = AdaBinsLiteHead(
            branch_ch // 2, n_bins=n_bins,
            min_depth=min_depth, max_depth=max_depth,
        )

    def forward(self, x):
        dog_fine, dog_coarse = self.dog(x, return_split=True)
        feats = self.encoder(x)
        c_out = self.coarse_dec(feats, dog_fine, dog_coarse)
        f_out = self.fine_dec(feats, dog_fine, dog_coarse)
        fused = self.fusion(torch.cat([c_out, f_out], dim=1))
        depth = self.head(fused)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return depth

    def boundary_map(self, x):
        return self.dog.boundary_map(x)

    def get_learned_sigmas(self):
        s1 = torch.exp(self.dog.log_sigma1).detach().tolist()
        s2 = torch.exp(self.dog.log_sigma2).detach().tolist()
        return [(round(a, 3), round(b, 3)) for a, b in zip(s1, s2)]

    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        return f'{n/1e6:.2f}M'
