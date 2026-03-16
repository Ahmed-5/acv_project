import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dog_module  import AdaptiveDoGModule
from models.encoder     import SharedEncoder
from models.decoder     import DecoderBranch


class DoGDepthNet(nn.Module):
    """
    Adaptive Multi-Scale DoG Depth Network.

    Architecture:
        RGB ──► AdaptiveDoG ──► fine_dog + coarse_dog
                    │
        RGB ──► SharedEncoder ──► 5-level pyramid
                    │
        CoarseDecoder(pyramid, fine_dog, coarse_dog) ──► coarse_feats
        FineDecoder  (pyramid, fine_dog, coarse_dog) ──► fine_feats
                    │
        Fusion(coarse_feats ⊕ fine_feats) ──► depth map

    Novelty axes:
        1. Learnable sigma pairs (adaptive DoG)
        2. DoG-gated skip connections (boundary-aware attention)
        3. Frequency-split dual-branch decoder
        4. DoG-weighted distillation loss (in losses/)
        5. Cross-scale consistency loss     (in losses/)
    """
    def __init__(self,
                 sigma_pairs=[(0.5,1.0),(1.0,2.0),(2.0,4.0)],
                 branch_ch=64,
                 pretrained=True):
        super().__init__()
        self.dog     = AdaptiveDoGModule(sigma_pairs)
        self.encoder = SharedEncoder(pretrained=pretrained)

        n   = len(sigma_pairs);  mid = n // 2
        fch = (mid + 1) * 3          # fine  DoG channels  e.g. 6
        cch = (n - mid) * 3          # coarse DoG channels e.g. 6

        self.coarse_dec = DecoderBranch(fch, cch, branch_ch)
        self.fine_dec   = DecoderBranch(fch, cch, branch_ch)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_ch*2, branch_ch,    3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch,   branch_ch//2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch//2, 1, 1),
            nn.Softplus()   # positive depth
        )

    def forward(self, x):
        dog_fine, dog_coarse = self.dog(x, return_split=True)
        feats  = self.encoder(x)
        c_out  = self.coarse_dec(feats, dog_fine, dog_coarse)
        f_out  = self.fine_dec  (feats, dog_fine, dog_coarse)
        depth  = self.fusion(torch.cat([c_out, f_out], dim=1))
        return F.interpolate(depth, size=x.shape[-2:],
                             mode='bilinear', align_corners=False)

    def boundary_map(self, x):
        return self.dog.boundary_map(x)

    def get_learned_sigmas(self):
        """Inspect what scale pairs the network has discovered."""
        import torch
        s1 = torch.exp(self.dog.log_sigma1).detach().tolist()
        s2 = torch.exp(self.dog.log_sigma2).detach().tolist()
        return [(round(a,3), round(b,3)) for a,b in zip(s1,s2)]
