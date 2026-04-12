import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveDoGModule(nn.Module):
    """
    Differentiable Difference-of-Gaussians at multiple scales.

    Sigma pairs are LEARNABLE — the network discovers which scale pairs are
    most informative for depth boundary detection during training.

    Key implementation details:
    - Parameterised as log(σ) to enforce positivity without clamping
    - Kernel is recomputed each forward pass from current σ values
    - return_split=True exposes (fine, coarse) DoG groups to the decoder
    - boundary_map() provides a normalised [0,1] edge saliency map used
      to weight the distillation loss

    Args:
        sigma_pairs : initial (σ_fine, σ_coarse) pairs — learned from here
        kernel_size : Gaussian kernel spatial extent (odd integer)
    """

    def __init__(self,
                 sigma_pairs: list = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
                 kernel_size: int = 9):
        super().__init__()
        self.num_scales = len(sigma_pairs)
        self.kernel_size = kernel_size

        log_s1 = torch.tensor([np.log(s1) for s1, _ in sigma_pairs], dtype=torch.float32)
        log_s2 = torch.tensor([np.log(s2) for _,  s2 in sigma_pairs], dtype=torch.float32)
        self.log_sigma1 = nn.Parameter(log_s1)   # (num_scales,) learnable
        self.log_sigma2 = nn.Parameter(log_s2)   # (num_scales,) learnable

        # Fixed coordinate grid for kernel evaluation
        half = kernel_size // 2
        ax = torch.arange(-half, half + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        self.register_buffer('grid', xx ** 2 + yy ** 2)   # (k, k)

    # ── internal helpers ────────────────────────────────────────────────────
    def _gaussian_kernel(self, sigma: torch.Tensor) -> torch.Tensor:
        """Differentiable, normalised Gaussian kernel for scalar sigma."""
        kernel = torch.exp(-self.grid / (2 * sigma ** 2 + 1e-8))
        return kernel / (kernel.sum() + 1e-8)               # (k, k)

    def _apply_gaussian(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        C, k = x.shape[1], self.kernel_size
        kernel = self._gaussian_kernel(sigma).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(C, 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2, groups=C)

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self,
                x: torch.Tensor,
                return_split: bool = False):
        """
        Args:
            x            : (B, 3, H, W) RGB image
            return_split : if True return (fine_dogs, coarse_dogs) tuple
                           for frequency-decomposed decoder injection;
                           otherwise return concatenated (B, 3·num_scales, H, W)
        """
        s1 = torch.exp(self.log_sigma1)
        s2 = torch.exp(self.log_sigma2)
        dogs = [self._apply_gaussian(x, s1[i]) - self._apply_gaussian(x, s2[i])
                for i in range(self.num_scales)]

        if return_split:
            mid = self.num_scales // 2
            return torch.cat(dogs[:mid + 1], dim=1), torch.cat(dogs[mid:], dim=1)
        return torch.cat(dogs, dim=1)

    # ── boundary map ────────────────────────────────────────────────────────
    def boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel boundary strength in [0, 1].
        High values = depth discontinuity likely present.
        Used to weight the DoG-guided distillation loss.
        """
        dogs = self.forward(x)
        b  = dogs.abs().mean(dim=1, keepdim=True)   # (B, 1, H, W)
        B  = b.shape[0]
        bf = b.view(B, -1)
        bmin = bf.min(1)[0].view(B, 1, 1, 1)
        bmax = bf.max(1)[0].view(B, 1, 1, 1)
        return (b - bmin) / (bmax - bmin + 1e-8)    # normalised [0, 1]

    # ── introspection ───────────────────────────────────────────────────────
    def get_learned_sigmas(self) -> list:
        s1 = torch.exp(self.log_sigma1).detach().cpu().tolist()
        s2 = torch.exp(self.log_sigma2).detach().cpu().tolist()
        return [(round(a, 4), round(b, 4)) for a, b in zip(s1, s2)]

    def sigma_summary(self) -> str:
        pairs = self.get_learned_sigmas()
        inner = ' '.join(f'({s1},{s2})' for s1, s2 in pairs)
        return f'σ[{inner}]'