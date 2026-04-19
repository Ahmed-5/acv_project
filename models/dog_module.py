import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveDoGModule(nn.Module):
    def __init__(self,
                 sigma_pairs=[(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
                 kernel_size=9):
        super().__init__()
        self.num_scales = len(sigma_pairs)
        self.kernel_size = kernel_size
        log_s1 = torch.tensor([np.log(s1) for s1, _ in sigma_pairs], dtype=torch.float32)
        log_s2 = torch.tensor([np.log(s2) for _, s2 in sigma_pairs], dtype=torch.float32)
        self.log_sigma1 = nn.Parameter(log_s1)
        self.log_sigma2 = nn.Parameter(log_s2)
        half = kernel_size // 2
        ax = torch.arange(-half, half + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        self.register_buffer('grid', xx ** 2 + yy ** 2)

    def _gaussian_kernel(self, sigma):
        kernel = torch.exp(-self.grid / (2 * sigma ** 2 + 1e-8))
        return kernel / (kernel.sum() + 1e-8)

    def _apply_gaussian(self, x, sigma):
        c, k = x.shape[1], self.kernel_size
        kernel = self._gaussian_kernel(sigma).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(c, 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2, groups=c)

    def forward(self, x, return_split=False):
        s1 = torch.exp(self.log_sigma1)
        s2 = torch.exp(self.log_sigma2)
        dogs = [self._apply_gaussian(x, s1[i]) - self._apply_gaussian(x, s2[i])
                for i in range(self.num_scales)]
        if return_split:
            mid = self.num_scales // 2
            return torch.cat(dogs[:mid + 1], dim=1), torch.cat(dogs[mid:], dim=1)
        return torch.cat(dogs, dim=1)

    def boundary_map(self, x):
        dogs = self.forward(x)
        b = dogs.abs().mean(dim=1, keepdim=True)
        B = b.shape[0]
        bf = b.view(B, -1)
        bmin = bf.min(1)[0].view(B, 1, 1, 1)
        bmax = bf.max(1)[0].view(B, 1, 1, 1)
        return (b - bmin) / (bmax - bmin + 1e-8)

    def get_learned_sigmas(self):
        s1 = torch.exp(self.log_sigma1).detach().cpu().tolist()
        s2 = torch.exp(self.log_sigma2).detach().cpu().tolist()
        return [(round(a, 4), round(b, 4)) for a, b in zip(s1, s2)]
