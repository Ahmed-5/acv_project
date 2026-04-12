import torch
import torch.nn as nn


# ── Reconstruction losses ───────────────────────────────────────────────────

class ScaleInvariantLoss(nn.Module):
    """
    Eigen et al. (2014) scale-invariant log loss.

    L_SI = mean(d²) − λ·mean(d)²
    where  d = log(pred) − log(gt)

    λ=0.85 (vs 0.5 in v1) pushes harder on global scale consistency,
    matching the setting used in most SOTA papers (AdaBins, BinsFormer, etc.).
    """
    def __init__(self, lam: float = 0.85):
        super().__init__()
        self.lam = lam

    def forward(self,
                pred:  torch.Tensor,
                gt:    torch.Tensor,
                mask:  torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        d = torch.log(pred[mask] + 1e-8) - torch.log(gt[mask] + 1e-8)
        return d.pow(2).mean() - self.lam * d.mean().pow(2)


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss — Owen (2012).

    L1 for small errors (preserves sharp edges), L2 for large ones (robust
    to outliers). Complementary to SILog: where SILog penalises log-space
    errors, BerHu penalises absolute metric errors.

    c = threshold_frac × max(|pred − gt|)  (adaptive threshold per batch)
    L_BerHu = |d|          if |d| ≤ c
            = (d² + c²) / 2c  otherwise
    """
    def __init__(self, threshold_frac: float = 0.2):
        super().__init__()
        self.c_frac = threshold_frac

    def forward(self,
                pred:  torch.Tensor,
                gt:    torch.Tensor,
                mask:  torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        diff  = (pred[mask] - gt[mask]).abs()
        c     = self.c_frac * diff.max().detach()
        berhu = torch.where(diff <= c,
                            diff,
                            (diff.pow(2) + c ** 2) / (2 * c + 1e-8))
        return berhu.mean()


class Log10Loss(nn.Module):
    """
    Mean absolute log₁₀ error.

    L_log10 = mean(|log₁₀(gt) − log₁₀(pred)|)

    Acts as an additional scale-aware metric loss that is sensitive to
    relative error at all depth ranges (near and far equally).
    """
    def forward(self,
                pred:  torch.Tensor,
                gt:    torch.Tensor,
                mask:  torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        return (torch.log10(gt[mask] + 1e-8)
                - torch.log10(pred[mask] + 1e-8)).abs().mean()


# ── Novel DoG-guided losses ──────────────────────────────────────────────────

class DoGWeightedDistillLoss(nn.Module):
    """
    NOVEL: Knowledge distillation weighted by the DoG boundary map.

    Boundary pixels contribute (1 + λ·DoG) times more to the loss,
    forcing the student to faithfully replicate the teacher at depth edges —
    the hardest part of depth estimation for lightweight models.

    L_distill = Σ_i (1 + λ·|DoG(I_i)|) · |D_student_i − D_teacher_i|
    """
    def __init__(self, lam: float = 2.0):
        super().__init__()
        self.lam = lam

    def forward(self,
                student:   torch.Tensor,
                teacher:   torch.Tensor,
                boundary:  torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.lam * boundary
        return (weight * (student - teacher).abs()).mean()


class CrossScaleConsistencyLoss(nn.Module):
    """
    NOVEL: Penalises depth gradients in smooth image regions.

    Enforces that depth only changes where the DoG says a real edge exists.
    In smooth regions (DoG < τ), depth gradients should be near zero.

    L_consist = ‖∇D_student · 1[DoG(I) < τ]‖₁
    """
    def __init__(self, tau: float = 0.05):
        super().__init__()
        self.tau = tau

    def forward(self,
                depth:    torch.Tensor,
                boundary: torch.Tensor) -> torch.Tensor:
        gx = (depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs()
        gy = (depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs()
        mx = (boundary[:, :, :, :-1] < self.tau).float()
        my = (boundary[:, :, :-1, :] < self.tau).float()
        return (gx * mx).mean() + (gy * my).mean()


class EdgeAwareLoss(nn.Module):
    """Standard image-guided depth smoothness loss."""
    def forward(self,
                depth: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        nd = depth / (depth.mean(dim=[2, 3], keepdim=True) + 1e-7)
        dx = (nd[:, :, :, :-1] - nd[:, :, :, 1:]).abs()
        dy = (nd[:, :, :-1, :] - nd[:, :, 1:, :]).abs()
        ix = image.mean(1, keepdim=True)
        ex = torch.exp(-(ix[:, :, :, :-1] - ix[:, :, :, 1:]).abs())
        ey = torch.exp(-(ix[:, :, :-1, :] - ix[:, :, 1:, :]).abs())
        return (dx * ex).mean() + (dy * ey).mean()


# ── Combined loss ────────────────────────────────────────────────────────────

class DoGDepthLoss(nn.Module):
    """
    Full combined loss for DoGDepthNet v2.

    L_total = α · (L_SI + L_BerHu + ε·L_log10)
            + β · L_distill_DoG(student, teacher, boundary)
            + γ · L_consistency(student, boundary)
            + δ · L_edge(student, image)

    Default weights are tuned for the v2 architecture.  Set beta=0 to
    disable distillation when no real teacher is available.

    Args:
        alpha   : weight for the reconstruction loss group
        beta    : weight for DoG-weighted distillation  (0 = disabled)
        gamma   : weight for cross-scale consistency
        delta   : weight for edge-aware smoothness
        epsilon : Log10 sub-weight within the reconstruction group
    """
    def __init__(self,
                 alpha:   float = 1.0,
                 beta:    float = 0.5,
                 gamma:   float = 0.1,
                 delta:   float = 0.1,
                 epsilon: float = 0.1):
        super().__init__()
        self.a = alpha
        self.b = beta
        self.g = gamma
        self.d = delta
        self.e = epsilon

        self.L_si      = ScaleInvariantLoss(lam=0.85)
        self.L_berhu   = BerHuLoss()
        self.L_log10   = Log10Loss()
        self.L_distill = DoGWeightedDistillLoss()
        self.L_consist = CrossScaleConsistencyLoss()
        self.L_edge    = EdgeAwareLoss()

    def forward(self,
                pred_s:   torch.Tensor,
                pred_t:   torch.Tensor,
                gt:       torch.Tensor,
                image:    torch.Tensor,
                boundary: torch.Tensor,
                mask:     torch.Tensor = None) -> dict:

        ls = self.L_si    (pred_s, gt, mask)
        lb = self.L_berhu (pred_s, gt, mask)
        ll = self.L_log10 (pred_s, gt, mask)
        ld = self.L_distill(pred_s, pred_t, boundary)
        lc = self.L_consist(pred_s, boundary)
        le = self.L_edge  (pred_s, image)

        total = (self.a * (ls + lb + self.e * ll)
                 + self.b * ld
                 + self.g * lc
                 + self.d * le)

        return dict(total=total, si=ls, berhu=lb, log10=ll,
                    distill=ld, consistency=lc, edge=le)