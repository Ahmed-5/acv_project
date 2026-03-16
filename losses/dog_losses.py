import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Eigen et al. (2014) scale-invariant log loss for metric depth GT."""
    def __init__(self, lam=0.5):
        super().__init__()
        self.lam = lam

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        d = torch.log(pred[mask] + 1e-8) - torch.log(gt[mask] + 1e-8)
        return d.pow(2).mean() - self.lam * d.mean().pow(2)


class DoGWeightedDistillLoss(nn.Module):
    """
    NOVEL: Distillation loss weighted by DoG boundary map.
    Boundary pixels contribute (1 + λ·DoG) times more to the loss,
    forcing the student to faithfully mimic the teacher at depth edges.

    L = Σ_i (1 + λ·|DoG(I_i)|) · |D_student_i − D_teacher_i|
    """
    def __init__(self, lam=2.0):
        super().__init__()
        self.lam = lam

    def forward(self, student, teacher, boundary):
        weight = 1.0 + self.lam * boundary
        return (weight * (student - teacher).abs()).mean()


class CrossScaleConsistencyLoss(nn.Module):
    """
    NOVEL: Penalizes depth gradients in smooth image regions.
    Enforces that depth only changes where the DoG says there is a real edge.

    L = || ∇D_student · 1[DoG(I) < τ] ||₁
    """
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau

    def forward(self, depth, boundary):
        gx = (depth[:,:,:,:-1] - depth[:,:,:,1:]).abs()
        gy = (depth[:,:,:-1,:] - depth[:,:,1:,:]).abs()
        mx = (boundary[:,:,:,:-1] < self.tau).float()   # smooth mask X
        my = (boundary[:,:,:-1,:] < self.tau).float()   # smooth mask Y
        return (gx * mx).mean() + (gy * my).mean()


class EdgeAwareLoss(nn.Module):
    """Standard image-guided depth smoothness loss."""
    def forward(self, depth, image):
        nd = depth / (depth.mean(dim=[2,3], keepdim=True) + 1e-7)
        dx = (nd[:,:,:,:-1] - nd[:,:,:,1:]).abs()
        dy = (nd[:,:,:-1,:] - nd[:,:,1:,:]).abs()
        ix = image.mean(1, keepdim=True)
        ex = torch.exp(-(ix[:,:,:,:-1] - ix[:,:,:,1:]).abs())
        ey = torch.exp(-(ix[:,:,:-1,:] - ix[:,:,1:,:]).abs())
        return (dx*ex).mean() + (dy*ey).mean()


class DoGDepthLoss(nn.Module):
    """
    Full combined loss. Maps to Equation (1) in the report:

    L_total = α·L_SI(student, gt)
            + β·L_distill_DoG(student, teacher, boundary)
            + γ·L_consistency(student, boundary)
            + δ·L_edge(student, image)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, delta=0.1):
        super().__init__()
        self.a, self.b, self.g, self.d = alpha, beta, gamma, delta
        self.L_si      = ScaleInvariantLoss()
        self.L_distill = DoGWeightedDistillLoss()
        self.L_consist = CrossScaleConsistencyLoss()
        self.L_edge    = EdgeAwareLoss()

    def forward(self, pred_s, pred_t, gt, image, boundary, mask=None):
        ls = self.L_si     (pred_s, gt, mask)
        ld = self.L_distill(pred_s, pred_t, boundary)
        lc = self.L_consist(pred_s, boundary)
        le = self.L_edge   (pred_s, image)
        total = self.a*ls + self.b*ld + self.g*lc + self.d*le
        return dict(total=total, si=ls, distill=ld, consistency=lc, edge=le)
