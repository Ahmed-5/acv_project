import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    def __init__(self, lam=0.85):
        super().__init__()
        self.lam = lam

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        d = torch.log(pred[mask] + 1e-8) - torch.log(gt[mask] + 1e-8)
        return d.pow(2).mean() - self.lam * d.mean().pow(2)


class BerHuLoss(nn.Module):
    def __init__(self, threshold_frac=0.2):
        super().__init__()
        self.c_frac = threshold_frac

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        diff = (pred[mask] - gt[mask]).abs()
        c = self.c_frac * diff.max().detach()
        berhu = torch.where(diff <= c, diff, (diff.pow(2) + c ** 2) / (2 * c + 1e-8))
        return berhu.mean()


class Log10Loss(nn.Module):
    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        return (torch.log10(gt[mask] + 1e-8) - torch.log10(pred[mask] + 1e-8)).abs().mean()


class DoGWeightedDistillLoss(nn.Module):
    def __init__(self, lam=2.0):
        super().__init__()
        self.lam = lam

    def forward(self, student, teacher, boundary):
        weight = 1.0 + self.lam * boundary
        return (weight * (student - teacher).abs()).mean()


class CrossScaleConsistencyLoss(nn.Module):
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau

    def forward(self, depth, boundary):
        gx = (depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs()
        gy = (depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs()
        mx = (boundary[:, :, :, :-1] < self.tau).float()
        my = (boundary[:, :, :-1, :] < self.tau).float()
        return (gx * mx).mean() + (gy * my).mean()


class EdgeAwareLoss(nn.Module):
    def forward(self, depth, image):
        nd = depth / (depth.mean(dim=[2, 3], keepdim=True) + 1e-7)
        dx = (nd[:, :, :, :-1] - nd[:, :, :, 1:]).abs()
        dy = (nd[:, :, :-1, :] - nd[:, :, 1:, :]).abs()
        ix = image.mean(1, keepdim=True)
        ex = torch.exp(-(ix[:, :, :, :-1] - ix[:, :, :, 1:]).abs())
        ey = torch.exp(-(ix[:, :, :-1, :] - ix[:, :, 1:, :]).abs())
        return (dx * ex).mean() + (dy * ey).mean()


class DoGDepthLoss(nn.Module):
    """
    v3: separate weights so BerHu cannot swamp SI.
    """
    def __init__(self,
                 alpha_si=1.0,
                 alpha_berhu=0.1,
                 alpha_log10=0.05,
                 beta=0.3,
                 gamma=0.05,
                 delta=0.05):
        super().__init__()
        self.a_si = alpha_si
        self.a_bh = alpha_berhu
        self.a_lg = alpha_log10
        self.b = beta
        self.g = gamma
        self.d = delta
        self.L_si = ScaleInvariantLoss(lam=0.85)
        self.L_berhu = BerHuLoss()
        self.L_log10 = Log10Loss()
        self.L_distill = DoGWeightedDistillLoss()
        self.L_consist = CrossScaleConsistencyLoss()
        self.L_edge = EdgeAwareLoss()

    def forward(self, pred_s, pred_t, gt, image, boundary, mask=None):
        ls = self.L_si(pred_s, gt, mask)
        lb = self.L_berhu(pred_s, gt, mask)
        ll = self.L_log10(pred_s, gt, mask)
        ld = self.L_distill(pred_s, pred_t, boundary)
        lc = self.L_consist(pred_s, boundary)
        le = self.L_edge(pred_s, image)
        total = self.a_si * ls + self.a_bh * lb + self.a_lg * ll + self.b * ld + self.g * lc + self.d * le
        return dict(total=total, si=ls, berhu=lb, log10=ll, distill=ld, consistency=lc, edge=le)
