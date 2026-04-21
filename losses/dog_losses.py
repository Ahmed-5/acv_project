import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """
    BTS-style scaled silog. We return:
        sqrt(D.pow(2).mean() - lam * D.mean().pow(2)) * scale
    where D = log(pred) - log(gt). `scale=10.0` is the BTS convention.
    """
    def __init__(self, lam: float = 0.85, scale: float = 10.0):
        super().__init__()
        self.lam = lam
        self.scale = scale

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = (gt > 0) & (pred > 0)
        d = torch.log(pred[mask] + 1e-8) - torch.log(gt[mask] + 1e-8)
        var = d.pow(2).mean() - self.lam * d.mean().pow(2)
        return torch.sqrt(var.clamp_min(1e-8)) * self.scale


class BerHuLoss(nn.Module):
    def __init__(self, threshold_frac: float = 0.2):
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


class LogGradientLoss(nn.Module):
    """
    L1 distance between Sobel gradients of log-depth.
    Improves edge sharpness; safe under per-image scale ambiguity since log-grad
    is invariant to global scale.
    """
    def forward(self, pred, gt, mask=None):
        eps = 1e-3
        lp = torch.log(pred.clamp_min(eps))
        lg = torch.log(gt.clamp_min(eps))
        gx_p = lp[..., :, 1:] - lp[..., :, :-1]
        gy_p = lp[..., 1:, :] - lp[..., :-1, :]
        gx_g = lg[..., :, 1:] - lg[..., :, :-1]
        gy_g = lg[..., 1:, :] - lg[..., :-1, :]
        if mask is not None:
            mx = (mask[..., :, 1:] & mask[..., :, :-1]).float()
            my = (mask[..., 1:, :] & mask[..., :-1, :]).float()
            lx = ((gx_p - gx_g).abs() * mx).sum() / mx.sum().clamp_min(1.0)
            ly = ((gy_p - gy_g).abs() * my).sum() / my.sum().clamp_min(1.0)
            return lx + ly
        return (gx_p - gx_g).abs().mean() + (gy_p - gy_g).abs().mean()


class VirtualNormalLoss(nn.Module):
    """
    Virtual Normal Loss (Yin et al. 2019), simplified.

    For each image we sample `n_groups` triplets of valid pixels, back-project
    them to 3D using a fixed pinhole intrinsic (NYU default focal ~518 px at
    480x640, scaled to current resolution), compute the unit normal of each
    triangle from pred and gt, and minimise the cosine distance.

    A single scaled focal is sufficient because the normal direction depends
    on relative point positions; absolute focal cancels in normalisation. We
    only need a focal-to-resolution ratio that is roughly correct.
    """
    NYU_FOCAL_AT_640 = 518.857

    def __init__(self, n_groups: int = 100, sample_per_img: int = 3):
        super().__init__()
        self.n_groups = n_groups
        self.sample_per_img = sample_per_img

    @staticmethod
    def _normals(points):
        a = points[:, :, 1] - points[:, :, 0]
        b = points[:, :, 2] - points[:, :, 0]
        n = torch.cross(a, b, dim=-1)
        return n / (n.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, pred, gt, mask=None):
        B, _, H, W = pred.shape
        device = pred.device
        focal = self.NYU_FOCAL_AT_640 * (W / 640.0)
        cx, cy = W / 2.0, H / 2.0

        if mask is None:
            mask = (gt > 0) & (pred > 0)
        mask = mask.view(B, -1)

        losses = []
        for b in range(B):
            valid = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() < self.n_groups * self.sample_per_img + 8:
                continue
            idx = valid[torch.randint(0, valid.numel(), (self.n_groups * self.sample_per_img,), device=device)]
            ys = (idx // W).float()
            xs = (idx % W).float()
            X = (xs - cx) / focal
            Y = (ys - cy) / focal

            zp = pred[b, 0].view(-1)[idx]
            zg = gt[b, 0].view(-1)[idx]
            pp = torch.stack([X * zp, Y * zp, zp], dim=-1).view(self.n_groups, self.sample_per_img, 3)
            gg = torch.stack([X * zg, Y * zg, zg], dim=-1).view(self.n_groups, self.sample_per_img, 3)

            np_ = self._normals(pp.unsqueeze(0)).squeeze(0)
            ng_ = self._normals(gg.unsqueeze(0)).squeeze(0)
            cos = (np_ * ng_).sum(dim=-1).clamp(-1.0, 1.0)
            losses.append((1.0 - cos).mean())

        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()


class CrossScaleConsistencyLoss(nn.Module):
    def __init__(self, tau: float = 0.05):
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
    v4 (no-teacher): supervised composite for DoGDepthNet.

    total = alpha_si * SI + alpha_berhu * BerHu + alpha_log10 * Log10
          + beta_grad * LogGrad + beta_vnl * VNL
          + gamma * CrossScaleConsistency + delta * EdgeAware

    Defaults are tuned for NYU @ 320x416 with the EfficientNet-B0 + AdaBins-lite stack.
    """
    def __init__(self,
                 alpha_si: float = 1.0,
                 alpha_berhu: float = 0.05,
                 alpha_log10: float = 0.025,
                 beta_grad: float = 0.25,
                 beta_vnl: float = 0.1,
                 gamma: float = 0.05,
                 delta: float = 0.05,
                 vnl_n_groups: int = 100):
        super().__init__()
        self.a_si = alpha_si
        self.a_bh = alpha_berhu
        self.a_lg = alpha_log10
        self.b_g = beta_grad
        self.b_v = beta_vnl
        self.g = gamma
        self.d = delta
        self.L_si = ScaleInvariantLoss(lam=0.85, scale=10.0)
        self.L_berhu = BerHuLoss()
        self.L_log10 = Log10Loss()
        self.L_grad = LogGradientLoss()
        self.L_vnl = VirtualNormalLoss(n_groups=vnl_n_groups)
        self.L_consist = CrossScaleConsistencyLoss()
        self.L_edge = EdgeAwareLoss()

    def forward(self, pred, gt, image, boundary, mask=None):
        ls = self.L_si(pred, gt, mask)
        lb = self.L_berhu(pred, gt, mask)
        ll = self.L_log10(pred, gt, mask)
        lg = self.L_grad(pred, gt, mask)
        lv = self.L_vnl(pred, gt, mask)
        lc = self.L_consist(pred, boundary)
        le = self.L_edge(pred, image)
        total = (
            self.a_si * ls
            + self.a_bh * lb
            + self.a_lg * ll
            + self.b_g * lg
            + self.b_v * lv
            + self.g * lc
            + self.d * le
        )
        return dict(
            total=total, si=ls, berhu=lb, log10=ll,
            log_grad=lg, vnl=lv, consistency=lc, edge=le,
        )
