import copy
import math
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.dog_depth_net import DoGDepthNet
from losses.dog_losses import DoGDepthLoss
from data.nyu_dataset import build_loaders


class Config:
    IMG_SIZE = (320, 416)
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    NUM_EPOCHS = 30
    LR = 1e-4
    LR_DOG = LR * 0.1
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    EMA_DECAY = 0.999
    SIGMA_PAIRS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    BRANCH_CH = 64
    H5_TRAIN_DIR = './hf_dataset/data/train/'
    H5_VAL_DIR = './hf_dataset/data/val/'
    SAVE_PATH = 'checkpoints/dog_depth_nyu_v3.pth'
    RESUME_PATH = None


cfg = Config()


def build_student(device, pretrained=True):
    model = DoGDepthNet(
        sigma_pairs=cfg.SIGMA_PAIRS,
        branch_ch=cfg.BRANCH_CH,
        pretrained=pretrained,
    ).to(device)
    print(f'Student params: {model.count_params()}')
    return model


def build_optimizer(model, lr=1e-4):
    dog_params = list(model.dog.parameters())
    dog_ids = {id(p) for p in dog_params}
    base_params = [p for p in model.parameters() if id(p) not in dog_ids]
    return torch.optim.AdamW([
        {'params': base_params, 'lr': lr},
        {'params': dog_params, 'lr': cfg.LR_DOG},
    ], weight_decay=cfg.WEIGHT_DECAY)


def update_ema(student, teacher, decay):
    with torch.no_grad():
        s_ms = student.state_dict()
        for k, v in teacher.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(s_ms[k])
            else:
                v.mul_(decay).add_(s_ms[k], alpha=1.0 - decay)


def scale_align_like_eval(pred, gt, mask):
    out = pred.clone()
    m = mask.squeeze(1) if mask.dim() == 4 else mask
    g = gt.squeeze(1) if gt.dim() == 4 else gt
    for b in range(out.shape[0]):
        valid = m[b]
        if valid.sum() < 10:
            continue
        p = out[b, 0][valid]
        gg = g[b][valid]
        scale = torch.median(gg) / (torch.median(p) + 1e-8)
        out[b, 0] = out[b, 0] * scale.clamp(0.1, 10.0)
    return out


def train_one_epoch(student, teacher, loader, optimizer, criterion, scaler, device, scheduler=None):
    student.train()
    # teacher.eval()
    totals = {k: 0.0 for k in ['total', 'si', 'berhu', 'log10', 'distill', 'consistency', 'edge']}
    grad_norms = []

    for batch in tqdm(loader, desc='Training', leave=False):
        image = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask = batch['mask'].to(device)

        # with torch.no_grad():
        #     pred_teacher = teacher(image)
        #     pred_teacher = F.interpolate(pred_teacher, size=image.shape[-2:], mode='bilinear', align_corners=False)
        #     pred_teacher = scale_align_like_eval(pred_teacher, depth_gt, mask)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred_student = student(image)
            boundary = student.boundary_map(image)
            # losses = criterion(pred_student, pred_teacher, depth_gt, image, boundary, mask)
            losses = criterion(pred_student, torch.zeros_like(pred_student), depth_gt, image, boundary, mask)

        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=cfg.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        # update_ema(student, teacher, cfg.EMA_DECAY)

        for k in totals:
            totals[k] += losses[k].item()
        if torch.isfinite(torch.as_tensor(grad_norm)):
            grad_norms.append(float(grad_norm))

    n = len(loader)
    out = {k: v / n for k, v in totals.items()}
    out['grad_norm'] = float(np.mean(grad_norms)) if grad_norms else float('nan')
    return out


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    abs_rels, rmses, delta1s = [], [], []
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        image = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask = batch['mask'].squeeze(1).to(device)
        pred = F.interpolate(student(image), size=depth_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        gt = depth_gt.squeeze(1)
        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p = p * scale
            thresh = torch.max(p / (g + 1e-8), g / (p + 1e-8))
            abs_rels.append(((p - g).abs() / (g + 1e-8)).mean().item())
            rmses.append(((p - g).pow(2).mean().sqrt()).item())
            delta1s.append((thresh < 1.25).float().mean().item())
    return dict(AbsRel=np.mean(abs_rels), RMSE=np.mean(rmses), delta1=np.mean(delta1s))


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_loader, val_loader = build_loaders(
        mode='h5',
        h5_train_dir=cfg.H5_TRAIN_DIR,
        h5_val_dir=cfg.H5_VAL_DIR,
        img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    student = build_student(device)
    # teacher = build_student(device)
    # teacher.load_state_dict(copy.deepcopy(student.state_dict()))
    # teacher.eval()
    # for p in teacher.parameters():
    #     p.requires_grad_(False)

    optim = build_optimizer(student, lr=cfg.LR)
    crit = DoGDepthLoss(
        alpha_si=1.0,
        alpha_berhu=0.1,
        alpha_log10=0.05,
        beta=0.0,
        gamma=0.05,
        delta=0.05,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    if cfg.RESUME_PATH and os.path.exists(cfg.RESUME_PATH):
        checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
        student.load_state_dict(checkpoint['model_state'])
        optim.load_state_dict(checkpoint['optimizer_state'])
        print(f'Loaded checkpoint from {cfg.RESUME_PATH}')

    print(f'Initial sigmas: {student.get_learned_sigmas()}')

    best_rmse = float('inf')
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_losses = train_one_epoch(student, None, train_loader, optim, crit, scaler, device)
        val_metrics = evaluate(student, val_loader, device)
        sched.step()

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        lr_now = sched.get_last_lr()[0]
        print(
            f'Epoch {epoch:02d}/{cfg.NUM_EPOCHS} | '
            f'Time: {mins:02d}:{secs:02d} | '
            f'LR:{lr_now:.6f} | '
            f'Loss: {train_losses["total"]:.4f} '
            f'(SI:{train_losses["si"]:.4f} '
            f'BerHu:{train_losses["berhu"]:.4f} '
            f'Log10:{train_losses["log10"]:.4f} '
            f'Distill:{train_losses["distill"]:.4f}) | '
            f'GradNorm:{train_losses["grad_norm"]:.3f} | '
            f'Val AbsRel:{val_metrics["AbsRel"]:.4f} '
            f'RMSE:{val_metrics["RMSE"]:.4f} '
            f'δ₁:{val_metrics["delta1"]:.4f}'
        )

        if val_metrics['RMSE'] < best_rmse:
            best_rmse = val_metrics['RMSE']
            best_sigmas = student.get_learned_sigmas()
            torch.save({
                'epoch': epoch,
                'model_state': student.state_dict(),
                'optimizer_state': optim.state_dict(),
                'val_metrics': val_metrics,
                'learned_sigmas': best_sigmas,
            }, cfg.SAVE_PATH)
            print(f' ✓ Saved best model (RMSE={best_rmse:.4f}) sigmas={best_sigmas}')
