import os

import torch
import torch.nn.functional as F
import numpy as np

from models.dog_depth_net import DoGDepthNet
from losses.dog_losses    import DoGDepthLoss
from data.nyu_dataset     import build_loaders   # <-- new

from tqdm import tqdm

import time


def build_student(device, pretrained=True):
    return DoGDepthNet(
        sigma_pairs=[(0.5,1.0),(1.0,2.0),(2.0,4.0)],
        branch_ch=64,
        pretrained=pretrained
    ).to(device)


def build_optimizer(model, lr=1e-4):
    dog_params  = list(model.dog.parameters())
    base_params = [p for p in model.parameters()
                   if not any(p is dp for dp in dog_params)]
    return torch.optim.AdamW([
        {'params': base_params, 'lr': lr},
        {'params': dog_params,  'lr': lr * 0.1}
    ], weight_decay=1e-5)


def train_one_epoch(student, teacher, loader, optimizer, criterion, device):
    student.train();  teacher.eval()
    totals = {k: 0.0 for k in ['total','si','distill','consistency','edge']}

    for batch in tqdm(loader, desc="Training", leave=False):
        image    = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask     = batch['mask'].to(device)

        with torch.no_grad():
            pred_teacher = teacher(image)
            pred_teacher = F.interpolate(pred_teacher, size=image.shape[-2:],
                                         mode='bilinear', align_corners=False)

        pred_student = student(image)
        boundary     = student.boundary_map(image)

        losses = criterion(pred_student, pred_teacher, depth_gt,
                           image, boundary, mask)
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals: totals[k] += losses[k].item()

    n = len(loader)
    return {k: v/n for k,v in totals.items()}


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    abs_rels, rmses, delta1s = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image    = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask     = batch['mask'].squeeze(1).to(device)

        pred = F.interpolate(student(image), size=depth_gt.shape[-2:],
                             mode='bilinear', align_corners=False).squeeze(1)
        gt   = depth_gt.squeeze(1)

        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p = p * scale
            thresh = torch.max(p/(g+1e-8), g/(p+1e-8))
            abs_rels.append(((p-g).abs()/(g+1e-8)).mean().item())
            rmses.append(((p-g).pow(2).mean().sqrt()).item())
            delta1s.append((thresh < 1.25).float().mean().item())

    return dict(AbsRel=np.mean(abs_rels),
                RMSE  =np.mean(rmses),
                delta1=np.mean(delta1s))


if __name__ == '__main__':
    # ── Config ─────────────────────────────────────────────────────────
    IMG_SIZE   = (256, 320)   # NYU native is 480x640; downscale for speed
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LR         = 1e-4
    SAVE_PATH = 'checkpoints/dog_depth_nyu.pth'
    os.makedirs('checkpoints', exist_ok=True)   # ← add this line

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ───────────────────────────────────────────────────────────
    # train_loader, val_loader = build_loaders(
    #     img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=4
    # )

    # # Mode A: labeled .mat file (recommended — fastest, no toolbox needed per sample)
    # train_loader, val_loader = build_loaders(
    #     mode='mat',
    #     mat_path='./dataset/nyu_depth_v2_labeled.mat',
    #     img_size=IMG_SIZE, batch_size=BATCH_SIZE
    # )

    # # Mode B: raw scene directories (full toolbox pipeline per sample)
    # train_loader, val_loader = build_loaders(
    #     mode='raw',
    #     raw_root='./dataset/raw_scenes/',
    #     img_size=IMG_SIZE, batch_size=BATCH_SIZE, fill_depth=True
    # )

    # Mode C: preprocessed .h5 files (fast loading, no toolbox needed per sample)
    train_loader, val_loader = build_loaders(
        mode='h5',
        h5_train_dir='./hf_dataset/data/train/',
        h5_val_dir='./hf_dataset/data/val/',
        img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    # ── Models ─────────────────────────────────────────────────────────
    student = build_student(device)
    # Teacher: use student itself in no-distill mode if no external teacher,
    # or load SPIdepth / another pretrained model here.
    teacher = build_student(device, pretrained=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optim = build_optimizer(student, lr=LR)
    crit  = DoGDepthLoss(alpha=1.0, beta=0, gamma=0.1, delta=0.1)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=NUM_EPOCHS)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode      = 'min',
        factor    = 0.5,     # halve LR, not 0.1 (too aggressive for 20 epochs)
        patience  = 2,       # only wait 2 epochs, not default 10
        threshold = 1e-4,    # minimum RMSE improvement to count
        min_lr    = 1e-6,    # floor
        verbose   = True     # prints when LR drops
    )

    # code for loading from checkpoint if needed
    # if os.path.exists(SAVE_PATH):
    #     checkpoint = torch.load(SAVE_PATH)
    #     student.load_state_dict(checkpoint['student'])
    #     optim.load_state_dict(checkpoint['optim'])
    #     sched.load_state_dict(checkpoint['sched'])
    #     print(f"Loaded checkpoint from {SAVE_PATH}")

    total_p = sum(p.numel() for p in student.parameters())
    print(f"Params: {total_p/1e6:.2f}M | Device: {device}")
    print(f"Initial sigmas: {student.get_learned_sigmas()}")

    # ── Training loop ──────────────────────────────────────────────────
    best_rmse = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            student, teacher, train_loader, optim, crit, device
        )
        val_metrics = evaluate(student, val_loader, device)
        # sched.step()
        sched.step(val_metrics['RMSE'])  # Reduce LR if RMSE plateaus

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Time: {mins:02d}:{secs:02d} | "
              f"Loss: {train_losses['total']:.4f} "
              f"(SI:{train_losses['si']:.4f} "
              f"Distill:{train_losses['distill']:.4f}) | "
              f"Val AbsRel:{val_metrics['AbsRel']:.4f} "
              f"RMSE:{val_metrics['RMSE']:.4f} "
              f"δ₁:{val_metrics['delta1']:.4f}")

        if val_metrics['RMSE'] < best_rmse:
            best_rmse = val_metrics['RMSE']
            best_sigmas = student.get_learned_sigmas()
            torch.save({
                'epoch': epoch,
                'model_state': student.state_dict(),
                'optimizer_state': optim.state_dict(),
                'val_metrics': val_metrics,
                'learned_sigmas': best_sigmas,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model (RMSE={best_rmse:.4f}) with sigmas {best_sigmas}")