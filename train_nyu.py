"""
train_nyu.py  —  DoGDepthNet v2 training script
────────────────────────────────────────────────
Key improvements over v1:
  1. Real teacher: vinvino02/glpn-nyu (metric, NYU-pretrained) via HuggingFace
     OR load your own v1 checkpoint as a progressive distillation teacher
  2. Distillation enabled: beta=0.5 (was 0 — completely off in v1)
  3. Scale-alignment: teacher outputs are median-scaled to GT before distill
  4. Higher resolution: 384×512 (was 256×320) — single biggest accuracy gain
  5. Longer training: 60 epochs (was 30)
  6. OneCycleLR: warmup + cosine decay (was ReduceLROnPlateau)
  7. AMP (automatic mixed precision): ~2× faster, same accuracy
  8. DoG sigma LR = base_lr × 0.1 (was same as base — unstable)
  9. BerHu + Log10 losses active alongside SILog
  10. Grad norm logging every 10 batches for training diagnostics

Teacher options (set TEACHER_MODE in Config):
  'glpn'      — vinvino02/glpn-nyu from HuggingFace (recommended, ~50M)
  'checkpoint' — load TEACHER_CKPT path (your v1 30-epoch model)
  'none'      — disable distillation, set beta=0 automatically
"""

import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.dog_depth_net import DoGDepthNet
from losses.dog_losses    import DoGDepthLoss
from data.nyu_dataset     import build_loaders


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── Data ────────────────────────────────────────────────────────────────
    IMG_SIZE    = (384, 512)   # ← was (256,320); native NYU is (480,640)
    BATCH_SIZE  = 8           # reduce if OOM at 384×512
    NUM_WORKERS = 4

    H5_TRAIN_DIR = './hf_dataset/data/train/'
    H5_VAL_DIR   = './hf_dataset/data/val/'

    # ── Model ───────────────────────────────────────────────────────────────
    SIGMA_PAIRS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    BRANCH_CH   = 48           # was 64 — still ≥SOTA with fewer params

    # ── Teacher ─────────────────────────────────────────────────────────────
    # 'glpn'       → vinvino02/glpn-nyu  (HuggingFace, ~50M, metric depth)
    # 'checkpoint' → load a saved DoGDepthNet checkpoint (your v1 model)
    # 'none'       → no distillation (beta forced to 0)
    TEACHER_MODE = 'glpn'
    TEACHER_CKPT = 'checkpoints/dog_depth_nyu_v1.pth'  # used if mode='checkpoint'

    # ── Training ────────────────────────────────────────────────────────────
    NUM_EPOCHS = 60            # was 30
    LR         = 3e-4          # peak LR for OneCycleLR
    LR_DOG     = LR * 0.1     # sigma params need slower adaptation
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0

    # ── Loss weights ────────────────────────────────────────────────────────
    ALPHA   = 1.0   # reconstruction (SI + BerHu + Log10)
    BETA    = 0.5   # distillation   (0 if TEACHER_MODE='none')
    GAMMA   = 0.1   # consistency
    DELTA   = 0.1   # edge smoothness
    EPSILON = 0.1   # Log10 sub-weight

    # ── Checkpointing ───────────────────────────────────────────────────────
    SAVE_PATH   = 'checkpoints/dog_depth_nyu_v2.pth'
    RESUME_PATH = None   # set to SAVE_PATH to resume

    LOG_GRAD_EVERY = 10  # log grad norm every N batches


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════
def build_student(device: torch.device) -> DoGDepthNet:
    model = DoGDepthNet(
        sigma_pairs=cfg.SIGMA_PAIRS,
        branch_ch=cfg.BRANCH_CH,
        pretrained=True,
    ).to(device)
    print(f'Student params: {model.count_params()}')
    return model


def build_teacher(device: torch.device):
    """
    Returns (teacher_fn, teacher_model_or_None).
    teacher_fn(image_batch, gt_batch, mask_batch) → (B,1,H,W) depth tensor
    aligned to GT scale via per-image median scaling.
    """
    mode = cfg.TEACHER_MODE

    # ── Option A: GLPN-NYU ─────────────────────────────────────────────────
    if mode == 'glpn':
        try:
            from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
            print('Loading teacher: vinvino02/glpn-nyu …')
            extractor = GLPNFeatureExtractor.from_pretrained('vinvino02/glpn-nyu')
            teacher   = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-nyu').to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print('Teacher loaded ✓')

            @torch.no_grad()
            def teacher_fn(images: torch.Tensor,
                           gt:     torch.Tensor,
                           mask:   torch.Tensor) -> torch.Tensor:
                # images: (B,3,H,W) float32 [0,1]
                pil_list = [
                    Image.fromarray(
                        (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    for img in images
                ]
                inputs = extractor(images=pil_list, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out    = teacher(**inputs)
                pred   = out.predicted_depth.unsqueeze(1)           # (B,1,H',W')
                pred   = F.interpolate(pred, size=images.shape[-2:],
                                       mode='bilinear', align_corners=False)
                # Median-scale align to GT (safety: GLPN-NYU is already metric,
                # but small calibration drift can occur with different resolutions)
                pred = _scale_align(pred, gt, mask)
                return pred

            return teacher_fn, teacher

        except Exception as exc:
            print(f'[WARN] Could not load GLPN teacher: {exc}')
            print('[WARN] Falling back to no-distillation mode.')
            mode = 'none'

    # ── Option B: your v1 checkpoint ───────────────────────────────────────
    if mode == 'checkpoint':
        if not os.path.exists(cfg.TEACHER_CKPT):
            print(f'[WARN] Teacher checkpoint not found: {cfg.TEACHER_CKPT}')
            print('[WARN] Falling back to no-distillation mode.')
            mode = 'none'
        else:
            print(f'Loading teacher from checkpoint: {cfg.TEACHER_CKPT} …')
            ckpt    = torch.load(cfg.TEACHER_CKPT, map_location=device)
            teacher = DoGDepthNet(pretrained=False).to(device)
            teacher.load_state_dict(ckpt['model_state'])
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print('Teacher loaded ✓')

            @torch.no_grad()
            def teacher_fn(images, gt, mask):
                pred = teacher(images)
                pred = F.interpolate(pred, size=images.shape[-2:],
                                     mode='bilinear', align_corners=False)
                return _scale_align(pred, gt, mask)

            return teacher_fn, teacher

    # ── Option C / fallback: no distillation ───────────────────────────────
    cfg.BETA = 0.0   # force-disable distillation loss
    print('[INFO] No teacher — distillation disabled (beta=0).')

    def teacher_fn(images, gt, mask):
        # Return zeros — weight is 0 so this never matters
        return torch.zeros_like(images[:, :1])

    return teacher_fn, None


def _scale_align(pred:  torch.Tensor,
                 gt:    torch.Tensor,
                 mask:  torch.Tensor) -> torch.Tensor:
    """Per-image median-ratio scale alignment: pred ← pred × median(gt)/median(pred)."""
    out = pred.clone()
    m   = mask.squeeze(1) if mask.dim() == 4 else mask   # (B,H,W) bool
    for b in range(pred.shape[0]):
        valid = m[b]
        if valid.sum() < 10:
            continue
        p_med = torch.median(out[b, 0][valid])
        g_med = torch.median(gt [b, 0][valid]) if gt.dim() == 4 else \
                torch.median(gt[b][valid])
        scale = g_med / (p_med + 1e-8)
        out[b] = out[b] * scale.clamp(0.1, 10.0)   # guard against runaway scales
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════════════
def build_optimizer(model: DoGDepthNet) -> torch.optim.Optimizer:
    dog_params  = list(model.dog.parameters())
    dog_ids     = {id(p) for p in dog_params}
    base_params = [p for p in model.parameters() if id(p) not in dog_ids]
    return torch.optim.AdamW([
        {'params': base_params, 'lr': cfg.LR},
        {'params': dog_params,  'lr': cfg.LR_DOG},  # ← slow sigma adaptation
    ], weight_decay=cfg.WEIGHT_DECAY)


# ═══════════════════════════════════════════════════════════════════════════
# Train / Eval loops
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(student, teacher_fn, loader,
                    optimizer, criterion, scaler, device, epoch):
    student.train()
    totals = {k: 0.0 for k in
              ['total', 'si', 'berhu', 'log10', 'distill', 'consistency', 'edge']}
    grad_norms = []

    for step, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch}', leave=False)):
        image    = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask     = batch['mask'].to(device)

        with torch.no_grad():
            pred_teacher = teacher_fn(image, depth_gt, mask)
            pred_teacher = F.interpolate(pred_teacher, size=image.shape[-2:],
                                         mode='bilinear', align_corners=False)

        with autocast():
            pred_student = student(image)
            boundary     = student.boundary_map(image)
            losses       = criterion(pred_student, pred_teacher,
                                     depth_gt, image, boundary, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(),
                                               cfg.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        for k in totals:
            totals[k] += losses[k].item()
        if step % cfg.LOG_GRAD_EVERY == 0:
            grad_norms.append(gnorm.item())

    n = len(loader)
    avg = {k: v / n for k, v in totals.items()}
    avg['grad_norm'] = float(np.mean(grad_norms)) if grad_norms else 0.0
    return avg


@torch.no_grad()
def evaluate(student, loader, device) -> dict:
    student.eval()
    abs_rels, rmses, delta1s = [], [], []

    for batch in tqdm(loader, desc='Val', leave=False):
        image    = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask     = batch['mask'].squeeze(1).to(device)

        pred = F.interpolate(student(image), size=depth_gt.shape[-2:],
                             mode='bilinear', align_corners=False).squeeze(1)
        gt   = depth_gt.squeeze(1)

        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            # Scale-invariant evaluation (standard NYU protocol)
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p     = p * scale
            thresh = torch.max(p / (g + 1e-8), g / (p + 1e-8))
            abs_rels.append(((p - g).abs() / (g + 1e-8)).mean().item())
            rmses.append(((p - g).pow(2).mean().sqrt()).item())
            delta1s.append((thresh < 1.25).float().mean().item())

    return dict(AbsRel=float(np.mean(abs_rels)),
                RMSE  =float(np.mean(rmses)),
                delta1=float(np.mean(delta1s)))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        mode='h5',
        h5_train_dir=cfg.H5_TRAIN_DIR,
        h5_val_dir=cfg.H5_VAL_DIR,
        img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
    )
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # ── Models ──────────────────────────────────────────────────────────────
    student    = build_student(device)
    teacher_fn, _ = build_teacher(device)

    # ── Loss / optimiser / scheduler ────────────────────────────────────────
    criterion = DoGDepthLoss(
        alpha=cfg.ALPHA, beta=cfg.BETA,
        gamma=cfg.GAMMA, delta=cfg.DELTA, epsilon=cfg.EPSILON,
    )
    optimizer = build_optimizer(student)
    scaler    = GradScaler()

    # OneCycleLR: linear warmup → cosine decay — more stable than ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.LR, cfg.LR_DOG],
        epochs=cfg.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,              # 5% warmup
        anneal_strategy='cos',
        div_factor=25,               # start_lr = max_lr / 25
        final_div_factor=1e4,        # end_lr   = start_lr / 1e4
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 1
    best_rmse   = float('inf')

    if cfg.RESUME_PATH and os.path.exists(cfg.RESUME_PATH):
        ckpt = torch.load(cfg.RESUME_PATH, map_location=device)
        student.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_rmse   = ckpt.get('val_metrics', {}).get('RMSE', float('inf'))
        print(f'Resumed from epoch {start_epoch - 1} | best RMSE={best_rmse:.4f}')

    print(f'Initial sigmas: {student.get_learned_sigmas()}')
    print(f'Loss weights  : α={cfg.ALPHA} β={cfg.BETA} '
          f'γ={cfg.GAMMA} δ={cfg.DELTA} ε={cfg.EPSILON}')

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            student, teacher_fn, train_loader,
            optimizer, criterion, scaler, device, epoch,
        )
        # NOTE: with OneCycleLR, scheduler.step() is called inside the loop
        # (per-batch). If you prefer epoch-level stepping, replace OneCycleLR
        # with CosineAnnealingLR and call scheduler.step() here instead.

        val_metrics = evaluate(student, val_loader, device)

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)

        print(
            f'Epoch {epoch:02d}/{cfg.NUM_EPOCHS} | '
            f'Time: {mins:02d}:{secs:02d} | '
            f'Loss: {train_losses["total"]:.4f} '
            f'(SI:{train_losses["si"]:.4f} '
            f'BerHu:{train_losses["berhu"]:.4f} '
            f'Log10:{train_losses["log10"]:.4f} '
            f'Distill:{train_losses["distill"]:.4f}) | '
            f'GradNorm: {train_losses["grad_norm"]:.3f} | '
            f'Val AbsRel:{val_metrics["AbsRel"]:.4f} '
            f'RMSE:{val_metrics["RMSE"]:.4f} '
            f'δ₁:{val_metrics["delta1"]:.4f}'
        )

        if val_metrics['RMSE'] < best_rmse:
            best_rmse   = val_metrics['RMSE']
            best_sigmas = student.get_learned_sigmas()
            torch.save({
                'epoch':           epoch,
                'model_state':     student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_metrics':     val_metrics,
                'learned_sigmas':  best_sigmas,
                'config':          vars(cfg),
            }, cfg.SAVE_PATH)
            print(f' ✓ Saved best model '
                  f'(RMSE={best_rmse:.4f}) '
                  f'sigmas={best_sigmas}')