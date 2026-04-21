"""
Train DoGDepthNet (v4: EfficientNet-B0 + AdaBins-lite head) on NYU Depth V2.

Highlights vs. previous version:
  - No teacher / distillation. Pure supervised training.
  - OneCycle LR schedule.
  - Gradient accumulation -> larger effective batch on small GPUs.
  - Weight EMA of the student; EMA model is used for validation + checkpointing.
  - Two-stage "FixEfficientNet" recipe: train at low res, fine-tune at higher res.
  - Reports the full SPIdepth-style metric dict (a1/a2/a3, abs_rel, sq_rel,
    rmse, log_10, rmse_log, silog) at validation time.
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.nyu_dataset import build_loaders
from losses.dog_losses import DoGDepthLoss
from models.dog_depth_net import DoGDepthNet


class Config:
    IMG_SIZE = (320, 416)
    FT_IMG_SIZE = (480, 640)
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 50
    NUM_FT_EPOCHS = 10
    MAX_LR = 3e-4
    LR_DOG_FRAC = 0.1
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    EMA_DECAY = 0.9995
    EMA_WARMUP_STEPS = 2000
    SIGMA_PAIRS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    BRANCH_CH = 64
    N_BINS = 64
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 10.0
    H5_TRAIN_DIR = './hf_dataset/data/train/'
    H5_VAL_DIR = './hf_dataset/data/val/'
    SAVE_PATH = 'checkpoints/dog_depth_nyu_v4.pth'
    SAVE_PATH_FT = 'checkpoints/dog_depth_nyu_v4_ft.pth'
    LATEST_PATH = 'checkpoints/dog_depth_nyu_v4_latest.pth'
    LATEST_PATH_FT = 'checkpoints/dog_depth_nyu_v4_ft_latest.pth'
    RESUME_PATH = None


cfg = Config()


def build_model(device, pretrained=True):
    model = DoGDepthNet(
        sigma_pairs=cfg.SIGMA_PAIRS,
        branch_ch=cfg.BRANCH_CH,
        n_bins=cfg.N_BINS,
        min_depth=cfg.MIN_DEPTH,
        max_depth=cfg.MAX_DEPTH,
        pretrained=pretrained,
    ).to(device)
    print(f'Model params: {model.count_params()}')
    return model


def build_optimizer(model, max_lr):
    dog_params = list(model.dog.parameters())
    dog_ids = {id(p) for p in dog_params}
    base_params = [p for p in model.parameters() if id(p) not in dog_ids]
    return torch.optim.AdamW([
        {'params': base_params, 'lr': max_lr},
        {'params': dog_params, 'lr': max_lr * cfg.LR_DOG_FRAC},
    ], weight_decay=cfg.WEIGHT_DECAY)


def make_ema(model):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def update_ema(student, ema, decay):
    s_ms = student.state_dict()
    for k, v in ema.state_dict().items():
        if not v.dtype.is_floating_point:
            v.copy_(s_ms[k])
        else:
            v.mul_(decay).add_(s_ms[k], alpha=1.0 - decay)


def effective_ema_decay(step: int, warmup_steps: int, target_decay: float) -> float:
    """Ramp EMA decay from ~0 to target during `warmup_steps` optimizer steps.

    Uses the Polyak ramp `min(target, (1+step)/(10+step))`. At step 0 the
    effective decay is 0.1 (fast update); it approaches `target` after
    roughly `warmup_steps` steps. For `warmup_steps <= 0` this is a no-op
    and the target decay is used from step 0.
    """
    if warmup_steps <= 0:
        return target_decay
    ramp = (1.0 + step) / (10.0 + step)
    return min(target_decay, ramp)


def compute_errors(gt: np.ndarray, pred: np.ndarray):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    return dict(
        a1=float(a1), a2=float(a2), a3=float(a3),
        abs_rel=float(abs_rel), sq_rel=float(sq_rel),
        rmse=float(rmse), log_10=float(log_10),
        rmse_log=float(rmse_log), silog=float(silog),
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, scheduler, ema, accum_steps,
                    ema_step_count: int = 0,
                    ema_warmup_steps: int = 0,
                    ema_target_decay: float = 0.9995):
    model.train()
    totals = {k: 0.0 for k in
              ['total', 'si', 'berhu', 'log10', 'log_grad', 'vnl',
               'consistency', 'edge']}
    grad_norms = []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc='Training', leave=False)):
        image = batch['image'].to(device, non_blocking=True)
        depth_gt = batch['depth'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        with autocast():
            pred = model(image)
            boundary = model.boundary_map(image)
            losses = criterion(pred, depth_gt, image, boundary, mask)
            loss = losses['total'] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=cfg.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            decay = effective_ema_decay(ema_step_count,
                                        ema_warmup_steps,
                                        ema_target_decay)
            update_ema(model, ema, decay)
            ema_step_count += 1
            if torch.isfinite(torch.as_tensor(grad_norm)):
                grad_norms.append(float(grad_norm))

        for k in totals:
            totals[k] += losses[k].item()

    n = len(loader)
    out = {k: v / n for k, v in totals.items()}
    out['grad_norm'] = float(np.mean(grad_norms)) if grad_norms else float('nan')
    out['ema_step_count'] = ema_step_count
    return out


@torch.no_grad()
def evaluate(model, loader, device, min_depth=1e-3, max_depth=10.0):
    """Per-image median scaling; full SPIdepth-style metric set."""
    model.eval()
    agg = {k: [] for k in
           ['a1', 'a2', 'a3', 'abs_rel', 'sq_rel', 'rmse',
            'log_10', 'rmse_log', 'silog']}
    skipped = 0
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        image = batch['image'].to(device, non_blocking=True)
        depth_gt = batch['depth'].to(device, non_blocking=True)
        mask = batch['mask'].squeeze(1).to(device, non_blocking=True)

        pred = F.interpolate(model(image),
                             size=depth_gt.shape[-2:],
                             mode='bilinear',
                             align_corners=False).squeeze(1)
        gt = depth_gt.squeeze(1)
        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            if p.numel() < 10:
                skipped += 1
                continue
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p = p * scale.clamp(0.1, 10.0)

            p_np = p.detach().cpu().numpy().astype(np.float64)
            g_np = g.detach().cpu().numpy().astype(np.float64)
            p_np = np.clip(p_np, min_depth, max_depth)
            g_np = np.clip(g_np, min_depth, max_depth)

            m = compute_errors(g_np, p_np)
            for k, v in m.items():
                agg[k].append(v)

    out = {k: (float(np.mean(v)) if v else float('nan')) for k, v in agg.items()}
    out['skipped'] = int(skipped)
    out['num_images'] = int(len(agg['a1']))
    return out


def fmt_metrics(m):
    return (
        f"a1={m['a1']:.4f} a2={m['a2']:.4f} a3={m['a3']:.4f} "
        f"abs_rel={m['abs_rel']:.4f} sq_rel={m['sq_rel']:.4f} "
        f"rmse={m['rmse']:.4f} log_10={m['log_10']:.4f} "
        f"silog={m['silog']:.4f}"
    )


def _pack_checkpoint(epoch, stage_name, model, ema, optim, sched, scaler,
                     val_metrics, best_a1, ema_step_count, learned_sigmas):
    return {
        'epoch': epoch,
        'stage': stage_name,
        'model_state': model.state_dict(),
        'ema_state': ema.state_dict(),
        'optimizer_state': optim.state_dict(),
        'scheduler_state': sched.state_dict() if sched is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'val_metrics': val_metrics,
        'best_a1': best_a1,
        'ema_step_count': ema_step_count,
        'learned_sigmas': learned_sigmas,
        'cfg': vars(cfg),
    }


def run_stage(stage_name, model, ema, criterion, optim_kwargs, loader_kwargs,
              save_path, num_epochs, device,
              resume_path=None, latest_path=None):
    """Run one training stage with full-state resume support.

    Resume precedence:
      1. If `latest_path` exists on disk -> full resume of the same stage
         (model + EMA + optimizer + scheduler + scaler + epoch + best_a1
         + ema_step_count). `resume_path` is ignored in this case.
      2. Else if `resume_path` is provided -> load ONLY model + EMA
         (stage handoff). Optimizer / scheduler / scaler start fresh.
      3. Else -> fresh training.
    """
    print(f"\n=== Stage: {stage_name} ===")
    train_loader, val_loader = build_loaders(**loader_kwargs)
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    optim = build_optimizer(model, **optim_kwargs)
    accum = cfg.GRAD_ACCUM
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * num_epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[g['lr'] for g in optim.param_groups],
        total_steps=total_steps, pct_start=0.1,
        anneal_strategy='cos', div_factor=25.0, final_div_factor=1e3,
    )
    scaler = GradScaler()

    start_epoch = 1
    best_a1 = -float('inf')
    ema_step_count = 0

    latest_exists = latest_path and os.path.exists(latest_path)
    if latest_exists:
        try:
            ck = torch.load(latest_path, map_location=device)
            model.load_state_dict(ck['model_state'])
            ema.load_state_dict(ck['ema_state'])
            optim.load_state_dict(ck['optimizer_state'])
            if ck.get('scheduler_state') is not None:
                sched.load_state_dict(ck['scheduler_state'])
            if ck.get('scaler_state') is not None:
                scaler.load_state_dict(ck['scaler_state'])
            start_epoch = int(ck.get('epoch', 0)) + 1
            best_a1 = float(ck.get('best_a1', -float('inf')))
            ema_step_count = int(ck.get('ema_step_count', 0))
            print(f'[resume] Loaded latest checkpoint from {latest_path}')
            print(f'[resume] start_epoch={start_epoch} best_a1={best_a1:.4f} '
                  f'ema_step_count={ema_step_count}')
        except Exception as e:
            print(f'[resume] Failed to load {latest_path} ({e}); fresh start.')
            start_epoch = 1
            best_a1 = -float('inf')
            ema_step_count = 0
    elif resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        try:
            model.load_state_dict(ck['model_state'])
            if 'ema_state' in ck:
                ema.load_state_dict(ck['ema_state'])
            print(f'[handoff] Loaded model+EMA from {resume_path} '
                  f'(optimizer/scheduler start fresh)')
        except Exception as e:
            print(f'[handoff] Failed to load {resume_path} ({e}); fresh start.')

    if start_epoch > num_epochs:
        print(f'[resume] start_epoch ({start_epoch}) > num_epochs ({num_epochs}); '
              f'stage already complete. Returning.')
        return save_path

    print(f'Initial sigmas: {model.get_learned_sigmas()}')

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optim, criterion, scaler, device,
            sched, ema, accum,
            ema_step_count=ema_step_count,
            ema_warmup_steps=cfg.EMA_WARMUP_STEPS,
            ema_target_decay=cfg.EMA_DECAY,
        )
        ema_step_count = train_losses['ema_step_count']

        val_metrics = evaluate(ema, val_loader, device,
                               min_depth=cfg.MIN_DEPTH, max_depth=cfg.MAX_DEPTH)

        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        lr_now = sched.get_last_lr()[0]
        print(
            f'[{stage_name}] Epoch {epoch:02d}/{num_epochs} | '
            f'Time: {mins:02d}:{secs:02d} | LR:{lr_now:.6f} | '
            f'Loss: {train_losses["total"]:.4f} '
            f'(SI:{train_losses["si"]:.4f} '
            f'BerHu:{train_losses["berhu"]:.4f} '
            f'Log10:{train_losses["log10"]:.4f} '
            f'LogGrad:{train_losses["log_grad"]:.4f} '
            f'VNL:{train_losses["vnl"]:.4f}) | '
            f'GradNorm:{train_losses["grad_norm"]:.3f} | '
            f'EMA val: {fmt_metrics(val_metrics)}'
        )

        learned_sigmas = model.get_learned_sigmas()
        improved = val_metrics['a1'] > best_a1
        if improved:
            best_a1 = val_metrics['a1']
        if latest_path:
            torch.save(
                _pack_checkpoint(epoch, stage_name, model, ema, optim, sched,
                                 scaler, val_metrics, best_a1, ema_step_count,
                                 learned_sigmas),
                latest_path,
            )
        if improved:
            torch.save(
                _pack_checkpoint(epoch, stage_name, model, ema, optim, sched,
                                 scaler, val_metrics, best_a1, ema_step_count,
                                 learned_sigmas),
                save_path,
            )
            print(f' >>> Saved best model (a1={best_a1:.4f}) sigmas={learned_sigmas}')

    return save_path


def main():
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = build_model(device)
    ema = make_ema(model)
    criterion = DoGDepthLoss(
        alpha_si=1.0,
        alpha_berhu=0.05,
        alpha_log10=0.025,
        beta_grad=0.25,
        beta_vnl=0.1,
        gamma=0.05,
        delta=0.05,
    )

    if cfg.RESUME_PATH and os.path.exists(cfg.RESUME_PATH):
        ck = torch.load(cfg.RESUME_PATH, map_location=device)
        model.load_state_dict(ck['model_state'])
        if 'ema_state' in ck:
            ema.load_state_dict(ck['ema_state'])
        print(f'Loaded checkpoint from {cfg.RESUME_PATH}')

    stage1_save = run_stage(
        stage_name='lowres-320x416',
        model=model, ema=ema, criterion=criterion,
        optim_kwargs=dict(max_lr=cfg.MAX_LR),
        loader_kwargs=dict(
            mode='h5', h5_train_dir=cfg.H5_TRAIN_DIR, h5_val_dir=cfg.H5_VAL_DIR,
            img_size=cfg.IMG_SIZE, batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
        ),
        save_path=cfg.SAVE_PATH,
        num_epochs=cfg.NUM_EPOCHS,
        device=device,
        latest_path=cfg.LATEST_PATH,
    )

    if cfg.NUM_FT_EPOCHS > 0:
        if not os.path.exists(cfg.LATEST_PATH_FT):
            ck = torch.load(stage1_save, map_location=device)
            model.load_state_dict(ck['model_state'])
            ema.load_state_dict(ck['ema_state'])
        run_stage(
            stage_name='fixres-480x640',
            model=model, ema=ema, criterion=criterion,
            optim_kwargs=dict(max_lr=cfg.MAX_LR / 10.0),
            loader_kwargs=dict(
                mode='h5', h5_train_dir=cfg.H5_TRAIN_DIR, h5_val_dir=cfg.H5_VAL_DIR,
                img_size=cfg.FT_IMG_SIZE,
                batch_size=max(2, cfg.BATCH_SIZE // 2),
                num_workers=cfg.NUM_WORKERS,
            ),
            save_path=cfg.SAVE_PATH_FT,
            num_epochs=cfg.NUM_FT_EPOCHS,
            device=device,
            resume_path=stage1_save,
            latest_path=cfg.LATEST_PATH_FT,
        )


if __name__ == '__main__':
    main()
