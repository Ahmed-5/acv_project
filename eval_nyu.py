"""
Evaluate a DoGDepthNet checkpoint on NYU Depth V2 validation / test split.

Metrics match the training loop in train_nyu.py (median scaling per image, valid mask):
  AbsRel, RMSE, δ₁; also reports δ₂ and δ₃ (standard depth benchmarks).

Example:
  python eval_nyu.py --checkpoint checkpoints/dog_depth_nyu_v3.pth
  python eval_nyu.py --checkpoint ckpt.pth --mode mat --mat-path /path/to/nyu_depth_v2_labeled.mat
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.nyu_dataset import (
    NYUMatFile,
    NYUParquet,
    NYUDepthH5,
    NYURawScenes,
)
from models.dog_depth_net import DoGDepthNet


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
    return {
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "log_10": float(log_10),
        "rmse_log": float(rmse_log),
        "silog": float(silog),
    }


def parse_img_size(s: str):
    parts = s.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("img_size must be H,W or HxW (e.g. 320,416)")
    return int(parts[0].strip()), int(parts[1].strip())


def build_val_loader(args):
    mode = args.mode
    img_size = args.img_size
    if mode == "mat":
        if not args.mat_path:
            raise ValueError("--mat-path is required for mode=mat")
        val_ds = NYUMatFile(args.mat_path, "test", img_size, apply_crop=True)
    elif mode == "parquet":
        if not args.parquet_val:
            raise ValueError("--parquet-val is required for mode=parquet")
        val_ds = NYUParquet(args.parquet_val, img_size, augment=False)
    elif mode == "h5":
        if not args.h5_val_dir:
            raise ValueError("--h5-val-dir is required for mode=h5")
        val_ds = NYUDepthH5(args.h5_val_dir, img_size, augment=False)
    elif mode == "raw":
        if not args.raw_root:
            raise ValueError("--raw-root is required for mode=raw")
        all_ds = NYURawScenes(args.raw_root, img_size)
        n_val = 654
        g = torch.Generator().manual_seed(args.raw_split_seed)
        _, val_ds = torch.utils.data.random_split(
            all_ds, [len(all_ds) - n_val, n_val], generator=g
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _empty_agg():
    return {
        "a1": [],
        "a2": [],
        "a3": [],
        "abs_rel": [],
        "sq_rel": [],
        "rmse": [],
        "log_10": [],
        "rmse_log": [],
        "silog": [],
    }


def _aggregate(agg, skipped, num):
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
    out["skipped"] = int(skipped)
    out["num_images"] = int(num)
    out["AbsRel"] = out["abs_rel"]
    out["RMSE"] = out["rmse"]
    out["delta1"] = out["a1"]
    out["delta2"] = out["a2"]
    out["delta3"] = out["a3"]
    return out


@torch.no_grad()
def evaluate(model, loader, device, tta_hflip=False, eigen_crop=False,
             garg_crop=False, min_depth=1e-3, max_depth=10.0):
    """
    Returns two metric dicts: with and without per-image median scaling.
    Optional eval-time TTA (horizontal flip) and crops (Eigen / Garg).
    """
    model.eval()
    agg_med = _empty_agg()
    agg_raw = _empty_agg()
    skipped = 0
    n = 0

    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"].to(device)
        depth_gt = batch["depth"].to(device)
        mask = batch["mask"].squeeze(1).to(device)

        pred = model(image)
        if tta_hflip:
            pred_flip = model(torch.flip(image, dims=[-1]))
            pred = 0.5 * (pred + torch.flip(pred_flip, dims=[-1]))
        pred = F.interpolate(pred, size=depth_gt.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze(1)
        gt = depth_gt.squeeze(1)

        for b in range(pred.shape[0]):
            valid = mask[b]
            if eigen_crop or garg_crop:
                h, w = valid.shape
                em = torch.zeros_like(valid)
                if garg_crop:
                    t0 = int(round(0.40810811 * h)); t1 = int(round(0.99189189 * h))
                    l0 = int(round(0.03594771 * w)); l1 = int(round(0.96405229 * w))
                else:
                    t0 = int(round(45 * h / 480)); t1 = int(round(471 * h / 480))
                    l0 = int(round(41 * w / 640)); l1 = int(round(601 * w / 640))
                em[t0:t1, l0:l1] = True
                valid = valid & em

            p, g = pred[b][valid], gt[b][valid]
            if p.numel() < 10:
                skipped += 1
                continue

            p_np_raw = p.detach().cpu().numpy().astype(np.float64)
            g_np = g.detach().cpu().numpy().astype(np.float64)
            p_np_raw = np.clip(p_np_raw, min_depth, max_depth)
            g_np = np.clip(g_np, min_depth, max_depth)
            for k, v in compute_errors(g_np, p_np_raw).items():
                agg_raw[k].append(v)

            scale = float(np.clip(np.median(g_np) / (np.median(p_np_raw) + 1e-8), 0.1, 10.0))
            p_np_med = np.clip(p_np_raw * scale, min_depth, max_depth)
            for k, v in compute_errors(g_np, p_np_med).items():
                agg_med[k].append(v)

            n += 1

    return {
        "median_scaled": _aggregate(agg_med, skipped, n),
        "no_scaling": _aggregate(agg_raw, skipped, n),
    }


def load_model(device, checkpoint_path, sigma_pairs, branch_ch,
               pretrained_backbone, n_bins=64, min_depth=1e-3, max_depth=10.0,
               use_ema=False):
    model = DoGDepthNet(
        sigma_pairs=sigma_pairs,
        branch_ch=branch_ch,
        n_bins=n_bins,
        min_depth=min_depth,
        max_depth=max_depth,
        pretrained=pretrained_backbone,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if use_ema and "ema_state" in ckpt:
            state = ckpt["ema_state"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    return model, ckpt if isinstance(ckpt, dict) else {}


def main():
    p = argparse.ArgumentParser(description="Evaluate DoGDepthNet on NYU Depth V2")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (with model_state or raw state_dict)")
    p.add_argument(
        "--mode",
        type=str,
        default="h5",
        choices=("h5", "mat", "parquet", "raw"),
        help="Dataset layout (must match training preprocessing)",
    )
    p.add_argument("--img-size", type=parse_img_size, default="320,416", help="H,W or HxW (default matches train_nyu)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    p.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Build encoder without ImageNet weights before loading checkpoint (saves download if weights are fully in ckpt)",
    )
    p.add_argument("--sigma-pairs", type=str, default="0.5,1.0;1.0,2.0;2.0,4.0", help="Semicolon-separated a,b pairs")
    p.add_argument("--branch-ch", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--use-ema", action="store_true",
                   help="Load ema_state from checkpoint instead of model_state.")
    p.add_argument("--tta-hflip", action="store_true",
                   help="Average prediction with horizontally-flipped pass.")
    p.add_argument("--eigen-crop", action="store_true",
                   help="Apply Eigen NYU crop to the eval mask.")
    p.add_argument("--garg-crop", action="store_true",
                   help="Apply Garg NYU crop to the eval mask.")
    p.add_argument("--h5-val-dir", type=str, default="./hf_dataset/data/val/")
    # mat
    p.add_argument("--mat-path", type=str, default=None)
    # parquet
    p.add_argument("--parquet-val", type=str, nargs="+", default=None, help="One or more val .parquet shards")
    # raw
    p.add_argument("--raw-root", type=str, default=None)
    p.add_argument("--raw-split-seed", type=int, default=42, help="Generator seed for raw mode val subset")
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write metrics as JSON")

    args = p.parse_args()

    def parse_sigma_pairs(s: str):
        pairs = []
        for chunk in s.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            a, b = chunk.split(",")
            pairs.append((float(a.strip()), float(b.strip())))
        return pairs

    args.sigma_pairs = parse_sigma_pairs(args.sigma_pairs)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    val_loader = build_val_loader(args)
    print(f"Val batches: {len(val_loader)}")

    model, ckpt_meta = load_model(
        device,
        args.checkpoint,
        args.sigma_pairs,
        args.branch_ch,
        pretrained_backbone=not args.no_pretrained_backbone,
        n_bins=args.n_bins,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_ema=args.use_ema,
    )
    print(f"Loaded checkpoint: {args.checkpoint}")
    if ckpt_meta.get("epoch") is not None:
        print(f"  checkpoint epoch: {ckpt_meta['epoch']}")
    if ckpt_meta.get("learned_sigmas") is not None:
        print(f"  stored learned_sigmas: {ckpt_meta['learned_sigmas']}")
    print(f"  current learned_sigmas: {model.get_learned_sigmas()}")
    print(f"  use_ema={args.use_ema} tta_hflip={args.tta_hflip} "
          f"eigen_crop={args.eigen_crop} garg_crop={args.garg_crop}")

    bundle = evaluate(model, val_loader, device,
                      tta_hflip=args.tta_hflip,
                      eigen_crop=args.eigen_crop,
                      garg_crop=args.garg_crop,
                      min_depth=args.min_depth,
                      max_depth=args.max_depth)

    def _round(d):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items()
                if k in ("a1", "a2", "a3", "abs_rel", "sq_rel", "rmse",
                         "log_10", "rmse_log", "silog", "skipped")}

    print(f"Metrics (median-scaled): {_round(bundle['median_scaled'])}")
    print(f"Metrics (no scaling)   : {_round(bundle['no_scaling'])}")

    if args.json_out:
        out = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "mode": args.mode,
            "img_size": list(args.img_size),
            "use_ema": args.use_ema,
            "tta_hflip": args.tta_hflip,
            "eigen_crop": args.eigen_crop,
            "garg_crop": args.garg_crop,
            **bundle,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
