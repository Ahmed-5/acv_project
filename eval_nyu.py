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


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Per-sample median scaling (same as train_nyu.evaluate), then aggregate means.
    δ_k: fraction of valid pixels with max(pred/gt, gt/pred) < 1.25^k.
    """
    model.eval()
    agg = {
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
    skipped = 0
    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"].to(device)
        depth_gt = batch["depth"].to(device)
        mask = batch["mask"].squeeze(1).to(device)
        pred = F.interpolate(
            model(image), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        gt = depth_gt.squeeze(1)
        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            if p.numel() < 10:
                skipped += 1
                continue
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p = p * scale
            p_np = p.detach().cpu().numpy().astype(np.float64)
            g_np = g.detach().cpu().numpy().astype(np.float64)
            p_np = np.clip(p_np, 1e-3, 10.0)
            g_np = np.clip(g_np, 1e-3, 10.0)

            m = compute_errors(g_np, p_np)
            for k, v in m.items():
                agg[k].append(v)

    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
    out["skipped"] = int(skipped)
    out["num_images"] = int(len(agg["a1"]))
    # Backward-compatible aliases used by previous output.
    out["AbsRel"] = out["abs_rel"]
    out["RMSE"] = out["rmse"]
    out["delta1"] = out["a1"]
    out["delta2"] = out["a2"]
    out["delta3"] = out["a3"]
    return out


def load_model(device, checkpoint_path, sigma_pairs, branch_ch, pretrained_backbone):
    model = DoGDepthNet(
        sigma_pairs=sigma_pairs,
        branch_ch=branch_ch,
        pretrained=pretrained_backbone,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
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
    # h5
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
    )
    print(f"Loaded checkpoint: {args.checkpoint}")
    if ckpt_meta.get("epoch") is not None:
        print(f"  checkpoint epoch: {ckpt_meta['epoch']}")
    if ckpt_meta.get("learned_sigmas") is not None:
        print(f"  stored learned_sigmas: {ckpt_meta['learned_sigmas']}")
    print(f"  current learned_sigmas: {model.get_learned_sigmas()}")

    metrics = evaluate(model, val_loader, device)
    full_metrics = {
        "a1": round(metrics["a1"], 4),
        "a2": round(metrics["a2"], 4),
        "a3": round(metrics["a3"], 4),
        "abs_rel": round(metrics["abs_rel"], 4),
        "sq_rel": round(metrics["sq_rel"], 4),
        "rmse": round(metrics["rmse"], 4),
        "log_10": round(metrics["log_10"], 4),
        "rmse_log": round(metrics["rmse_log"], 4),
        "silog": round(metrics["silog"], 4),
        "skipped": metrics["skipped"],
    }
    print(f"Metrics: {full_metrics}")
    print(
        f"AbsRel: {metrics['AbsRel']:.4f} | RMSE: {metrics['RMSE']:.4f} | "
        f"d1: {metrics['delta1']:.4f} | d2: {metrics['delta2']:.4f} | d3: {metrics['delta3']:.4f} | "
        f"n={metrics['num_images']}"
    )

    if args.json_out:
        out = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "mode": args.mode,
            "img_size": list(args.img_size),
            **metrics,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
