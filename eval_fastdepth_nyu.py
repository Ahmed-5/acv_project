"""
Evaluate FastDepth (Hagaik92 reproduction) on the NYU Depth V2 validation split
using the SAME H5 loader and metric protocol as eval_nyu.py, so the numbers are
apples-to-apples with our DoGDepthNet checkpoints.

FastDepth design spec (from the ICRA 2019 paper and Hagaik92/FastDepth):
  * Input         : RGB in [0, 1] (no ImageNet mean/std), resized to 224x224.
  * Output        : 1-channel depth at 224x224.
  * Encoders      : MobileNetV1 (FastDepth) or MobileNetV2 (FastDepthV2),
                    with additive skip connections into an NNConv5 decoder.
  * Weights       : checkpoint dict with key 'model_state_dict'.

This script:
  1. Builds the same NYUDepthH5 val loader we use for DoGDepthNet
     (which normalises images with ImageNet mean/std at eval_img_size).
  2. Un-normalises each batch back to [0, 1] and resizes to 224x224
     before feeding FastDepth.
  3. Upsamples FastDepth output to the ground-truth resolution.
  4. Reports both median-scaled and no-scaling metrics, with optional
     H-flip TTA and Eigen / Garg crops, exactly like eval_nyu.py.

Example:
  # fully automatic — weights are downloaded on first use
  python eval_fastdepth_nyu.py --variant v1 --eigen-crop --num-workers 0

  # explicit checkpoint path still works
  python eval_fastdepth_nyu.py \
      --variant v1 \
      --checkpoint fastdepth_ref/Weights/FastDepth_L1_Best.pth \
      --h5-val-dir ./hf_dataset/data/val/ \
      --eigen-crop --num-workers 0

  # RealSense capture folder (rgb/, depth/, meta.json)
  python eval_fastdepth_nyu.py --mode realsense --rs-root depthData/runs/my_run \\
      --variant v1 --img-size 320,416 --num-workers 0
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# The Hagaik92 checkpoints pickle their original `nyudepthv2.NYUDataset` and
# argparse.Namespace — register a shim so torch.load doesn't crash before we
# can extract 'model_state_dict'.
from fastdepth_ref import nyudepthv2_shim as _nyudepthv2_shim
sys.modules.setdefault("nyudepthv2", _nyudepthv2_shim)

from data.nyu_dataset import NYUDepthH5, NYU_MEAN, NYU_STD, RealSenseAlignedPNG
from eval_nyu import compute_errors, _aggregate, _empty_agg, parse_img_size
from fastdepth_ref.models import FastDepth, FastDepthV2


FASTDEPTH_INPUT = (224, 224)

# Default local cache for auto-downloaded weights.
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fastdepth_ref", "Weights")

# Known Hagaik92/FastDepth checkpoints we can pull from GitHub on demand.
# Sizes are the ground-truth byte counts reported by the GitHub contents API.
_GH_RAW = "https://raw.githubusercontent.com/Hagaik92/FastDepth/main/Weights"
FASTDEPTH_CHECKPOINTS = {
    # canonical key          -> (variant, filename,                 url,                             size_bytes)
    "v1":                       ("v1", "FastDepth_L1_Best.pth",      f"{_GH_RAW}/FastDepth_L1_Best.pth",      22963973),
    "v1-l1":                    ("v1", "FastDepth_L1_Best.pth",      f"{_GH_RAW}/FastDepth_L1_Best.pth",      22963973),
    "v1-l1gn":                  ("v1", "FastDepth_L1GN_Best.pth",    f"{_GH_RAW}/FastDepth_L1GN_Best.pth",    22963973),
    "v2":                       ("v2", "FastDepthV2_L1_Best.pth",    f"{_GH_RAW}/FastDepthV2_L1_Best.pth",    17321291),
    "v2-l1":                    ("v2", "FastDepthV2_L1_Best.pth",    f"{_GH_RAW}/FastDepthV2_L1_Best.pth",    17321291),
    "v2-l1gn":                  ("v2", "FastDepthV2_L1GN_Best.pth",  f"{_GH_RAW}/FastDepthV2_L1GN_Best.pth",  17321291),
    "v2-rmslegn":               ("v2", "FastDepthV2_RMSLEGN_Best.pth", f"{_GH_RAW}/FastDepthV2_RMSLEGN_Best.pth", 17321291),
}


def _download_with_progress(url: str, dest: str, expected_size: int | None = None,
                            max_retries: int = 3, connect_timeout: float = 30.0) -> None:
    """Stream-download `url` to `dest` with a tqdm progress bar and size check.

    Supports resume on partial downloads by sending a Range request if a
    ``.part`` file already exists from a previous interrupted attempt.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        start_byte = os.path.getsize(part) if os.path.exists(part) else 0
        if expected_size and start_byte >= expected_size:
            # We already fully downloaded the file on a previous pass.
            break

        headers = {"User-Agent": "eval_fastdepth_nyu/1.0"}
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                content_length = resp.headers.get("Content-Length")
                remaining = int(content_length) if content_length is not None else None
                total = expected_size or (
                    (start_byte + remaining) if remaining is not None else None
                )

                mode = "ab" if start_byte > 0 and resp.status == 206 else "wb"
                if mode == "wb":
                    start_byte = 0  # server didn't honour Range

                bar = tqdm(
                    total=total,
                    initial=start_byte,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(dest)}",
                    leave=False,
                )
                try:
                    with open(part, mode) as f:
                        while True:
                            chunk = resp.read(1 << 16)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
                finally:
                    bar.close()
            break  # success
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            sleep_for = min(2 ** attempt, 10)
            print(f"  [download] attempt {attempt}/{max_retries} failed: {e}"
                  f" — retrying in {sleep_for}s")
            time.sleep(sleep_for)
    else:
        # All retries exhausted.
        raise RuntimeError(
            f"Failed to download {url} after {max_retries} attempts: {last_err}"
        )

    actual = os.path.getsize(part)
    if expected_size is not None and actual != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {url}: got {actual:,} bytes, "
            f"expected {expected_size:,}. Delete {part} and retry."
        )
    os.replace(part, dest)


def ensure_fastdepth_checkpoint(key: str,
                                dest_dir: str = WEIGHTS_DIR) -> tuple[str, str]:
    """Resolve a checkpoint key (e.g. 'v1', 'v2-l1gn') to a local path,
    downloading from GitHub if the file is missing. Returns (variant, path)."""
    key = key.lower()
    if key not in FASTDEPTH_CHECKPOINTS:
        raise ValueError(
            f"Unknown checkpoint key {key!r}. "
            f"Available: {sorted(FASTDEPTH_CHECKPOINTS.keys())}"
        )
    variant, filename, url, size = FASTDEPTH_CHECKPOINTS[key]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"[fastdepth] using cached weights: {dest}")
        return variant, dest

    if os.path.exists(dest):
        print(f"[fastdepth] existing {dest} is wrong size "
              f"({os.path.getsize(dest):,} != {size:,}); re-downloading.")
        try:
            os.remove(dest)
        except OSError:
            pass

    print(f"[fastdepth] downloading {filename} ({size/1e6:.1f} MB) from {url}")
    _download_with_progress(url, dest, expected_size=size)
    print(f"[fastdepth] saved to {dest}")
    return variant, dest


def build_fastdepth(variant: str) -> torch.nn.Module:
    v = variant.lower()
    if v in ("v1", "mobilenet", "mobilenetv1", "fastdepth"):
        return FastDepth()
    if v in ("v2", "mobilenetv2", "fastdepthv2"):
        return FastDepthV2()
    raise ValueError(f"Unknown FastDepth variant: {variant!r} (use 'v1' or 'v2')")


def load_fastdepth_checkpoint(model: torch.nn.Module, ckpt_path: str, device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    return ckpt if isinstance(ckpt, dict) else {}


def unnormalize_to_unit(image: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet mean/std normalisation applied by NYUDepthH5."""
    mean = torch.tensor(NYU_MEAN, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(NYU_STD, device=image.device).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_fastdepth(model, loader, device,
                       tta_hflip=False, eigen_crop=False, garg_crop=False,
                       min_depth=1e-3, max_depth=10.0):
    model.eval()
    agg_med = _empty_agg()
    agg_raw = _empty_agg()
    skipped = 0
    n = 0

    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].squeeze(1).to(device, non_blocking=True)

        # Convert to FastDepth input format: [0, 1] range at 224x224.
        img_fd = unnormalize_to_unit(image)
        img_fd = F.interpolate(img_fd, size=FASTDEPTH_INPUT,
                               mode="bilinear", align_corners=False)

        pred = model(img_fd)
        if tta_hflip:
            pred_flip = model(torch.flip(img_fd, dims=[-1]))
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

            scale = float(np.clip(np.median(g_np) / (np.median(p_np_raw) + 1e-8),
                                  0.1, 10.0))
            p_np_med = np.clip(p_np_raw * scale, min_depth, max_depth)
            for k, v in compute_errors(g_np, p_np_med).items():
                agg_med[k].append(v)

            n += 1

    return {
        "median_scaled": _aggregate(agg_med, skipped, n),
        "no_scaling": _aggregate(agg_raw, skipped, n),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate FastDepth on NYU Depth V2")
    p.add_argument("--variant", type=str, default="v1", choices=("v1", "v2"),
                   help="v1 = MobileNetV1 + NNConv5 (FastDepth); v2 = MobileNetV2 + NNConv5V2")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to FastDepth .pth (with model_state_dict). "
                        "Omit to auto-download the default weights for --variant "
                        "from GitHub (Hagaik92/FastDepth) into fastdepth_ref/Weights/. "
                        "If the given path does not exist, will also attempt to "
                        "auto-download if the filename matches a known release.")
    p.add_argument("--ckpt-key", type=str, default=None,
                   choices=sorted(FASTDEPTH_CHECKPOINTS.keys()),
                   help="Optional alternative to --variant for selecting a specific "
                        "FastDepth checkpoint (e.g. v1-l1gn, v2-rmslegn). Overrides --variant.")
    p.add_argument("--weights-dir", type=str, default=WEIGHTS_DIR,
                   help="Local cache directory for auto-downloaded weights.")
    p.add_argument("--mode", type=str, default="h5", choices=("h5", "realsense"),
                   help="h5 = NYUDepthH5 val; realsense = depthData run (rgb/depth PNGs).")
    p.add_argument("--h5-val-dir", type=str, default="./hf_dataset/data/val/")
    p.add_argument("--rs-root", type=str, default=None,
                   help="Run folder for --mode realsense (SAVE_DIR from collect_data.py).")
    p.add_argument("--rs-rgb-subdir", type=str, default=None,
                   help="Optional colour subfolder (default: auto-detect rgb, color, …).")
    p.add_argument("--rs-depth-subdir", type=str, default=None,
                   help="Optional depth subfolder (default: auto-detect depth, …).")
    p.add_argument("--rs-depth-scale", type=float, default=None,
                   help="Override depth_scale (m per uint16); default from meta.json.")
    p.add_argument("--rs-color-space", type=str, default=None,
                   choices=("rgb", "bgr_on_disk"),
                   help="rgb = true RGB PNGs; bgr_on_disk = legacy OpenCV BGR save.")
    p.add_argument("--img-size", type=parse_img_size, default="480,640",
                   help="Eval H,W (GT resolution). 480,640 = native NYU.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tta-hflip", action="store_true",
                   help="Average prediction with horizontally-flipped pass.")
    p.add_argument("--eigen-crop", action="store_true",
                   help="Apply Eigen NYU crop to the eval mask.")
    p.add_argument("--garg-crop", action="store_true",
                   help="Apply Garg NYU crop to the eval mask.")
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Resolve the checkpoint path — supports three modes:
    #   1. Explicit --checkpoint that exists  -> use as-is.
    #   2. --ckpt-key (or no path) -> auto-download the matching release.
    #   3. --checkpoint that doesn't exist but matches a known filename
    #      -> auto-download into the provided path.
    ckpt_key = args.ckpt_key or args.variant
    if args.checkpoint is None:
        variant, ckpt_path = ensure_fastdepth_checkpoint(ckpt_key, args.weights_dir)
        args.variant = variant
    elif not os.path.exists(args.checkpoint):
        fname = os.path.basename(args.checkpoint)
        hit_key = next((k for k, (_, n, _, _) in FASTDEPTH_CHECKPOINTS.items()
                        if n == fname), None)
        if hit_key is None:
            raise FileNotFoundError(
                f"--checkpoint {args.checkpoint} does not exist and the filename "
                f"does not match any known auto-downloadable release "
                f"({sorted({n for _, n, _, _ in FASTDEPTH_CHECKPOINTS.values()})})."
            )
        variant, ckpt_path = ensure_fastdepth_checkpoint(
            hit_key, os.path.dirname(os.path.abspath(args.checkpoint)) or args.weights_dir
        )
        args.variant = variant
    else:
        ckpt_path = args.checkpoint

    model = build_fastdepth(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FastDepth variant: {args.variant}  |  params: {n_params:,}")

    meta = load_fastdepth_checkpoint(model, ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")
    if isinstance(meta, dict):
        if "epoch" in meta:
            print(f"  checkpoint epoch: {meta['epoch']}")
        if "loss" in meta:
            print(f"  checkpoint loss : {meta['loss']}")
        if "args" in meta and hasattr(meta["args"], "criterion"):
            print(f"  training criterion: {meta['args'].criterion}")

    if args.mode == "h5":
        val_ds = NYUDepthH5(args.h5_val_dir, img_size=args.img_size, augment=False)
    else:
        if not args.rs_root:
            raise ValueError("--rs-root is required when --mode realsense")
        val_ds = RealSenseAlignedPNG(
            args.rs_root,
            args.img_size,
            depth_scale=args.rs_depth_scale,
            max_depth=args.max_depth,
            color_space=args.rs_color_space,
            rgb_subdir=args.rs_rgb_subdir,
            depth_subdir=args.rs_depth_subdir,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data mode: {args.mode} | Val samples: {len(val_ds)} | "
          f"batches: {len(val_loader)} | eval resolution: {args.img_size}")
    print(f"  tta_hflip={args.tta_hflip} eigen_crop={args.eigen_crop} "
          f"garg_crop={args.garg_crop}")

    bundle = evaluate_fastdepth(model, val_loader, device,
                                tta_hflip=args.tta_hflip,
                                eigen_crop=args.eigen_crop,
                                garg_crop=args.garg_crop,
                                min_depth=args.min_depth,
                                max_depth=args.max_depth)

    def _round(d):
        keep = ("a1", "a2", "a3", "abs_rel", "sq_rel", "rmse",
                "log_10", "rmse_log", "silog", "skipped")
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items() if k in keep}

    print(f"Metrics (median-scaled per image): {_round(bundle['median_scaled'])}")
    print(f"Metrics (no scaling)   : {_round(bundle['no_scaling'])}")

    if args.json_out:
        out = {
            "checkpoint": os.path.abspath(ckpt_path),
            "variant": args.variant,
            "params": n_params,
            "data_mode": args.mode,
            "h5_val_dir": args.h5_val_dir if args.mode == "h5" else None,
            "rs_root": os.path.abspath(args.rs_root) if args.rs_root else None,
            "img_size": list(args.img_size),
            "fastdepth_input": list(FASTDEPTH_INPUT),
            "tta_hflip": args.tta_hflip,
            "eigen_crop": args.eigen_crop,
            "garg_crop": args.garg_crop,
            **bundle,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
