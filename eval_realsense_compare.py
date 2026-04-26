"""
Evaluate DoGDepthNet and FastDepth on the same RealSense capture (rgb/ + depth/ + meta.json).

Uses the same metrics, masking, and optional Eigen / Garg crops as ``eval_nyu.py`` /
``eval_fastdepth_nyu.py``.

Example:
  python eval_realsense_compare.py \\
      --rs-root depthData/runs/indoor_april \\
      --dog-checkpoint checkpoints/dog_depth_nyu_v3.pth \\
      --json-out results/realsense_compare.json
"""
import argparse
import json
import os

import torch

from eval_nyu import (
    parse_img_size,
    evaluate,
    load_model,
    build_val_loader,
)
from eval_fastdepth_nyu import (
    FASTDEPTH_CHECKPOINTS,
    WEIGHTS_DIR,
    build_fastdepth,
    ensure_fastdepth_checkpoint,
    load_fastdepth_checkpoint,
    evaluate_fastdepth,
)


def _parse_sigma_pairs(s: str):
    pairs = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pairs.append((float(a.strip()), float(b.strip())))
    return pairs


def _round_metrics(d):
    keep = (
        "a1", "a2", "a3", "abs_rel", "sq_rel", "rmse",
        "log_10", "rmse_log", "silog", "skipped", "num_images",
    )
    return {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in d.items() if k in keep
    }


def main():
    p = argparse.ArgumentParser(
        description="Compare DoGDepthNet and FastDepth on a RealSense PNG run."
    )
    p.add_argument("--rs-root", type=str, required=True,
                   help="Run folder with rgb/, depth/, and meta.json (depth_scale).")
    p.add_argument("--dog-checkpoint", type=str, required=True)
    p.add_argument("--img-size", type=parse_img_size, default="320,416")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--no-pretrained-backbone", action="store_true")
    p.add_argument("--sigma-pairs", type=str, default="0.5,1.0;1.0,2.0;2.0,4.0")
    p.add_argument("--branch-ch", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--use-ema", action="store_true")

    p.add_argument("--rs-depth-scale", type=float, default=None)
    p.add_argument("--rs-rgb-subdir", type=str, default=None)
    p.add_argument("--rs-depth-subdir", type=str, default=None)
    p.add_argument("--rs-color-space", type=str, default=None,
                   choices=("rgb", "bgr_on_disk"))

    p.add_argument("--tta-hflip", action="store_true")
    p.add_argument("--eigen-crop", action="store_true")
    p.add_argument("--garg-crop", action="store_true")

    p.add_argument("--fastdepth-variant", type=str, default="v1", choices=("v1", "v2"))
    p.add_argument("--fastdepth-checkpoint", type=str, default=None)
    p.add_argument("--fastdepth-ckpt-key", type=str, default=None,
                   choices=sorted(FASTDEPTH_CHECKPOINTS.keys()))
    p.add_argument("--fastdepth-weights-dir", type=str, default=WEIGHTS_DIR)

    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    sigma_pairs = _parse_sigma_pairs(args.sigma_pairs)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    loader_args = argparse.Namespace(
        mode="realsense",
        img_size=args.img_size,
        mat_path=None,
        parquet_val=None,
        h5_val_dir="",
        raw_root=None,
        raw_split_seed=42,
        rs_root=args.rs_root,
        rs_depth_scale=args.rs_depth_scale,
        rs_color_space=args.rs_color_space,
        rs_rgb_subdir=args.rs_rgb_subdir,
        rs_depth_subdir=args.rs_depth_subdir,
        max_depth=args.max_depth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_val_loader(loader_args)
    print(
        f"RealSense: {os.path.abspath(args.rs_root)} | "
        f"samples: {len(val_loader.dataset)} | batches: {len(val_loader)}"
    )

    dog, ckpt_meta = load_model(
        device,
        args.dog_checkpoint,
        sigma_pairs,
        args.branch_ch,
        pretrained_backbone=not args.no_pretrained_backbone,
        n_bins=args.n_bins,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_ema=args.use_ema,
    )
    print(f"DoGDepthNet: {args.dog_checkpoint}")
    dog_bundle = evaluate(
        dog, val_loader, device,
        tta_hflip=args.tta_hflip,
        eigen_crop=args.eigen_crop,
        garg_crop=args.garg_crop,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    print(f"  DoG (median-scaled): {_round_metrics(dog_bundle['median_scaled'])}")
    print(f"  DoG (no scaling)   : {_round_metrics(dog_bundle['no_scaling'])}")

    ckpt_key = args.fastdepth_ckpt_key or args.fastdepth_variant
    if args.fastdepth_checkpoint is None:
        fd_variant, fd_path = ensure_fastdepth_checkpoint(
            ckpt_key, args.fastdepth_weights_dir)
    elif not os.path.exists(args.fastdepth_checkpoint):
        fname = os.path.basename(args.fastdepth_checkpoint)
        hit_key = next(
            (k for k, (_, n, _, _) in FASTDEPTH_CHECKPOINTS.items() if n == fname),
            None,
        )
        if hit_key is None:
            raise FileNotFoundError(
                f"--fastdepth-checkpoint {args.fastdepth_checkpoint!r} missing and "
                f"filename not a known release."
            )
        fd_variant, fd_path = ensure_fastdepth_checkpoint(
            hit_key,
            os.path.dirname(os.path.abspath(args.fastdepth_checkpoint))
            or args.fastdepth_weights_dir,
        )
    else:
        fd_path = args.fastdepth_checkpoint
        fd_variant = args.fastdepth_variant

    fd_model = build_fastdepth(fd_variant).to(device)
    load_fastdepth_checkpoint(fd_model, fd_path, device)
    print(f"FastDepth: {fd_path} (variant {fd_variant})")
    fd_bundle = evaluate_fastdepth(
        fd_model, val_loader, device,
        tta_hflip=args.tta_hflip,
        eigen_crop=args.eigen_crop,
        garg_crop=args.garg_crop,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    print(f"  FastDepth (median-scaled): {_round_metrics(fd_bundle['median_scaled'])}")
    print(f"  FastDepth (no scaling)   : {_round_metrics(fd_bundle['no_scaling'])}")

    if args.json_out:
        meta_subset = {}
        for k in ("epoch", "learned_sigmas"):
            if ckpt_meta.get(k) is not None:
                meta_subset[k] = ckpt_meta[k]
        out = {
            "rs_root": os.path.abspath(args.rs_root),
            "img_size": list(args.img_size),
            "n_samples": len(val_loader.dataset),
            "dog_checkpoint": os.path.abspath(args.dog_checkpoint),
            "dog_meta": meta_subset,
            "dog_depthnet": dog_bundle,
            "fastdepth_checkpoint": os.path.abspath(fd_path),
            "fastdepth_variant": fd_variant,
            "fastdepth": fd_bundle,
            "tta_hflip": args.tta_hflip,
            "eigen_crop": args.eigen_crop,
            "garg_crop": args.garg_crop,
        }
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
