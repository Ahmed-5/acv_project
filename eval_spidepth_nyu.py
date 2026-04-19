"""
Evaluate SPIdepth (SQLdepth) on NYU Depth V2.

Uses SPIdepth/finetune DepthDataLoader (official test list) and the same metrics as
SPIdepth/finetune/evaluate_metric_depth.py.

Why two weight sources?
  - encoder.pth / depth.pth under exps/ (e.g. inc_kitti_exps_v13) are KITTI-pretrained
    weights shipped with SPIdepth, not NYU-finetuned.
  - A --checkpoint .pt from NYU finetuning holds weights after training on NYU; use that
    for results that match a finetuned run.

  --exps-only: evaluate using only KITTI exps (no finetune file; no --checkpoint).
  --init-from-exps plus --checkpoint: load exps in __init__, then overwrite with the
    finetune checkpoint (typical for evaluating a NYU-trained model whose training started
    from those exps).

Examples:
  python eval_spidepth_nyu.py --model-opt SPIdepth/conf/cvnXt.txt \\
    --checkpoint nyu_finetune.pt --nyu-root path/to/nyu_sync

  python eval_spidepth_nyu.py --model-opt SPIdepth/conf/cvnXt.txt \\
    --exps-only --nyu-root path/to/nyu_sync

  # Val split as .h5 under hf_dataset/data/val/official (see train_nyu.py)
  python eval_spidepth_nyu.py --data-source h5 --model-opt SPIdepth/conf/cvnXt.txt \\
    --exps-only

  python eval_spidepth_nyu.py my_eval.txt SPIdepth/conf/cvnXt.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _convert_arg_line_to_args(arg_line):
    stripped = arg_line.split("#", 1)[0].strip()
    if not stripped:
        return
    for arg in stripped.split():
        if arg.strip():
            yield str(arg)


def _resolve_config_path(name: str, repo_root: Path) -> str:
    p = Path(name).expanduser()
    if p.is_file():
        return str(p.resolve())
    for base in (Path.cwd(), repo_root):
        cand = (base / name).resolve()
        if cand.is_file():
            return str(cand)
    return name


def _resolve_dir(name: str, repo_root: Path) -> str:
    p = Path(name).expanduser()
    if p.is_dir():
        return str(p.resolve())
    for base in (Path.cwd(), repo_root):
        cand = (base / name).resolve()
        if cand.is_dir():
            return str(cand)
    return name


def _parse_h5_img_size(s: str):
    parts = s.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("h5-img-size must be H,W or HxW (e.g. 320,416)")
    return int(parts[0].strip()), int(parts[1].strip())


def _eigen_mask_scaled(h: int, w: int):
    """Eigen NYU crop mapped from 480×640 to (h, w)."""
    t0, t1 = int(round(45 * h / 480)), int(round(471 * h / 480))
    l0, l1 = int(round(41 * w / 640)), int(round(601 * w / 640))
    m = np.zeros((h, w), dtype=bool)
    m[t0:t1, l0:l1] = True
    return m


def _garg_mask_scaled(h: int, w: int):
    t0, t1 = int(round(0.40810811 * h)), int(round(0.99189189 * h))
    l0, l1 = int(round(0.03594771 * w)), int(round(0.96405229 * w))
    m = np.zeros((h, w), dtype=bool)
    m[t0:t1, l0:l1] = True
    return m


def _setup_spidepth_paths(spidepth_root: Path) -> None:
    root = str(spidepth_root.resolve())
    finetune = str((spidepth_root / "finetune").resolve())
    for p in (root, finetune):
        if p not in sys.path:
            sys.path.insert(0, p)


def _compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100
    log_10 = np.abs(np.log10(gt) - np.log10(pred)).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
    )


def _parse_args():
    repo_root = Path(__file__).resolve().parent
    default_spidepth = repo_root / "SPIdepth"
    default_list = (
        default_spidepth
        / "finetune"
        / "train_test_inputs"
        / "nyudepthv2_test_files_with_gt.txt"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate SPIdepth (SQLdepth) on NYU Depth V2",
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = _convert_arg_line_to_args

    parser.add_argument(
        "--spidepth-root",
        type=str,
        default=str(default_spidepth),
        help="SPIdepth clone root (default: ./SPIdepth next to this script)",
    )
    parser.add_argument(
        "--model-opt",
        type=str,
        default=None,
        help="MonodepthOptions arg file (e.g. SPIdepth/conf/cvnXt.txt)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        choices=("list", "h5"),
        default="list",
        help="list: SPIdepth sync layout + filenames file. h5: NYUDepthH5 shards (e.g. hf_dataset/.../official).",
    )
    parser.add_argument(
        "--h5-val-dir",
        type=str,
        default=None,
        help="Directory with val .h5 files (default: ./hf_dataset/data/val/official next to repo root)",
    )
    parser.add_argument(
        "--h5-img-size",
        type=_parse_h5_img_size,
        default="480,640",
        help=(
            "H,W resize for NYUDepthH5. SPIdepth/cvnXt depth decoder needs enough patch tokens "
            "(patch_size=32); 320x416 often yields too few and crashes (bins_regressor 2048 vs 960). "
            "Use 480,640 to match SPIdepth NYU finetune/eval, or larger multiples of 32."
        ),
    )
    parser.add_argument("--h5-batch-size", type=int, default=1)
    parser.add_argument(
        "--nyu-root",
        type=str,
        default=None,
        help="Dataset root for SPIdepth list mode (rgb/depth relative paths in filenames file)",
    )
    parser.add_argument(
        "--filenames-eval",
        type=str,
        default=str(default_list),
        help="SPIdepth-format test list (rgb relpath, depth relpath, focal)",
    )
    parser.add_argument("--min-depth-eval", type=float, default=1e-3)
    parser.add_argument("--max-depth-eval", type=float, default=10.0)
    parser.add_argument(
        "--eigen-crop",
        dest="eigen_crop",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-eigen-crop", dest="eigen_crop", action="store_false")
    parser.add_argument("--garg-crop", dest="garg_crop", action="store_true", default=False)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument(
        "--median-align",
        action="store_true",
        help="Per-image median scaling (like repo eval_nyu.py); off by default",
    )
    parser.add_argument(
        "--init-from-exps",
        action="store_true",
        help="Keep --load_pretrained_model from model-opt; load encoder.pth/depth.pth before finetune ckpt",
    )
    parser.add_argument(
        "--exps-only",
        action="store_true",
        help="Use only KITTI exps (encoder.pth/depth.pth); do not load --checkpoint. Conflicts with --checkpoint.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)

    raw = sys.argv[1:]
    eval_file_used = None
    two_file = (
        len(raw) >= 2
        and not raw[0].startswith("-")
        and not raw[1].startswith("-")
        and not raw[0].startswith("@")
    )
    if two_file:
        eval_path = _resolve_config_path(raw[0], repo_root)
        model_path = _resolve_config_path(raw[1], repo_root)
        eval_file_used = eval_path
        if not Path(eval_path).is_file():
            parser.error(
                f"eval args file not found: {raw[0]!r} (tried cwd and {repo_root}). "
                "It should list --checkpoint and --nyu-root."
            )
        if Path(eval_path).stat().st_size == 0:
            parser.error(
                f"eval args file is empty: {eval_path!r}. "
                "Add lines, e.g.:\n"
                "  --checkpoint path/to/finetune_best.pt\n"
                "  --nyu-root path/to/nyu_sync\n"
                "Or KITTI-exps-only (no finetune .pt):\n"
                "  --exps-only\n"
                "  --nyu-root path/to/nyu_sync\n"
                "Or H5 val shards (no nyu-root):\n"
                "  --data-source h5 --h5-val-dir hf_dataset/data/val/official --exps-only\n"
                "Optional after the two files: --init-from-exps --json-out out.json"
            )
        if not Path(model_path).is_file():
            parser.error(f"model-opt file not found: {raw[1]!r}")
        args = parser.parse_args(["@" + eval_path, *raw[2:]])
        args.model_opt = model_path
    else:
        args = parser.parse_args()

    if not args.model_opt:
        parser.error(
            "Pass --model-opt PATH or two positionals: eval_spidepth_nyu.py EVAL.txt MODEL.txt"
        )
    mo = args.model_opt
    if not Path(mo).is_file():
        resolved = _resolve_config_path(mo, repo_root)
        if Path(resolved).is_file():
            args.model_opt = resolved
        else:
            parser.error(f"--model-opt not found: {mo!r}")

    if args.exps_only and args.checkpoint:
        parser.error("--exps-only cannot be used with --checkpoint (finetune weights are only in the .pt file).")

    if not args.exps_only and not args.checkpoint:
        extra = ""
        if eval_file_used:
            extra = f" Missing in {eval_file_used!r} (or pass --checkpoint on the command line)."
        parser.error(
            "--checkpoint is required unless you pass --exps-only (KITTI exps only, no NYU finetune)."
            + extra
        )

    if args.data_source == "list" and not args.nyu_root:
        extra = ""
        if eval_file_used:
            extra = f" Missing in {eval_file_used!r} (or pass --nyu-root on the command line)."
        parser.error(
            "--nyu-root is required for --data-source list. For H5 val shards use --data-source h5 "
            "and --h5-val-dir." + extra
        )

    if args.data_source == "h5":
        if args.h5_val_dir is None:
            args.h5_val_dir = str(repo_root / "hf_dataset" / "data" / "val" / "official")
        args.h5_val_dir = _resolve_dir(args.h5_val_dir, repo_root)
        if not Path(args.h5_val_dir).is_dir():
            parser.error(f"--h5-val-dir is not a directory: {args.h5_val_dir!r}")
    else:
        args.h5_val_dir = args.h5_val_dir or ""

    alt_list = Path(args.spidepth_root) / "finetune" / "train_test_inputs" / "nyudepthv2_test_files_with_gt.txt"
    if not Path(args.filenames_eval).is_file() and alt_list.is_file():
        args.filenames_eval = str(alt_list)

    return args


def main():
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    args = _parse_args()
    spidepth_root = Path(args.spidepth_root)
    if not spidepth_root.is_dir():
        raise SystemExit(f"SPIdepth root not found: {spidepth_root}")

    _setup_spidepth_paths(spidepth_root)

    from SQLdepth import MonodepthOptions, SQLdepth
    from dataloader import DepthDataLoader
    import model_io
    from utils import RunningAverageDict

    mo = args.model_opt
    opt_argv = [mo] if mo.startswith("@") else ["@" + mo]
    opt = MonodepthOptions().parser.parse_args(opt_argv)

    if args.exps_only:
        opt.load_pretrained_model = True
    elif not args.init_from_exps:
        opt.load_pretrained_model = False

    if args.device == "auto":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if args.data_source == "h5":
        from data.nyu_dataset import NYUDepthH5
        from torch.utils.data import DataLoader

        h5_ds = NYUDepthH5(args.h5_val_dir, img_size=args.h5_img_size, augment=False)
        test_loader = DataLoader(
            h5_ds,
            batch_size=args.h5_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        use_h5 = True
    else:
        dl_args = SimpleNamespace(
            dataset="nyu",
            data_path=args.nyu_root,
            gt_path=args.nyu_root,
            filenames_file=args.filenames_eval,
            data_path_eval=args.nyu_root,
            gt_path_eval=args.nyu_root,
            filenames_file_eval=os.path.abspath(args.filenames_eval),
            input_height=416,
            input_width=544,
            max_depth=args.max_depth_eval,
            min_depth=args.min_depth_eval,
            min_depth_eval=args.min_depth_eval,
            max_depth_eval=args.max_depth_eval,
            do_kb_crop=False,
            eigen_crop=args.eigen_crop,
            garg_crop=args.garg_crop,
            distributed=False,
            do_random_rotate=False,
            degree=2.5,
            use_right=False,
        )
        test_loader = DepthDataLoader(dl_args, "online_eval").data
        use_h5 = False

    model = SQLdepth(opt).to(device)
    if args.exps_only:
        print("Using KITTI exps only (encoder.pth / depth.pth); no NYU finetune checkpoint loaded.")
    else:
        model = model_io.load_checkpoint(args.checkpoint, model)[0]
    model.eval()

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    metrics = RunningAverageDict()
    n_skip = 0

    desc = "NYU H5 (SPIdepth)" if use_h5 else "NYU (SPIdepth)"
    for batch in tqdm(test_loader, desc=desc):
        if not use_h5 and not batch.get("has_valid_depth", True):
            n_skip += 1
            continue

        image = batch["image"].to(device)
        if use_h5:
            image_in = image * std_im + mean_im
        else:
            image_in = image

        with torch.no_grad():
            pred = model(image_in)
            if args.tta_hflip:
                pred_flip = model(torch.flip(image_in, dims=[-1]))
                pred = 0.5 * (pred + torch.flip(pred_flip, dims=[-1]))

        depth_t = batch["depth"]
        bs = depth_t.shape[0]
        mask_t = batch.get("mask")

        for bi in range(bs):
            gt = depth_t[bi].squeeze().cpu().numpy().astype(np.float64)
            pred_b = pred[bi : bi + 1]
            pred_b = F.interpolate(pred_b, size=gt.shape, mode="bilinear", align_corners=True)
            pred_np = pred_b.squeeze().cpu().numpy().astype(np.float64)

            pred_np = np.clip(pred_np, args.min_depth_eval, args.max_depth_eval)
            pred_np[np.isnan(pred_np)] = args.min_depth_eval
            pred_np[np.isinf(pred_np)] = args.max_depth_eval

            valid = np.logical_and(gt > args.min_depth_eval, gt < args.max_depth_eval)
            if use_h5 and mask_t is not None:
                m = mask_t[bi].squeeze().cpu().numpy().astype(bool)
                valid = np.logical_and(valid, m)

            if args.garg_crop:
                eval_mask = _garg_mask_scaled(gt.shape[0], gt.shape[1])
                valid = np.logical_and(valid, eval_mask)
            elif args.eigen_crop:
                if use_h5:
                    eval_mask = _eigen_mask_scaled(gt.shape[0], gt.shape[1])
                else:
                    eval_mask = np.zeros(valid.shape, dtype=bool)
                    eval_mask[45:471, 41:601] = True
                valid = np.logical_and(valid, eval_mask)

            if valid.sum() < 10:
                n_skip += 1
                continue

            p = pred_np[valid]
            g = gt[valid]
            if args.median_align:
                scale = float(np.clip(np.median(g) / (np.median(p) + 1e-8), 0.1, 10.0))
                p = p * scale

            metrics.update(_compute_errors(g, p))

    out = {k: round(float(v), 4) for k, v in metrics.get_value().items()}
    out["skipped"] = n_skip
    print(f"Metrics: {out}")
    if n_skip:
        print(f"Skipped {n_skip} samples (missing/invalid GT or too few valid pixels)")

    if args.json_out:
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
            "exps_only": args.exps_only,
            "data_source": args.data_source,
            "nyu_root": os.path.abspath(args.nyu_root) if args.nyu_root else None,
            "h5_val_dir": os.path.abspath(args.h5_val_dir) if args.data_source == "h5" else None,
            "h5_img_size": list(args.h5_img_size) if args.data_source == "h5" else None,
            "init_from_exps": args.init_from_exps,
            "eigen_crop": args.eigen_crop,
            "garg_crop": args.garg_crop,
            "median_align": args.median_align,
            "tta_hflip": args.tta_hflip,
            **out,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
