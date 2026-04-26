"""
Run FastDepth (Hagaik92 reproduction) on arbitrary RGB images and save the
predicted depth as both a raw ``.npy`` array and a colourised ``.png`` preview.
Weights are auto-downloaded from GitHub on first use (same machinery as
``eval_fastdepth_nyu.py``).

Accepts any mix of:
  * a single image file   (``--input path/to/img.jpg``)
  * a directory of images (``--input path/to/folder``)
  * a glob pattern        (``--input "path/to/*.jpg"``)
  * multiple ``--input`` args

FastDepth design spec (matches ICRA 2019 paper and the eval script):
  * Input  : RGB in [0, 1] at 224x224 (no ImageNet mean/std).
  * Output : 1-channel depth at 224x224 in metres (NYU scale).

Examples:
  # single image, auto-download v1 weights, write outputs next to the input
  python infer_fastdepth.py --variant v1 --input demo.jpg

  # whole folder to ./preds/, use MobileNetV2 variant, write a side-by-side
  # RGB|depth comparison PNG as well
  python infer_fastdepth.py \
      --variant v2 --input ./my_images --output-dir ./preds --side-by-side

  # specific checkpoint key (e.g. v2 trained with RMSLE + group norm)
  python infer_fastdepth.py --ckpt-key v2-rmslegn --input demo.jpg

  # median-scale predictions to metric GT (same rule as eval_fastdepth_nyu)
  python infer_fastdepth.py --variant v1 --input depthData/runs/merged/rgb \\
      --gt-depth-dir depthData/runs/merged/depth --gt-depth-scale 0.001 \\
      --native-resolution --output-dir preds/ --save-raw-pred
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Re-use the download + loading pipeline from the eval script. This also
# triggers the ``nyudepthv2`` shim registration needed by torch.load.
from data.nyu_dataset import REALSENSE_DEFAULT_DEPTH_SCALE
from eval_fastdepth_nyu import (
    FASTDEPTH_CHECKPOINTS,
    FASTDEPTH_INPUT,
    WEIGHTS_DIR,
    build_fastdepth,
    ensure_fastdepth_checkpoint,
    load_fastdepth_checkpoint,
)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _gather_inputs(patterns: List[str]) -> List[str]:
    """Expand --input args into a concrete list of image file paths."""
    out: List[str] = []
    for pat in patterns:
        if os.path.isdir(pat):
            for root, _, files in os.walk(pat):
                for f in files:
                    if f.lower().endswith(IMG_EXTS):
                        out.append(os.path.join(root, f))
        elif any(ch in pat for ch in "*?[]"):
            out.extend(glob.glob(pat, recursive=True))
        elif os.path.isfile(pat):
            out.append(pat)
        else:
            print(f"  [warn] no match for --input {pat!r}")
    # Keep only real files with image extensions, deduped, sorted.
    filtered = sorted({p for p in out
                       if os.path.isfile(p) and p.lower().endswith(IMG_EXTS)})
    return filtered


def _load_image_tensor(path: str, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load an image as a normalised (1, 3, 224, 224) tensor in [0, 1].
    Returns (tensor, original_hw)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size  # PIL reports (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    t = F.interpolate(t, size=FASTDEPTH_INPUT, mode="bilinear", align_corners=False)
    return t, (h, w)


def _colourise(depth: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Map a (H, W) depth array to a uint8 RGB image using a matplotlib colormap,
    or a plain grayscale ramp as a fallback if matplotlib is unavailable."""
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if finite.any():
        lo = float(np.percentile(d[finite], 2))
        hi = float(np.percentile(d[finite], 98))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        d = (d - lo) / (hi - lo)
    else:
        d = np.zeros_like(d)
    d = np.clip(d, 0.0, 1.0)
    try:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgba = cmap(d)  # (H, W, 4) float in [0, 1]
        return (rgba[..., :3] * 255.0).astype(np.uint8)
    except Exception:
        g = (d * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


def _default_output_path(src: str, output_dir: str | None, suffix: str, ext: str) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(src))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}{suffix}.{ext}")


def _find_gt_depth_path(rgb_path: str, gt_dir: str) -> str | None:
    base = os.path.basename(rgb_path)
    stem, _ = os.path.splitext(base)
    for cand in (base, f"{stem}.png", f"{stem}.PNG"):
        p = os.path.join(gt_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def _load_gt_depth_metres(path: str, uint16_scale: float) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * float(uint16_scale)
    return arr.astype(np.float32)


def _resize_depth_nearest(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(h, w), mode="nearest").squeeze().numpy()


def _median_scale_factor(pred: np.ndarray, gt_m: np.ndarray,
                         min_d: float, max_d: float) -> float:
    """Same ratio as eval_fastdepth_nyu / eval_nyu (per-image median alignment)."""
    valid = (gt_m > min_d) & (gt_m <= max_d) & (pred > min_d) & np.isfinite(pred) & np.isfinite(gt_m)
    if valid.sum() < 10:
        return 1.0
    g = gt_m[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    p = np.clip(p, min_d, max_d)
    g = np.clip(g, min_d, max_d)
    scale = float(np.median(g) / (np.median(p) + 1e-8))
    return float(np.clip(scale, 0.1, 10.0))


@torch.no_grad()
def run_inference(model: torch.nn.Module, image_paths: List[str], *,
                  device: torch.device,
                  output_dir: str | None,
                  save_npy: bool,
                  save_png: bool,
                  save_side_by_side: bool,
                  cmap: str,
                  min_depth: float,
                  max_depth: float,
                  keep_input_resolution: bool,
                  gt_depth_dir: str | None = None,
                  gt_depth_scale: float = REALSENSE_DEFAULT_DEPTH_SCALE,
                  save_raw_pred: bool = False,
                  scales_json_path: str | None = None) -> None:
    model.eval()
    scales_log: list[dict] = []
    for src in tqdm(image_paths, desc="Inferring"):
        try:
            inp, (H, W) = _load_image_tensor(src, device)
        except Exception as e:
            print(f"  [skip] {src}: failed to load ({e})")
            continue

        pred = model(inp)  # (1, 1, 224, 224)

        if keep_input_resolution:
            pred = F.interpolate(pred, size=(H, W),
                                 mode="bilinear", align_corners=False)
        depth_raw = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        depth_raw = np.clip(depth_raw, min_depth, max_depth)

        scale_applied = 1.0
        depth = depth_raw
        gt_path = None
        if gt_depth_dir:
            gt_path = _find_gt_depth_path(src, gt_depth_dir)
            if gt_path is None:
                print(f"  [warn] no GT depth for {os.path.basename(src)} in {gt_depth_dir!r}; saving unscaled")
            else:
                try:
                    gt_m = _load_gt_depth_metres(gt_path, gt_depth_scale)
                    gt_m = _resize_depth_nearest(gt_m, depth_raw.shape[0], depth_raw.shape[1])
                    scale_applied = _median_scale_factor(depth_raw, gt_m, min_depth, max_depth)
                    depth = np.clip(depth_raw * scale_applied, min_depth, max_depth)
                except Exception as e:
                    print(f"  [warn] GT load/scale failed for {src}: {e}; saving unscaled")
                    depth = depth_raw
                    scale_applied = 1.0
            scales_log.append({
                "rgb": os.path.abspath(src),
                "gt_depth": os.path.abspath(gt_path) if gt_path else None,
                "median_scale": scale_applied,
            })

        if save_raw_pred and gt_depth_dir and scale_applied != 1.0:
            raw_path = _default_output_path(src, output_dir, "_depth_raw", "npy")
            np.save(raw_path, depth_raw)

        if save_npy:
            npy_path = _default_output_path(src, output_dir, "_depth", "npy")
            np.save(npy_path, depth)

        if save_png or save_side_by_side:
            depth_rgb = _colourise(depth, cmap)
            if save_png:
                png_path = _default_output_path(src, output_dir, "_depth", "png")
                Image.fromarray(depth_rgb).save(png_path)

            if save_side_by_side:
                rgb = np.asarray(Image.open(src).convert("RGB"))
                if depth_rgb.shape[:2] != rgb.shape[:2]:
                    depth_rgb = np.asarray(
                        Image.fromarray(depth_rgb).resize(
                            (rgb.shape[1], rgb.shape[0]), Image.BILINEAR
                        )
                    )
                combo = np.concatenate([rgb, depth_rgb], axis=1)
                sbs_path = _default_output_path(src, output_dir, "_sidebyside", "png")
                Image.fromarray(combo).save(sbs_path)

    if scales_json_path and scales_log:
        outp = os.path.abspath(scales_json_path)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(scales_log, f, indent=2)
        print(f"Wrote per-image median scales to {outp}")


def main():
    p = argparse.ArgumentParser(description="FastDepth inference on RGB images")
    p.add_argument("--input", action="append", required=True,
                   help="Input image path, directory, or glob. May be passed "
                        "multiple times.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write outputs. Defaults to each input's own "
                        "directory.")
    p.add_argument("--variant", type=str, default="v1", choices=("v1", "v2"),
                   help="v1 = MobileNetV1 + NNConv5; v2 = MobileNetV2 + NNConv5V2.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to FastDepth .pth. Omit to auto-download the "
                        "default weights for --variant from GitHub into "
                        "fastdepth_ref/Weights/. If the given path does not "
                        "exist but the filename matches a known release, it "
                        "will be auto-downloaded to that location.")
    p.add_argument("--ckpt-key", type=str, default=None,
                   choices=sorted(FASTDEPTH_CHECKPOINTS.keys()),
                   help="Alternative to --variant; pick a specific release "
                        "(e.g. v1-l1gn, v2-rmslegn). Overrides --variant.")
    p.add_argument("--weights-dir", type=str, default=WEIGHTS_DIR,
                   help="Local cache directory for auto-downloaded weights.")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda', 'cpu', etc. Default: cuda if available.")
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--cmap", type=str, default="magma",
                   help="matplotlib colormap for the depth preview PNG.")
    p.add_argument("--native-resolution", action="store_true",
                   help="Upsample predictions back to each input image's "
                        "original (H, W). Default: keep 224x224 output.")
    p.add_argument("--no-npy", action="store_true",
                   help="Skip writing the raw .npy depth file.")
    p.add_argument("--no-png", action="store_true",
                   help="Skip writing the colourised .png preview.")
    p.add_argument("--side-by-side", action="store_true",
                   help="Additionally save an RGB|depth concatenated PNG "
                        "(uses native image resolution).")
    p.add_argument("--gt-depth-dir", type=str, default=None,
                   help="Folder of GT depth PNGs (paired by filename with each RGB). "
                        "If set, apply per-image median scaling to predictions before save "
                        "(same as eval_nyu / eval_fastdepth 'median_scaled').")
    p.add_argument("--gt-depth-scale", type=float, default=REALSENSE_DEFAULT_DEPTH_SCALE,
                   help="Metres per uint16 unit for GT depth PNGs (RealSense default 0.001).")
    p.add_argument("--save-raw-pred", action="store_true",
                   help="With --gt-depth-dir: also save unscaled network output as "
                        "*_depth_raw.npy next to *_depth.npy.")
    p.add_argument("--scales-json", type=str, default=None,
                   help="Optional path to write per-image median_scale factors as JSON.")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

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
    n_params = sum(x.numel() for x in model.parameters())
    print(f"FastDepth variant: {args.variant}  |  params: {n_params:,}")

    load_fastdepth_checkpoint(model, ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    image_paths = _gather_inputs(args.input)
    if not image_paths:
        print("No input images found. Nothing to do.")
        sys.exit(1)
    print(f"Found {len(image_paths)} input image(s).")
    if args.output_dir:
        print(f"Writing outputs to: {os.path.abspath(args.output_dir)}")

    run_inference(
        model, image_paths,
        device=device,
        output_dir=args.output_dir,
        save_npy=not args.no_npy,
        save_png=not args.no_png,
        save_side_by_side=args.side_by_side,
        cmap=args.cmap,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        keep_input_resolution=args.native_resolution,
        gt_depth_dir=args.gt_depth_dir,
        gt_depth_scale=args.gt_depth_scale,
        save_raw_pred=args.save_raw_pred,
        scales_json_path=args.scales_json,
    )
    print("Done.")


if __name__ == "__main__":
    main()
