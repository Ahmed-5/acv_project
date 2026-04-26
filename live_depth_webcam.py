"""
Live webcam: show RGB plus DoGDepthNet and FastDepth depth maps side by side.
Each depth map has a vertical colorbar (same matplotlib ``--cmap`` as the image)
with numeric ticks for the 2-98%ile depth range used to stretch colours.

Preprocessing matches ``eval_nyu.py`` / ``eval_fastdepth_nyu.py`` (ImageNet norm at
``--img-size`` for DoG; un-normalise to [0,1] and 224x224 for FastDepth).

Example:
  python live_depth_webcam.py --dog-checkpoint checkpoints/dog_depth_nyu_v3.pth

  # front camera, flip horizontally
  python live_depth_webcam.py --dog-checkpoint ckpt.pth --camera 1 --mirror

  # explicit FastDepth weights
  python live_depth_webcam.py --dog-checkpoint ckpt.pth \\
      --fastdepth-checkpoint fastdepth_ref/Weights/FastDepth_L1_Best.pth

  # OpenCV without GUI (headless wheel): use matplotlib for the preview window
  python live_depth_webcam.py --dog-checkpoint ckpt.pth --display matplotlib
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastdepth_ref import nyudepthv2_shim as _nyudepthv2_shim

sys.modules.setdefault("nyudepthv2", _nyudepthv2_shim)

from data.nyu_dataset import _img_transform
from eval_nyu import load_model, parse_img_size
from eval_fastdepth_nyu import (
    FASTDEPTH_CHECKPOINTS,
    WEIGHTS_DIR,
    FASTDEPTH_INPUT,
    build_fastdepth,
    ensure_fastdepth_checkpoint,
    load_fastdepth_checkpoint,
    unnormalize_to_unit,
)
from infer_fastdepth import _colourise

_DEFAULT_PANEL_LABELS = ("Camera (RGB)", "DoGDepthNet", "FastDepth")


def _fmt_depth_m(x: float) -> str:
    ax = abs(x)
    if ax >= 10:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.3f}"
    return f"{x:.4g}"


def _viz_depth_range_m(depth: np.ndarray) -> tuple[float, float]:
    """2nd–98th percentile of finite depths (same idea as ``infer_fastdepth._colourise``)."""
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return 0.0, 1.0
    lo = float(np.percentile(d[finite], 2))
    hi = float(np.percentile(d[finite], 98))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def _vertical_colorbar_bgr(height: int, bar_width: int, cmap_name: str) -> np.ndarray:
    """Vertical ramp top=t=0 (maps to ``lo``), bottom=t=1 (maps to ``hi``), as in ``_colourise``."""
    bw = max(1, bar_width)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    t = np.repeat(t[:, np.newaxis], bw, axis=1)
    try:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgba = cmap(t)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        bar = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        g = (t * 255.0).astype(np.uint8)
        bar = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bar, (0, 0), (bar.shape[1] - 1, bar.shape[0] - 1), (220, 220, 220), 1)
    return bar


def _colorbar_column_bgr(
    height: int,
    cmap_name: str,
    lo: float,
    hi: float,
    bar_width: int,
    label_pad: int,
) -> np.ndarray:
    """Colorbar strip plus tick labels (hi at top, lo at bottom) in metres."""
    bar = _vertical_colorbar_bgr(height, bar_width, cmap_name)
    lp = max(36, label_pad)
    labels = np.zeros((height, lp, 3), dtype=np.uint8)
    labels[:] = (28, 28, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(height / 420.0 * 0.55, 0.38, 0.72))
    th = 1
    hi_s = _fmt_depth_m(hi)
    lo_s = _fmt_depth_m(lo)
    margin = max(4, height // 55)
    (tw_h, th_h), bl_h = cv2.getTextSize(hi_s, font, scale, th)
    cv2.putText(labels, hi_s, (0, margin + th_h), font, scale, (235, 235, 235), th, cv2.LINE_AA)
    (tw_l, th_l), bl_l = cv2.getTextSize(lo_s, font, scale, th)
    cv2.putText(labels, lo_s, (0, height - margin - bl_l), font, scale, (235, 235, 235), th, cv2.LINE_AA)
    unit_scale = scale * 0.82
    (tw_u, th_u), _ = cv2.getTextSize("m", font, unit_scale, th)
    cv2.putText(
        labels, "m", (0, (height + th_u) // 2), font, unit_scale, (160, 160, 160), th, cv2.LINE_AA)
    gap = np.full((height, 2, 3), 18, dtype=np.uint8)
    return np.concatenate([bar, gap, labels], axis=1)


def _parse_sigma_pairs(s: str):
    pairs = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pairs.append((float(a.strip()), float(b.strip())))
    return pairs


def _draw_panel_label(
    img_bgr: np.ndarray,
    text: str,
    extra_lines: tuple[str, ...] = (),
) -> np.ndarray:
    """Return a copy of ``img_bgr`` with a title and optional smaller lines at top-left."""
    out = img_bgr.copy()
    h, w_img = out.shape[:2]
    margin = max(6, h // 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale = float(np.clip(h / 360.0 * 0.9, 0.5, 1.15))
    sub_scale = main_scale * 0.58
    main_th = max(1, int(round(main_scale * 2)))
    sub_th = max(1, int(round(sub_scale * 2)))

    lines: list[tuple[str, float, int]] = [(text, main_scale, main_th)]
    for ex in extra_lines:
        lines.append((ex, sub_scale, sub_th))

    line_sizes: list[tuple[int, int, int, float, int]] = []
    max_tw = 0
    total_h = margin
    for line, sc, th in lines:
        (tw, th_px), bl = cv2.getTextSize(line, font, sc, th)
        line_sizes.append((tw, th_px, bl, sc, th))
        max_tw = max(max_tw, tw)
        total_h += th_px + bl + max(2, margin // 3)
    total_h += margin // 2

    box_w = min(w_img - 1, max_tw + margin * 2)
    box_h = min(h - 1, total_h)
    cv2.rectangle(out, (0, 0), (box_w, box_h), (0, 0, 0), -1)

    y = margin
    for line, (_, th_px, bl, sc, th) in zip([L[0] for L in lines], line_sizes):
        y += th_px
        cv2.putText(out, line, (margin, y), font, sc, (255, 255, 255), th, cv2.LINE_AA)
        y += bl + max(2, margin // 3)
    return out


def _stack_panels(
    rgb_bgr: np.ndarray,
    dog_bgr: np.ndarray,
    fd_bgr: np.ndarray,
    target_h: int,
    labels: tuple[str, str, str] | None = None,
    extra_lines: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None,
    *,
    cmap_name: str = "magma",
    depth_colormap_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    colorbar_bar_width: int = 22,
    colorbar_label_width: int = 52,
) -> np.ndarray:
    """Resize each panel to height ``target_h``, annotate; depth columns get a colorbar."""
    if labels is None:
        labels = _DEFAULT_PANEL_LABELS
    if extra_lines is None:
        extra_lines = ((), (), ())
    panels = []
    for idx, (img, label, extras) in enumerate(
            zip((rgb_bgr, dog_bgr, fd_bgr), labels, extra_lines)):
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        panel = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        panel = _draw_panel_label(panel, label, extras)
        if idx > 0 and depth_colormap_ranges is not None:
            lo, hi = depth_colormap_ranges[idx - 1]
            cb = _colorbar_column_bgr(
                panel.shape[0], cmap_name, lo, hi,
                colorbar_bar_width, colorbar_label_width,
            )
            sep = np.full((panel.shape[0], 3, 3), 40, dtype=np.uint8)
            panel = np.concatenate([panel, sep, cb], axis=1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _opencv_highgui_works() -> bool:
    """Return False when OpenCV has no GUI (e.g. opencv-python-headless)."""
    try:
        cv2.imshow("__live_depth_gui_probe__", np.zeros((2, 2, 3), np.uint8))
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False
    finally:
        try:
            cv2.destroyWindow("__live_depth_gui_probe__")
        except cv2.error:
            pass


def _safe_destroy_opencv_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main():
    p = argparse.ArgumentParser(description="Webcam live depth: DoGDepthNet vs FastDepth")
    p.add_argument("--dog-checkpoint", type=str, required=True)
    p.add_argument("--img-size", type=parse_img_size, default="320,416",
                   help="H,W resize for DoGDepthNet input (default matches training).")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--mirror", action="store_true",
                   help="Flip image horizontally (typical for front-facing cameras).")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--no-pretrained-backbone", action="store_true")
    p.add_argument("--sigma-pairs", type=str, default="0.5,1.0;1.0,2.0;2.0,4.0")
    p.add_argument("--branch-ch", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--min-depth", type=float, default=1e-3)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--use-ema", action="store_true")

    p.add_argument("--fastdepth-variant", type=str, default="v1", choices=("v1", "v2"))
    p.add_argument("--fastdepth-checkpoint", type=str, default=None)
    p.add_argument("--fastdepth-ckpt-key", type=str, default=None,
                   choices=sorted(FASTDEPTH_CHECKPOINTS.keys()))
    p.add_argument("--fastdepth-weights-dir", type=str, default=WEIGHTS_DIR)

    p.add_argument("--display-height", type=int, default=360,
                   help="Pixel height of each panel in the OpenCV window.")
    p.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap for depth.")
    p.add_argument("--colorbar-width", type=int, default=22,
                   help="Width (px) of the vertical colorbar gradient beside each depth map.")
    p.add_argument("--colorbar-label-width", type=int, default=52,
                   help="Width (px) for min/max depth labels next to each colorbar.")
    p.add_argument("--window", type=str, default="RGB | DoGDepthNet | FastDepth")
    p.add_argument(
        "--display",
        type=str,
        default="auto",
        choices=("auto", "opencv", "matplotlib"),
        help="Preview backend. Use matplotlib if cv2.imshow fails (headless OpenCV on Windows).",
    )
    args = p.parse_args()

    sigma_pairs = _parse_sigma_pairs(args.sigma_pairs)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dog, _meta = load_model(
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
    dog.eval()
    print(f"Loaded DoGDepthNet: {args.dog_checkpoint}")

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
    fd_model.eval()
    print(f"Loaded FastDepth: {fd_path} (variant {fd_variant})")

    dog_tf = _img_transform(args.img_size)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    display_backend = args.display
    if display_backend == "auto":
        display_backend = "opencv" if _opencv_highgui_works() else "matplotlib"
        if display_backend == "matplotlib":
            print(
                "OpenCV HighGUI is not available (common with opencv-python-headless). "
                "Using matplotlib for the preview window."
            )

    if display_backend == "opencv":
        print("Press 'q' in the OpenCV window to quit.")
    else:
        print("Press 'q' in the matplotlib window (or close it) to quit.")

    t_prev = time.perf_counter()
    n_frames = 0
    quit_requested = False

    mpl_fig = mpl_ax = mpl_im = None
    mpl_plt = None
    if display_backend == "matplotlib":
        import matplotlib.pyplot as mpl_plt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mpl_plt.ion()
        mpl_fig, mpl_ax = mpl_plt.subplots(figsize=(12, 4))
        mpl_ax.axis("off")
        mpl_im = mpl_ax.imshow(
            np.zeros((args.display_height, args.display_height * 3, 3), dtype=np.uint8))
        mpl_fig.canvas.manager.set_window_title(args.window)

        def _on_mpl_key(event):
            nonlocal quit_requested
            if event.key in ("q", "Q"):
                quit_requested = True

        mpl_fig.canvas.mpl_connect("key_press_event", _on_mpl_key)
        mpl_plt.show(block=False)

    try:
        while True:
            if display_backend == "matplotlib" and mpl_fig is not None and mpl_plt is not None:
                if quit_requested or not mpl_plt.fignum_exists(mpl_fig.number):
                    break

            ok, frame_bgr = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dog_in = dog_tf(Image.fromarray(rgb)).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                dog_pred = dog(dog_in)
                dog_pred = F.interpolate(
                    dog_pred,
                    size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )

                img_fd = unnormalize_to_unit(dog_in)
                img_fd = F.interpolate(
                    img_fd, size=FASTDEPTH_INPUT,
                    mode="bilinear", align_corners=False,
                )
                fd_pred = fd_model(img_fd)
                fd_pred = F.interpolate(
                    fd_pred,
                    size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                    mode="bilinear", align_corners=False,
                )

            dog_np = dog_pred.squeeze().clamp(args.min_depth, args.max_depth).cpu().numpy()
            fd_np = fd_pred.squeeze().clamp(args.min_depth, args.max_depth).cpu().numpy()

            dog_color = cv2.cvtColor(_colourise(dog_np, args.cmap), cv2.COLOR_RGB2BGR)
            fd_color = cv2.cvtColor(_colourise(fd_np, args.cmap), cv2.COLOR_RGB2BGR)

            dog_vlo, dog_vhi = _viz_depth_range_m(dog_np)
            fd_vlo, fd_vhi = _viz_depth_range_m(fd_np)
            mn, mx = args.min_depth, args.max_depth
            clip_line = (f"depth (m), clipped to [{_fmt_depth_m(mn)}, {_fmt_depth_m(mx)}]",)
            panel_extras = ((), clip_line, clip_line)
            composite = _stack_panels(
                frame_bgr, dog_color, fd_color, args.display_height,
                extra_lines=panel_extras,
                cmap_name=args.cmap,
                depth_colormap_ranges=((dog_vlo, dog_vhi), (fd_vlo, fd_vhi)),
                colorbar_bar_width=args.colorbar_width,
                colorbar_label_width=args.colorbar_label_width,
            )

            if display_backend == "opencv":
                cv2.imshow(args.window, composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                mpl_im.set_data(composite_rgb)
                mpl_fig.canvas.draw_idle()
                mpl_fig.canvas.flush_events()
                mpl_plt.pause(0.001)

            n_frames += 1
            if n_frames % 30 == 0:
                now = time.perf_counter()
                fps = 30.0 / (now - t_prev)
                t_prev = now
                print(f"~{fps:.1f} fps (rolling over last 30 frames)")
    finally:
        cap.release()
        _safe_destroy_opencv_windows()
        if display_backend == "matplotlib" and mpl_fig is not None and mpl_plt is not None:
            mpl_plt.close(mpl_fig)


if __name__ == "__main__":
    main()
