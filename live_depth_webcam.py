"""
Live webcam: show RGB plus DoGDepthNet and FastDepth depth maps side by side.

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


def _parse_sigma_pairs(s: str):
    pairs = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pairs.append((float(a.strip()), float(b.strip())))
    return pairs


def _draw_panel_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Return a copy of ``img_bgr`` with a readable title at the top-left."""
    out = img_bgr.copy()
    h = out.shape[0]
    margin = max(6, h // 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(h / 360.0 * 0.9, 0.5, 1.15))
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + margin * 2
    box_h = th + baseline + margin * 2
    cv2.rectangle(out, (0, 0), (min(out.shape[1] - 1, box_w), min(h - 1, box_h)),
                  (0, 0, 0), -1)
    org = (margin, margin + th)
    cv2.putText(out, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _stack_panels(
    rgb_bgr: np.ndarray,
    dog_bgr: np.ndarray,
    fd_bgr: np.ndarray,
    target_h: int,
    labels: tuple[str, str, str] | None = None,
) -> np.ndarray:
    """Resize each panel to height ``target_h``, annotate, concatenate horizontally."""
    if labels is None:
        labels = _DEFAULT_PANEL_LABELS
    panels = []
    for img, label in zip((rgb_bgr, dog_bgr, fd_bgr), labels):
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        panel = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        panels.append(_draw_panel_label(panel, label))
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

            composite = _stack_panels(frame_bgr, dog_color, fd_color, args.display_height)

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
