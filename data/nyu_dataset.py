"""
NYU Depth V2 dataset — loading modes:
  1. 'hf'  : HuggingFace sayakpaul/nyu_depth_v2  (no local files needed)
  2. 'mat' : local nyu_depth_v2_labeled.mat       (depth already in metres)
  3. 'raw' : raw scene directories (.pgm/.ppm)    (full toolbox pipeline)
  4. RealSenseAlignedPNG: ``depthData/runs/...`` with rgb/, depth/, meta.json
"""
import os
import re
import io
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa
import h5py
import glob

from data.nyu_toolbox import (
    project_depth_map, crop_image,
    CROP_SIZE, MAX_DEPTH
)

NYU_MEAN = [0.485, 0.456, 0.406]
NYU_STD  = [0.229, 0.224, 0.225]

# Intel RealSense D400-class devices typically use ~1 mm per uint16 depth unit.
REALSENSE_DEFAULT_DEPTH_SCALE = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resize_depth(depth_np: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Nearest-neighbour resize to avoid interpolating across edges."""
    t = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(h, w), mode='nearest').squeeze(0)  # (1,H,W)


def _img_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NYU_MEAN, std=NYU_STD),
    ])



class NYUParquet(Dataset):
    """
    Loads NYU Depth V2 from locally downloaded HuggingFace Parquet shards.

    Each row contains:
        'image'     : {'bytes': ..., 'path': ...}  — uint8 RGB JPEG
        'depth_map' : {'bytes': ..., 'path': ...}  — uint8 or uint16 PNG

    Args:
        parquet_paths : list of .parquet file paths (all train shards, or val)
        img_size      : (H, W) tuple for output tensors
        augment       : apply random flip/crop/jitter (True for train only)
    """

    def __init__(self, parquet_paths: list, img_size=(256, 320), augment=False):
        tables      = [pq.read_table(p) for p in parquet_paths]
        self.df     = pa.concat_tables(tables).to_pandas()
        self.tf     = _img_transform(img_size)
        self.h, self.w = img_size
        self.augment = augment

        # Auto-detect depth scale from first row
        sample_depth = np.array(
            Image.open(io.BytesIO(self.df.iloc[0]['depth_map']['bytes']))
        )
        if sample_depth.dtype == np.uint8:
            self.depth_scale = 255.0 / MAX_DEPTH    # uint8  → metres
        else:
            self.depth_scale = 65535.0 / MAX_DEPTH  # uint16 → metres

        print(f"NYUParquet: {len(self.df)} samples | "
              f"depth dtype={sample_depth.dtype} | "
              f"augment={augment}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load RGB ──────────────────────────────────────────────────────────
        image = Image.open(
            io.BytesIO(row['image']['bytes'])
        ).convert('RGB')

        # ── Load depth ────────────────────────────────────────────────────────
        depth_raw = np.array(
            Image.open(io.BytesIO(row['depth_map']['bytes']))
        ).astype(np.float32)
        depth_m = depth_raw / self.depth_scale   # → metres

        # ── Augmentation (train only) ─────────────────────────────────────────
        if self.augment:
            import torchvision.transforms.functional as TF
            import random

            # Random horizontal flip
            if random.random() > 0.5:
                image   = TF.hflip(image)
                depth_m = depth_m[:, ::-1].copy()

            # Random crop ±15 %
            w, h      = image.size
            crop_frac = random.uniform(0.85, 1.0)
            ch, cw    = int(h * crop_frac), int(w * crop_frac)
            top       = random.randint(0, h - ch)
            left      = random.randint(0, w - cw)
            image     = TF.crop(image, top, left, ch, cw)
            depth_m   = depth_m[top:top + ch, left:left + cw]

            # Color jitter on RGB only
            image = transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05
            )(image)

            # Random rotation ±5°
            if random.random() > 0.7:
                angle   = random.uniform(-5, 5)
                image   = TF.rotate(image, angle)
                depth_t = torch.from_numpy(depth_m).unsqueeze(0)
                depth_m = TF.rotate(depth_t, angle).squeeze(0).numpy()

        # ── Tensorise ─────────────────────────────────────────────────────────
        image = self.tf(image)
        depth = _resize_depth(depth_m, self.h, self.w).clamp(0, MAX_DEPTH)
        mask  = (depth > 0) & (depth <= MAX_DEPTH)

        return {'image': image, 'depth': depth, 'mask': mask}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: HuggingFace dataset
# ─────────────────────────────────────────────────────────────────────────────
class NYUHuggingFace(Dataset):
    """sayakpaul/nyu_depth_v2 — depth already in metres, aligned to RGB."""

    def __init__(self, split='train', img_size=(256, 320)):
        from datasets import load_dataset
        hf_split = 'train' if split == 'train' else 'validation'
        self.data = load_dataset('sayakpaul/nyu_depth_v2', split=hf_split)
        self.tf   = _img_transform(img_size)
        self.h, self.w = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        image = self.tf(s['image'].convert('RGB'))

        d = np.array(s['depth_map'], dtype=np.float32)
        if d.max() > 100:           # stored as mm → convert to metres
            d /= 1000.0
        depth = _resize_depth(d, self.h, self.w).clamp(0, MAX_DEPTH)
        mask  = (depth > 0) & (depth <= MAX_DEPTH)
        return {'image': image, 'depth': depth, 'mask': mask}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: local .mat file  (nyu_depth_v2_labeled.mat)
# ─────────────────────────────────────────────────────────────────────────────
class NYUMatFile(Dataset):
    """
    Load from the official nyu_depth_v2_labeled.mat file.
    Depth is already metric (metres) and aligned to RGB at 480×640.
    We apply the toolbox crop (427×561) then resize to img_size.

    Download: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
              (nyu_depth_v2_labeled.mat, ~2.8 GB)
    """

    # Standard 249/215 train/test split indices (0-indexed)
    # Obtained from: https://github.com/dwofk/fast-depth
    SPLITS_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'

    def __init__(self, mat_path: str, split='train', img_size=(256, 320),
                 apply_crop=True):
        import h5py
        self.tf          = _img_transform(img_size)
        self.h, self.w   = img_size
        self.apply_crop  = apply_crop

        with h5py.File(mat_path, 'r') as f:
            # shapes: images (3,480,640,N)  depths (480,640,N) — HDF5 transposes
            self.images = f['images'][:]   # (N, 3, 640, 480) after h5py transpose
            self.depths = f['depths'][:]   # (N, 640, 480)

        # h5py loads in C order (transposed from MATLAB), fix to (N,3,H,W) etc.
        # MATLAB stores as (H,W,C,N); h5py reads as (N,C,W,H) — transpose back
        self.images = self.images.transpose(0, 1, 3, 2)  # (N, 3, H, W) uint8
        self.depths = self.depths.transpose(0, 2, 1)      # (N, H, W) float64

        # Load train/test split
        split_indices = self._get_split_indices(mat_path, split)
        self.images = self.images[split_indices]
        self.depths = self.depths[split_indices]

    def _get_split_indices(self, mat_path, split):
        """
        Uses the standard 654-test / remaining-train split.
        Falls back to last 654 frames as test if splits.mat not found.
        """
        splits_path = os.path.join(os.path.dirname(mat_path), 'splits.mat')
        if os.path.exists(splits_path):
            import scipy.io as sio
            s = sio.loadmat(splits_path)
            key = 'trainNdxs' if split == 'train' else 'testNdxs'
            return s[key].ravel().astype(int) - 1  # MATLAB 1-indexed → 0-indexed
        else:
            N = self.images.shape[0]
            idx = np.arange(N)
            return idx[:-654] if split == 'train' else idx[-654:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ── RGB ─────────────────────────────────────────────────────────────
        img_np = self.images[idx]               # (3, H, W) uint8
        img_pil = Image.fromarray(
            img_np.transpose(1, 2, 0)           # (H, W, 3)
        )
        if self.apply_crop:
            # crop_image expects (H,W,C); PIL crop is faster
            img_pil = img_pil.crop((40, 44, 601, 471))  # (left,top,right,bottom)
        image = self.tf(img_pil)

        # ── Depth ────────────────────────────────────────────────────────────
        d = self.depths[idx].astype(np.float32)  # (H, W) in metres
        if self.apply_crop:
            d = crop_image(d)                    # (427, 561)
        depth = _resize_depth(d, self.h, self.w).clamp(0, MAX_DEPTH)
        mask  = (depth > 0) & (depth <= MAX_DEPTH)
        return {'image': image, 'depth': depth, 'mask': mask}


# ─────────────────────────────────────────────────────────────────────────────
# RealSense aligned captures (rgb/ + depth/ PNG folders, optional meta.json)
# ─────────────────────────────────────────────────────────────────────────────
def _load_realsense_meta(root_dir: str) -> dict:
    path = os.path.join(root_dir, "meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _realsense_resolve_rgb_depth_dirs(
    root_dir: str,
    rgb_subdir: str | None = None,
    depth_subdir: str | None = None,
) -> tuple[str, str]:
    """
    Locate rgb and depth folders under a capture root.

    Default names ``rgb/`` and ``depth/``; also accepts common variants
    (case-insensitive): ``color``, ``Depth``, ``depth_raw``, etc.
    Optional ``rgb_subdir`` / ``depth_subdir`` are path segments relative to
    ``root_dir`` (e.g. ``images`` → ``root_dir/images``).
    """
    root = os.path.abspath(root_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"RealSenseAlignedPNG: run folder does not exist or is not a directory: {root!r}"
        )

    child_dirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    lower_to_path = {name.lower(): os.path.join(root, name) for name in child_dirs}

    rgb_candidates = ("rgb", "color", "colour", "images", "image")
    depth_candidates = ("depth", "depth_raw", "depths", "depth_maps")

    def pick(cands):
        for c in cands:
            if c in lower_to_path:
                return lower_to_path[c]
        return None

    if rgb_subdir is not None:
        rgb_path = os.path.join(root, rgb_subdir)
        if not os.path.isdir(rgb_path):
            raise FileNotFoundError(
                f"RealSenseAlignedPNG: rgb folder {rgb_path!r} does not exist "
                f"(check --rs-rgb-subdir)"
            )
    else:
        rgb_path = pick(rgb_candidates)

    if depth_subdir is not None:
        depth_path = os.path.join(root, depth_subdir)
        if not os.path.isdir(depth_path):
            raise FileNotFoundError(
                f"RealSenseAlignedPNG: depth folder {depth_path!r} does not exist "
                f"(check --rs-depth-subdir)"
            )
    else:
        depth_path = pick(depth_candidates)

    if rgb_path is None or depth_path is None or not os.path.isdir(rgb_path) \
            or not os.path.isdir(depth_path):
        hint = (
            "Expected a folder containing rgb/ and depth/ (or use "
            "--rs-rgb-subdir / --rs-depth-subdir). "
            "If you recorded with collect_data.py, set SAVE_DIR to this path or run "
            "depthData/organize_run.py to move a capture into depthData/runs/<name>/."
        )
        listing = ", ".join(sorted(child_dirs)) if child_dirs else "(empty — no subfolders)"
        raise FileNotFoundError(
            f"RealSenseAlignedPNG: could not find rgb + depth image folders under {root!r}. "
            f"Subfolders present: {listing}. {hint}"
        )
    return rgb_path, depth_path


class RealSenseAlignedPNG(Dataset):
    """
    Loads paired RGB + uint16 depth PNGs from ``root_dir`` (aligned depth to color).

    Layout::

        root_dir/
          meta.json          # optional; should contain depth_scale (metres per uint16 unit)
          rgb/000000.png
          depth/000000.png

    ``meta.json`` from ``depthData/collect_data.py`` includes ``depth_scale`` and
    ``color_space`` (``rgb`` for current script; older runs may be ``bgr_on_disk``
    or missing — treated as OpenCV BGR-on-disk).

    Args:
        root_dir: Run folder (e.g. ``depthData/runs/my_run``).
        img_size: Output (H, W) for tensors.
        depth_scale: Metres per depth PNG unit; if None, read from ``meta.json``.
        max_depth: Clip depth and mask upper bound (default matches NYU toolbox).
        color_space: ``rgb`` | ``bgr_on_disk`` | None (infer from meta, else
            ``bgr_on_disk`` for backward compatibility).
        rgb_subdir: Optional subdirectory name for colour images (under ``root_dir``).
        depth_subdir: Optional subdirectory name for depth PNGs (under ``root_dir``).
    """

    def __init__(self, root_dir: str, img_size=(320, 416),
                 depth_scale: float | None = None,
                 max_depth: float = MAX_DEPTH,
                 color_space: str | None = None,
                 rgb_subdir: str | None = None,
                 depth_subdir: str | None = None):
        self.root = os.path.abspath(root_dir)
        self.rgb_dir, self.depth_dir = _realsense_resolve_rgb_depth_dirs(
            self.root, rgb_subdir=rgb_subdir, depth_subdir=depth_subdir,
        )

        meta = _load_realsense_meta(self.root)
        if depth_scale is None:
            if "depth_scale" in meta:
                depth_scale = float(meta["depth_scale"])
            else:
                depth_scale = REALSENSE_DEFAULT_DEPTH_SCALE
                if meta:
                    print(
                        f"[RealSenseAlignedPNG] meta.json has no 'depth_scale'; "
                        f"using default {depth_scale} m/unit (typical D400 RealSense). "
                        f"Override with --rs-depth-scale if wrong."
                    )
                else:
                    print(
                        f"[RealSenseAlignedPNG] no meta.json under {self.root!r}; "
                        f"using default depth_scale={depth_scale} m/unit. "
                        f"Add meta.json or pass --rs-depth-scale for an accurate scale."
                    )
        self.depth_scale = depth_scale

        if color_space is None:
            color_space = meta.get("color_space", "bgr_on_disk")
        self.color_space = color_space.lower()

        self.tf = _img_transform(img_size)
        self.h, self.w = img_size
        self.max_depth = float(max_depth)

        rgb_names = {f for f in os.listdir(self.rgb_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        depth_names = {f for f in os.listdir(self.depth_dir)
                       if f.lower().endswith(".png")}
        paired = sorted(rgb_names & depth_names)
        if not paired:
            raise FileNotFoundError(
                f"RealSenseAlignedPNG: no matching PNG basenames between "
                f"{self.rgb_dir!r} and {self.depth_dir!r}"
            )
        self._names = paired
        print(f"RealSenseAlignedPNG: {len(self._names)} pairs | "
              f"rgb={self.rgb_dir} | depth={self.depth_dir} | "
              f"depth_scale={self.depth_scale} | color_space={self.color_space} | "
              f"max_depth={self.max_depth}")

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        name = self._names[idx]
        rgb_path = os.path.join(self.rgb_dir, name)
        depth_path = os.path.join(self.depth_dir, name)

        rgb_np = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        # Legacy ``collect_data`` saved BGR arrays with cv2.imwrite; PNG "RGB"
        # channels are then B,G,R — reverse to true RGB for the model.
        if self.color_space != "rgb":
            rgb_np = rgb_np[..., ::-1].copy()

        depth_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
        depth_m = depth_u16.astype(np.float32) * self.depth_scale

        image = self.tf(Image.fromarray(rgb_np))
        depth = _resize_depth(depth_m, self.h, self.w).clamp(0, self.max_depth)
        mask = (depth > 0) & (depth <= self.max_depth)
        return {'image': image, 'depth': depth, 'mask': mask}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: raw scene directories (.pgm depth + .ppm RGB)
# ─────────────────────────────────────────────────────────────────────────────
class NYURawScenes(Dataset):
    """
    Processes raw Kinect scene dumps using the full toolbox pipeline
    (undistort → linearise → project → crop).

    Expected directory layout:
        root/
          bedroom_0001/
            d-<timestamp>-<id>.pgm    ← raw uint16 depth (little-endian)
            r-<timestamp>-<id>.ppm    ← uint8 RGB
          kitchen_0003/
            ...

    get_synched_frames.m logic is replicated: depth frames are primary,
    nearest RGB by timestamp is matched.
    """

    def __init__(self, root_dir: str, img_size=(256, 320), fill_depth=False, fill_method='colorization'):
        self.root       = root_dir
        self.tf         = _img_transform(img_size)
        self.h, self.w  = img_size
        self.fill       = fill_depth
        self.pairs      = self._sync_all_scenes()
        self.fill_method = fill_method

    def _sync_scene(self, scene_dir: str):
        """Replicate get_synched_frames.m for one scene directory."""
        files = sorted(os.listdir(scene_dir))
        depth_files = [f for f in files if f.startswith('d-')]
        rgb_files   = [f for f in files if f.startswith('r-')]

        def ts(fname):
            parts = re.split(r'-', fname[2:])
            return float(parts[0])

        rgb_ts = [(ts(f), f) for f in rgb_files]
        pairs  = []
        for d_file in depth_files:
            td = ts(d_file)
            nearest_rgb = min(rgb_ts, key=lambda x: abs(x[0] - td))[1]
            pairs.append((
                os.path.join(scene_dir, d_file),
                os.path.join(scene_dir, nearest_rgb),
            ))
        return pairs

    def _sync_all_scenes(self):
        pairs = []
        for scene in sorted(os.listdir(self.root)):
            scene_path = os.path.join(self.root, scene)
            if os.path.isdir(scene_path):
                pairs.extend(self._sync_scene(scene_path))
        print(f"NYURawScenes: found {len(pairs)} depth-RGB pairs")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        depth_path, rgb_path = self.pairs[idx]

        # ── Load raw depth: uint16 little-endian .pgm, swap bytes ───────────
        raw_depth = np.array(Image.open(depth_path), dtype=np.uint16)
        raw_depth = raw_depth.byteswap()   # MATLAB swapbytes()

        # ── Load RGB ─────────────────────────────────────────────────────────
        rgb_np = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)

        # ── Full toolbox projection pipeline ─────────────────────────────────
        depth_proj, rgb_undist = project_depth_map(raw_depth, rgb_np)

        # ── Crop to valid 427×561 region ─────────────────────────────────────
        depth_cropped = crop_image(depth_proj).astype(np.float32)
        rgb_cropped   = crop_image(rgb_undist)

        # ── Optional: fill missing depth via colorization ────────────────────
        if self.fill:
            if self.fill_method == 'colorization':
                from data.nyu_toolbox import fill_depth_colorization
                depth_cropped = fill_depth_colorization(
                    rgb_cropped.astype(np.float32) / 255.0,
                    depth_cropped.astype(np.float64)
                ).astype(np.float32)
            elif self.fill_method == 'cbf':
                from data.nyu_toolbox import fill_depth_cross_bf
                depth_cropped = fill_depth_cross_bf(
                    rgb_cropped, depth_cropped.astype(np.float64)
                ).astype(np.float32)

        # ── Tensorise ────────────────────────────────────────────────────────
        image = self.tf(Image.fromarray(rgb_cropped))
        depth = _resize_depth(depth_cropped, self.h, self.w).clamp(0, MAX_DEPTH)
        mask  = (depth > 0) & (depth <= MAX_DEPTH)
        return {'image': image, 'depth': depth, 'mask': mask}
    

class NYUDepthH5(Dataset):
    """
    Load NYU Depth V2 from .h5 files organised by scene directory.

    Expected layout (matches your download):
        root/
          train/{scene_name}/*.h5
          val/official/*.h5

    Args:
        h5_dir       : path to train/ or val/official/
        img_size     : (H, W) output size
        augment      : random flip/crop/jitter (True for train only)
        scene_filter : optional list of scene type strings to keep,
                       e.g. ['bedroom', 'kitchen'] — uses get_scene_type()
    """

    def __init__(self, h5_dir: str, img_size=(256, 320),
                 augment=False, scene_filter=None):
        all_files = sorted(glob.glob(
            os.path.join(h5_dir, '**', '*.h5'), recursive=True
        ))
        assert all_files, f"No .h5 files found under {h5_dir}"

        # Build (filepath, scene_name) pairs from directory name
        self.samples = [
            (fp, os.path.basename(os.path.dirname(fp)))
            for fp in all_files
        ]

        # Optional scene-type filter (uses get_scene_type from nyu_toolbox)
        if scene_filter is not None:
            from data.nyu_toolbox import get_scene_type
            self.samples = [
                (fp, scene) for fp, scene in self.samples
                if get_scene_type(scene) in scene_filter
            ]
            assert self.samples, \
                f"scene_filter={scene_filter} matched no scenes in {h5_dir}"

        self.tf        = _img_transform(img_size)
        self.h, self.w = img_size
        self.augment   = augment

        # Collect unique scene types for reporting
        if scene_filter is None:
            from data.nyu_toolbox import get_scene_type
        scene_types = sorted({get_scene_type(s) for _, s in self.samples})
        print(f"NYUDepthH5 : {len(self.samples)} samples | "
              f"{len(scene_types)} scene types | augment={augment}")
        print(f"  scenes   : {', '.join(scene_types)}")

        self._color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.08
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, scene_name = self.samples[idx]
        with h5py.File(filepath, 'r') as f:
            rgb     = np.transpose(np.array(f['rgb']),   (1, 2, 0)).astype(np.uint8)
            depth_m = np.array(f['depth'], dtype=np.float32)

        image = Image.fromarray(rgb)

        if self.augment:
            import torchvision.transforms.functional as TF
            import random

            # ── 1. Horizontal flip ────────────────────────────────────────────────
            if random.random() > 0.5:
                image   = TF.hflip(image)
                depth_m = depth_m[:, ::-1].copy()

            # ── 2. Random crop (85–100%) ──────────────────────────────────────────
            w, h      = image.size
            crop_frac = random.uniform(0.85, 1.0)
            ch, cw    = int(h * crop_frac), int(w * crop_frac)
            top       = random.randint(0, h - ch)
            left      = random.randint(0, w - cw)
            image     = TF.crop(image, top, left, ch, cw)
            depth_m   = depth_m[top:top + ch, left:left + cw]

            depth_m = depth_m * random.uniform(0.95, 1.05)

            # # ── 4. CutFlip (from PMC11243791 — improves REL ~4%) ─────────────────
            # # Split image at random horizontal line, flip top/bottom halves
            # # and swap their depth accordingly
            # if random.random() > 0.5:
            #     cut = random.randint(ch // 4, 3 * ch // 4)
            #     top_img   = np.array(image)[:cut]
            #     bot_img   = np.array(image)[cut:]
            #     top_dep   = depth_m[:cut]
            #     bot_dep   = depth_m[cut:]
            #     image   = Image.fromarray(np.concatenate([
            #         bot_img[::-1], top_img[::-1]], axis=0))
            #     depth_m = np.concatenate([bot_dep[::-1], top_dep[::-1]], axis=0)

            if random.random() > 0.7:
                angle   = random.uniform(-2.5, 2.5)
                image   = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                dep_t   = torch.from_numpy(depth_m).unsqueeze(0)
                depth_m = TF.rotate(dep_t, angle,
                                    interpolation=TF.InterpolationMode.BILINEAR,
                                    fill=0).squeeze(0).numpy()

            # ── 6. Color / photometric augmentations (RGB only) ───────────────────
            # Instantiate once, not per-call
            image = self._color_jitter(image)    # see __init__ below

            # if random.random() > 0.8:
            #     image = TF.gaussian_blur(image, kernel_size=5, sigma=(0.5, 1.5))
            #     # ↑ Especially useful for DoGDepthNet — teaches multi-scale blur

            if random.random() > 0.9:
                image = TF.to_grayscale(image, num_output_channels=3)

            if random.random() > 0.75:
                iw, ih = image.size
                cw_ = random.randint(int(0.05 * iw), int(0.20 * iw))
                ch_ = random.randint(int(0.05 * ih), int(0.20 * ih))
                cx = random.randint(0, iw - cw_)
                cy = random.randint(0, ih - ch_)
                arr = np.array(image)
                fill = np.array([int(255 * c) for c in NYU_MEAN], dtype=arr.dtype)
                arr[cy:cy + ch_, cx:cx + cw_] = fill
                image = Image.fromarray(arr)

        image = self.tf(image)
        depth = _resize_depth(depth_m, self.h, self.w).clamp(0, MAX_DEPTH)
        mask  = (depth > 0) & (depth <= MAX_DEPTH)

        return {'image': image, 'depth': depth, 'mask': mask, 'scene': scene_name}


# ─────────────────────────────────────────────────────────────────────────────
# Unified builder
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(mode='mat',
                  mat_path=None,
                  raw_root=None,
                  parquet_train=None,    # ← new
                  parquet_val=None,      # ← new
                  h5_train_dir=None,    # ← new
                  h5_val_dir=None,      # ← new
                  img_size=(256, 320),
                  batch_size=8,
                  num_workers=4,
                  apply_crop=True):

    if mode == 'mat':
        assert mat_path, "Provide mat_path= for mode='mat'"
        train_ds = NYUMatFile(mat_path, 'train', img_size, apply_crop)
        val_ds   = NYUMatFile(mat_path, 'test',  img_size, apply_crop)

    elif mode == 'parquet':
        assert parquet_train and parquet_val, \
            "Provide parquet_train= and parquet_val= for mode='parquet'"
        train_ds = NYUParquet(parquet_train, img_size, augment=True)
        val_ds   = NYUParquet(parquet_val,   img_size, augment=False)

    elif mode == 'h5':
        assert h5_train_dir and h5_val_dir
        train_ds = NYUDepthH5(h5_train_dir, img_size, augment=True)
        val_ds   = NYUDepthH5(h5_val_dir,   img_size, augment=False)

    elif mode == 'raw':
        assert raw_root, "Provide raw_root= for mode='raw'"
        all_ds = NYURawScenes(raw_root, img_size)
        n_val  = 654
        train_ds, val_ds = torch.utils.data.random_split(
            all_ds, [len(all_ds) - n_val, n_val]
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'mat', 'parquet', or 'raw'.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[{mode}] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader