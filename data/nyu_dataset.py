"""
NYU Depth V2 dataset — three loading modes:
  1. 'hf'  : HuggingFace sayakpaul/nyu_depth_v2  (no local files needed)
  2. 'mat' : local nyu_depth_v2_labeled.mat       (depth already in metres)
  3. 'raw' : raw scene directories (.pgm/.ppm)    (full toolbox pipeline)
"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from data.nyu_toolbox import (
    project_depth_map, crop_image,
    CROP_SIZE, MAX_DEPTH
)

NYU_MEAN = [0.485, 0.456, 0.406]
NYU_STD  = [0.229, 0.224, 0.225]


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


# ─────────────────────────────────────────────────────────────────────────────
# Unified builder
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(mode='mat',
                  mat_path=None,
                  raw_root=None,
                  img_size=(256, 320),
                  batch_size=8,
                  num_workers=4,
                  apply_crop=True):
    """
    mode:
      'hf'  — HuggingFace (no local files needed)
      'mat' — nyu_depth_v2_labeled.mat  (recommended)
      'raw' — raw scene directories
    """
    if mode == 'hf':
        train_ds = NYUHuggingFace('train',      img_size)
        val_ds   = NYUHuggingFace('validation', img_size)

    elif mode == 'mat':
        assert mat_path, "Provide mat_path= for mode='mat'"
        train_ds = NYUMatFile(mat_path, 'train', img_size, apply_crop)
        val_ds   = NYUMatFile(mat_path, 'test',  img_size, apply_crop)

    elif mode == 'raw':
        assert raw_root, "Provide raw_root= for mode='raw'"
        all_ds = NYURawScenes(raw_root, img_size)
        n_val  = 654
        train_ds, val_ds = torch.utils.data.random_split(
            all_ds, [len(all_ds) - n_val, n_val]
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[{mode}] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader