"""
Python port of the NYU Depth V2 MATLAB toolbox.
Translates the following .m files:
  camera_params.m, depth_rel2depth_abs.m, depth_plane2depth_world.m,
  depth_world2rgb_world.m, rgb_world2rgb_plane.m, project_depth_map.m,
  get_projection_mask.m, crop_image.m, fill_depth_colorization.m
"""
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────────────────────────────────────
# camera_params.m — all constants verbatim
# ─────────────────────────────────────────────────────────────────────────────
MAX_DEPTH = 10.0  # metres

# RGB intrinsics
FX_RGB = 5.1885790117450188e+02
FY_RGB = 5.1946961112127485e+02
CX_RGB = 3.2558244941119034e+02
CY_RGB = 2.5373616633400465e+02
K_RGB  = np.array([2.0796615318809061e-01, -5.8613825163911781e-01,
                   7.2231363135888329e-04,  1.0479627195765181e-03,
                    4.9856986684705107e-01])   # [k1,k2,p1,p2,k3]

# Depth intrinsics
FX_D = 5.8262448167737955e+02
FY_D = 5.8269103270988637e+02
CX_D = 3.1304475870804731e+02
CY_D = 2.3844389626620386e+02
K_D  = np.array([-9.9897236553084481e-02,  3.9065324602765344e-01,
                   1.9290592870229277e-03, -1.9422022475975055e-03,
                  -5.1031725053400578e-01])    # [k1,k2,p1,p2,k3]

# Depth-to-RGB extrinsics (from camera_params.m, exactly reproduced)
# MATLAB: R_flat = -[...]; R_mat = reshape(R_flat,[3,3]); R = inv(R_mat')
# reshape in MATLAB is column-major
_R_flat = -np.array([
    9.9997798940829263e-01,  5.0518419386157446e-03,  4.3011152014118693e-03,
   -5.0359919480810989e-03,  9.9998051861143999e-01, -3.6879781309514218e-03,
   -4.3196624923060242e-03,  3.6662365748484798e-03,  9.9998394948385538e-01,
])
_R_col_major = _R_flat.reshape(3, 3, order='F')   # MATLAB column-major fill
R_D2RGB = np.linalg.inv(_R_col_major.T)           # inv(R')

# Translation: MATLAB uses [t_x; t_z; t_y] order
T_D2RGB = np.array([
    2.5031875059141302e-02,   # t_x
   -2.9342312935846411e-04,   # t_z  ← note z before y, matching MATLAB
    6.6238747008330102e-04,   # t_y
])

# Depth linearisation parameters (depth_rel2depth_abs.m)
DEPTH_PARAM1 = 351.3
DEPTH_PARAM2 = 1092.5

# Valid crop after projection (get_projection_mask.m)
# MATLAB 1-indexed [45:471, 41:601] → Python 0-indexed [44:471, 40:601]
CROP_ROWS = slice(44, 471)   # → 427 rows
CROP_COLS = slice(40, 601)   # → 561 cols
CROP_SIZE = (427, 561)


# ─────────────────────────────────────────────────────────────────────────────
# depth_rel2depth_abs.m
# ─────────────────────────────────────────────────────────────────────────────
def depth_rel2depth_abs(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw Kinect disparity values to metric depth (metres).
    Formula: depth = DEPTH_PARAM1 / (DEPTH_PARAM2 - raw)
    raw must be float64 with bytes already swapped (swapbytes in MATLAB).
    """
    depth = DEPTH_PARAM1 / (DEPTH_PARAM2 - raw.astype(np.float64))
    depth = np.clip(depth, 0.0, MAX_DEPTH)
    return depth


# ─────────────────────────────────────────────────────────────────────────────
# depth_plane2depth_world.m
# ─────────────────────────────────────────────────────────────────────────────
def depth_plane2depth_world(depth_abs: np.ndarray) -> np.ndarray:
    """
    Unproject depth image to 3-D using depth camera intrinsics.
    Input : (480, 640) float depth in metres
    Output: (480*640, 3) array of (X, Y, Z) world points
    """
    H, W = depth_abs.shape
    xx, yy = np.meshgrid(np.arange(1, W + 1), np.arange(1, H + 1))
    X = (xx - CX_D) * depth_abs / FX_D
    Y = (yy - CY_D) * depth_abs / FY_D
    Z = depth_abs
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# depth_world2rgb_world.m
# ─────────────────────────────────────────────────────────────────────────────
def depth_world2rgb_world(points3d: np.ndarray) -> np.ndarray:
    """
    Rigid body transform: depth camera world → RGB camera world.
    Input/Output: (N, 3)
    """
    return (R_D2RGB @ points3d.T + T_D2RGB[:, None]).T


# ─────────────────────────────────────────────────────────────────────────────
# rgb_world2rgb_plane.m
# ─────────────────────────────────────────────────────────────────────────────
def rgb_world2rgb_plane(points3d: np.ndarray):
    """
    Project 3-D RGB-world points onto the RGB image plane.
    Returns x_plane (col), y_plane (row) — both (N,)
    """
    X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    x_plane = (X * FX_RGB / Z) + CX_RGB
    y_plane = (Y * FY_RGB / Z) + CY_RGB
    return x_plane, y_plane


# ─────────────────────────────────────────────────────────────────────────────
# project_depth_map.m
# ─────────────────────────────────────────────────────────────────────────────
def project_depth_map(raw_depth_u16: np.ndarray,
                      rgb_u8: np.ndarray):
    """
    Full projection pipeline matching project_depth_map.m exactly.

    Args:
        raw_depth_u16 : (480, 640) uint16 — bytes ALREADY swapped
                        (load with PIL/cv2; Kinect raw .pgm needs byteswap)
        rgb_u8        : (480, 640, 3) uint8 RGB image

    Returns:
        depth_out     : (480, 640) float64 metric depth aligned to RGB, metres
        rgb_undist    : (480, 640, 3) uint8 undistorted RGB
    """
    H, W = raw_depth_u16.shape

    # Build OpenCV camera matrices
    K_rgb_mat = np.array([[FX_RGB, 0, CX_RGB],
                          [0, FY_RGB, CY_RGB],
                          [0,      0,      1]], dtype=np.float64)
    K_d_mat   = np.array([[FX_D,  0,  CX_D],
                          [0, FY_D,   CY_D],
                          [0,    0,      1]], dtype=np.float64)

    # ── Undistort RGB (channel-wise, matching MATLAB loop) ─────────────────
    rgb_undist = np.zeros_like(rgb_u8)
    for c in range(3):
        rgb_undist[:, :, c] = cv2.undistort(
            rgb_u8[:, :, c].astype(np.float32),
            K_rgb_mat, K_RGB
        ).astype(np.uint8)

    # ── Build noise mask (saturated Kinect pixels = 2047 raw value) ────────
    noise_mask = (raw_depth_u16 == raw_depth_u16.max()).astype(np.float32) * 255
    noise_mask = cv2.undistort(noise_mask, K_d_mat, K_D) > 0

    # ── Undistort depth image ───────────────────────────────────────────────
    depth_f = cv2.undistort(
        raw_depth_u16.astype(np.float32),
        K_d_mat, K_D
    ).astype(np.float64)

    # Fix artefacts introduced by distortion (matching MATLAB thresholds)
    depth_f[depth_f < 600]  = 2047
    depth_f[noise_mask]     = 2047

    # ── Linearise depth values ──────────────────────────────────────────────
    depth_abs = depth_rel2depth_abs(depth_f)

    # ── Project onto RGB plane via 3-D ─────────────────────────────────────
    pts3d         = depth_plane2depth_world(depth_abs)
    pts3d_rgb     = depth_world2rgb_world(pts3d)
    x_proj, y_proj = rgb_world2rgb_plane(pts3d_rgb)

    x_proj = np.round(x_proj).astype(int)
    y_proj = np.round(y_proj).astype(int)

    # ── Z-buffer projection (back-to-front: farthest written first) ────────
    flat_depth = depth_abs.ravel()
    good = (x_proj > 0) & (x_proj < W) & (y_proj > 0) & (y_proj < H)
    good_idx = np.where(good)[0]

    # Sort farthest→nearest so nearest wins (z-buffer)
    order = np.argsort(-flat_depth[good_idx])
    sorted_idx = good_idx[order]

    depth_out = np.zeros((H, W), dtype=np.float64)
    depth_out[y_proj[sorted_idx], x_proj[sorted_idx]] = flat_depth[sorted_idx]

    # Clip and sanitise
    depth_out = np.clip(depth_out, 0.0, MAX_DEPTH)
    depth_out[np.isnan(depth_out)] = 0.0

    return depth_out, rgb_undist


# ─────────────────────────────────────────────────────────────────────────────
# get_projection_mask.m + crop_image.m
# ─────────────────────────────────────────────────────────────────────────────
def get_projection_mask():
    """
    Returns boolean (480, 640) mask of valid projected pixels.
    Valid region: rows 44:471, cols 40:601 (0-indexed) → 427×561.
    """
    mask = np.zeros((480, 640), dtype=bool)
    mask[CROP_ROWS, CROP_COLS] = True
    return mask


def crop_image(img: np.ndarray) -> np.ndarray:
    """
    Crop to the valid 427×561 projection region.
    Accepts (H, W) or (H, W, C).
    """
    return img[CROP_ROWS, CROP_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# fill_depth_colorization.m  (Levin et al. colorization-based depth fill)
# ─────────────────────────────────────────────────────────────────────────────
def fill_depth_colorization(rgb: np.ndarray,
                             depth: np.ndarray,
                             alpha: float = 1.0) -> np.ndarray:
    """
    Dense depth completion using colorization-based propagation.
    Matches fill_depth_colorization.m exactly.

    Args:
        rgb   : (H, W, 3) float32 in [0, 1]
        depth : (H, W) float64 metric depth (0 = missing)
        alpha : penalty weight for known depth pixels (default 1.0)

    Returns:
        filled_depth : (H, W) float64, same scale as input depth
    """
    is_noise  = (depth == 0) | (depth == MAX_DEPTH)
    max_depth = depth[~is_noise].max()
    depth_n   = np.clip(depth / max_depth, 0, 1)

    H, W   = depth_n.shape
    N      = H * W
    known  = ~is_noise

    gray = (0.2989 * rgb[:, :, 0] +
            0.5870 * rgb[:, :, 1] +
            0.1140 * rgb[:, :, 2]).ravel()  # (N,)

    WIN = 1
    rows_buf, cols_buf, vals_buf = [], [], []

    for j in range(W):
        for i in range(H):
            idx = j * H + i   # column-major index (MATLAB compatible)
            nbrs, gvals = [], []

            for ii in range(max(0, i - WIN), min(H, i + WIN + 1)):
                for jj in range(max(0, j - WIN), min(W, j + WIN + 1)):
                    if ii == i and jj == j:
                        continue
                    nbr_idx = jj * H + ii
                    nbrs.append(nbr_idx)
                    gvals.append(gray[nbr_idx])

            cur_val = gray[idx]
            gvals_arr = np.array(gvals + [cur_val])
            c_var = np.mean((gvals_arr - gvals_arr.mean()) ** 2)

            csig = c_var * 0.6
            mgv  = np.min((np.array(gvals) - cur_val) ** 2) if gvals else 0.0
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)
            csig = max(csig, 2e-6)

            w = np.exp(-(np.array(gvals) - cur_val) ** 2 / csig)
            w /= w.sum() + 1e-12

            for nbr_idx, wi in zip(nbrs, w):
                rows_buf.append(idx);  cols_buf.append(nbr_idx);  vals_buf.append(-wi)
            rows_buf.append(idx);  cols_buf.append(idx);  vals_buf.append(1.0)

    A = sparse.csr_matrix(
        (vals_buf, (rows_buf, cols_buf)), shape=(N, N)
    )

    g_vals  = known.ravel().astype(float) * alpha
    G       = sparse.diags(g_vals, 0, shape=(N, N), format='csr')
    rhs     = g_vals * depth_n.ravel(order='F')

    solution = spsolve(A + G, rhs)
    filled   = solution.reshape(H, W, order='F') * max_depth
    return filled

# ─────────────────────────────────────────────────────────────────────────────
# fill_depth_cross_bf.m  (multi-scale cross bilateral filter)
# cbf.h defaults: num_scales=3, sigma_s=[12,5,8], sigma_r=[0.2,0.08,0.02]
# cbf_windows.h confirms hardcoded output size 427x561 (= CROP_SIZE)
# ─────────────────────────────────────────────────────────────────────────────
def fill_depth_cross_bf(rgb_u8: np.ndarray,
                        depth: np.ndarray,
                        space_sigmas=(12.0, 5.0, 8.0),
                        range_sigmas=(0.2, 0.08, 0.02)) -> np.ndarray:
    """
    Multi-scale cross bilateral filter depth inpainting.
    Port of fill_depth_cross_bf.m + cbf.cpp using cv2.ximgproc.

    Args:
        rgb_u8       : (H, W, 3) uint8
        depth        : (H, W) float64 metric depth (0 = missing)
        space_sigmas : spatial Gaussian sigmas per scale (default from cbf.h)
        range_sigmas : intensity range sigmas per scale (default from cbf.h)

    Returns:
        filled_depth : (H, W) float64, same scale as input

    Install: pip install opencv-contrib-python
    """
    assert len(space_sigmas) == len(range_sigmas), "sigma lists must match"

    is_noise   = (depth == 0) | (depth >= MAX_DEPTH)
    max_depth  = depth[~is_noise].max() if (~is_noise).any() else 1.0

    # Normalise to [0, 255] uint8 — matching MATLAB's im2uint8 step
    depth_norm = np.clip(depth / max_depth, 0, 1)
    depth_u8   = (depth_norm * 255).astype(np.uint8)
    gray_u8    = (0.2989 * rgb_u8[:, :, 0] +
                  0.5870 * rgb_u8[:, :, 1] +
                  0.1140 * rgb_u8[:, :, 2]).astype(np.uint8)

    result = depth_u8.copy()

    # Run one joint bilateral filter pass per scale
    for ss, rs in zip(space_sigmas, range_sigmas):
        # sigma_r is in [0,1] intensity space; cv2 uses [0,255] → scale up
        result = cv2.ximgproc.jointBilateralFilter(
            joint=gray_u8,
            src=result,
            d=-1,
            sigmaColor=rs * 255,
            sigmaSpace=ss
        )

    # Convert back to metric depth
    filled = result.astype(np.float64) / 255.0 * max_depth
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# rgb_plane2rgb_world.m — inverse projection (note −y sign flip vs depth cam)
# ─────────────────────────────────────────────────────────────────────────────
def rgb_plane2rgb_world(depth: np.ndarray) -> np.ndarray:
    """
    Unproject RGB-aligned depth back to 3D world coordinates using
    RGB intrinsics. Note the −Y sign convention matching the MATLAB source.

    Input : (H, W) float depth in metres (aligned to RGB frame)
    Output: (H*W, 3) array of (X, Y, Z) — Y is negated per MATLAB convention
    """
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(1, W + 1), np.arange(1, H + 1))
    x3 = (xx - CX_RGB) * depth / FX_RGB
    y3 = (yy - CY_RGB) * depth / FY_RGB
    z3 = depth
    return np.stack([x3.ravel(), -y3.ravel(), z3.ravel()], axis=1)  # −Y !


# ─────────────────────────────────────────────────────────────────────────────
# get_scene_type_from_scene.m — strip trailing number from scene name
# ─────────────────────────────────────────────────────────────────────────────
def get_scene_type(scene_name: str) -> str:
    """
    Extract scene type from scene directory name.
    e.g. 'bedroom_0002k' → 'bedroom',  'living_room_0013' → 'living_room'

    Matches MATLAB: finds first digit run, takes everything before it (−2 chars).
    """
    import re
    m = re.search(r'\d+\w?', scene_name)
    if m:
        return scene_name[:m.start() - 1]
    return scene_name