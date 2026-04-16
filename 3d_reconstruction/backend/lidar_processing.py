"""
lidar_processing.py
Synthetic LiDAR point cloud generation and processing for truck bed reconstruction.
Optimised for speed — uses NumPy vectorised ops + SciPy Delaunay.
"""

import numpy as np
import time

try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────
# 1. Synthetic point cloud generation
# ─────────────────────────────────────────────

def generate_truck_bed_pointcloud():
    """
    Generate a synthetic LiDAR scan of a truck bed with an uneven load.
    Returns an (N, 3) numpy array of XYZ points.
    """
    rng = np.random.default_rng()
    points = []

    # --- Flat truck bed floor (z ≈ 0) ---
    n_floor = 800
    points.append(np.column_stack([
        rng.uniform(-2.0, 2.0, n_floor),
        rng.uniform(-1.2, 1.2, n_floor),
        rng.uniform(-0.02, 0.02, n_floor),
    ]))

    # --- Side walls ---
    for x_val in [-2.0, 2.0]:
        n = 200
        points.append(np.column_stack([
            np.full(n, x_val) + rng.normal(0, 0.01, n),
            rng.uniform(-1.2, 1.2, n),
            rng.uniform(0.0, 0.6, n),
        ]))

    # --- Front / Back walls ---
    for y_val in [-1.2, 1.2]:
        n = 200
        points.append(np.column_stack([
            rng.uniform(-2.0, 2.0, n),
            np.full(n, y_val) + rng.normal(0, 0.01, n),
            rng.uniform(0.0, 0.6, n),
        ]))

    # --- Uneven load surface ---
    phase = time.time() * 0.3
    n_load = 1500
    x_load = rng.uniform(-1.6, 1.6, n_load)
    y_load = rng.uniform(-0.9, 0.9, n_load)

    z_load = (
        0.35 * np.exp(-((x_load + 0.6)**2 + (y_load - 0.3)**2) / 0.5)
      + 0.55 * np.exp(-((x_load - 0.5)**2 + (y_load + 0.2)**2) / 0.4)
      + 0.20 * np.exp(-((x_load + 1.0)**2 + (y_load + 0.5)**2) / 0.3)
      + 0.10 * np.sin(x_load * 2.5 + phase) * np.cos(y_load * 2.5)
      + rng.normal(0, 0.015, n_load)
    )
    z_load = np.clip(z_load, 0.03, 0.65)
    points.append(np.column_stack([x_load, y_load, z_load]))

    return np.vstack(points).astype(np.float32)


# ─────────────────────────────────────────────
# 2. Point cloud processing (fully vectorised)
# ─────────────────────────────────────────────

def process_pointcloud(raw_points: np.ndarray):
    """Clean, downsample, and segment. Returns (all_pts, load_pts)."""
    pts = raw_points.copy()

    # Statistical outlier removal (vectorised)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0) + 1e-8
    mask = np.all(np.abs(pts - mean) < 3.0 * std, axis=1)
    pts = pts[mask]

    # Voxel-grid downsampling (0.05 m) — vectorised
    voxel = 0.05
    keys = np.floor(pts / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    pts = pts[idx]

    # Fast RANSAC — only 100 iterations needed for synthetic data
    best_inliers = np.array([], dtype=int)
    rng = np.random.default_rng()
    n_pts = len(pts)

    for _ in range(100):
        sample_idx = rng.choice(n_pts, 3, replace=False)
        sample = pts[sample_idx]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal /= norm_len
        d = -normal @ sample[0]
        dist = np.abs(pts @ normal + d)
        inliers = np.where(dist < 0.04)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Use boolean mask instead of set operations (much faster)
    is_floor = np.zeros(n_pts, dtype=bool)
    if len(best_inliers) > 0:
        is_floor[best_inliers] = True
    load_pts = pts[~is_floor]

    return pts, load_pts


# ─────────────────────────────────────────────
# 3. Surface mesh generation
# ─────────────────────────────────────────────

def build_mesh(load_pts: np.ndarray):
    """Build mesh from load points using Delaunay triangulation."""
    if len(load_pts) < 4:
        return [], []

    # SciPy Delaunay — fast C implementation
    if SCIPY_AVAILABLE:
        try:
            tri = Delaunay(load_pts[:, :2])
            return load_pts.tolist(), tri.simplices.tolist()
        except Exception:
            pass

    # Fallback: simple grid mesh (vectorised)
    return _fast_grid_mesh(load_pts)


def _fast_grid_mesh(pts, grid_res=25):
    """Fast grid mesh — reduced resolution, vectorised IDW."""
    if len(pts) < 3:
        return [], []

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    xx, yy = np.meshgrid(xi, yi)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    # Vectorised IDW: compute all distances at once
    # Shape: (grid_points, data_points)
    dx = flat_x[:, None] - x[None, :]  # (G, N)
    dy = flat_y[:, None] - y[None, :]  # (G, N)
    d2 = dx**2 + dy**2 + 1e-6
    w = 1.0 / d2
    zz = (w * z[None, :]).sum(axis=1) / w.sum(axis=1)

    verts = np.column_stack([flat_x, flat_y, zz]).tolist()

    faces = []
    for i in range(grid_res - 1):
        for j in range(grid_res - 1):
            a = i * grid_res + j
            b = a + 1
            c = (i + 1) * grid_res + j
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])

    return verts, faces


# ─────────────────────────────────────────────
# 4. Volume estimation (vectorised)
# ─────────────────────────────────────────────

def compute_volume(load_pts: np.ndarray) -> float:
    """Approximate volume above z=0 plane using histogram binning."""
    if len(load_pts) < 10:
        return 0.0

    x, y, z = load_pts[:, 0], load_pts[:, 1], load_pts[:, 2]
    grid_res = 20  # reduced from 30

    # Use numpy histogram2d for fast binning
    z_sum, x_edges, y_edges = np.histogram2d(
        x, y, bins=grid_res, weights=z
    )
    z_count, _, _ = np.histogram2d(x, y, bins=grid_res)

    # Average height per cell
    mask = z_count > 0
    z_avg = np.zeros_like(z_sum)
    z_avg[mask] = z_sum[mask] / z_count[mask]

    dx = (x.max() - x.min()) / grid_res
    dy = (y.max() - y.min()) / grid_res

    volume = float(z_avg.sum() * dx * dy)
    return round(volume, 4)


def compute_load_balance(load_pts: np.ndarray) -> dict:
    """Return load balance metrics."""
    if len(load_pts) < 5:
        return {"cx": 0.0, "cy": 0.0, "balance_x": "Centre", "balance_y": "Centre"}

    weights = np.maximum(load_pts[:, 2], 0.01)  # avoid zero weights
    cx = float(np.average(load_pts[:, 0], weights=weights))
    cy = float(np.average(load_pts[:, 1], weights=weights))

    def label(v, axis):
        if abs(v) < 0.25:
            return "Centre"
        side = ("Left" if v < 0 else "Right") if axis == "x" else ("Front" if v < 0 else "Rear")
        severity = "Slightly " if abs(v) < 0.55 else "Heavily "
        return severity + side

    return {
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "balance_x": label(cx, "x"),
        "balance_y": label(cy, "y"),
    }


# ─────────────────────────────────────────────
# 5. Top-level pipeline
# ─────────────────────────────────────────────

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 3  # seconds

def run_pipeline() -> dict:
    """Run full pipeline. Returns JSON-serialisable dict. Cached for 3s."""
    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    t0 = time.perf_counter()

    raw = generate_truck_bed_pointcloud()
    all_pts, load_pts = process_pointcloud(raw)
    verts, faces = build_mesh(load_pts)
    volume = compute_volume(load_pts)
    balance = compute_load_balance(load_pts)

    # Subsample for browser (≤ 3000 pts for speed)
    if len(all_pts) > 3000:
        idx = np.random.choice(len(all_pts), 3000, replace=False)
        all_pts = all_pts[idx]

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "timestamp": now,
        "backend": "scipy" if SCIPY_AVAILABLE else "numpy",
        "pipeline_ms": elapsed,
        "points": {
            "positions": all_pts.tolist(),
            "count": int(len(all_pts)),
        },
        "mesh": {
            "vertices": verts,
            "faces": faces,
        },
        "stats": {
            "volume_m3": volume,
            "load_point_count": int(len(load_pts)),
            "total_points": int(len(all_pts)),
            **balance,
        },
    }

    _cache["data"] = result
    _cache["ts"] = now
    print(f"  Pipeline: {elapsed}ms | {len(all_pts)} pts | {len(faces)} faces")
    return result
