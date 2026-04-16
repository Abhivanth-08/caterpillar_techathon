"""
lidar_processing2.py
Synthetic point cloud processing using true Open3D logic.
"""

import numpy as np
import time
import open3d as o3d

# ─────────────────────────────────────────────
# 1. Synthetic point cloud generation
# ─────────────────────────────────────────────

def generate_truck_bed_pointcloud():
    """Generate synthetic LiDAR points."""
    rng = np.random.default_rng()
    points = []

    # --- Flat truck bed floor (z ≈ 0) ---
    n_floor = 1000
    points.append(np.column_stack([
        rng.uniform(-2.0, 2.0, n_floor),
        rng.uniform(-1.2, 1.2, n_floor),
        rng.uniform(-0.02, 0.02, n_floor),
    ]))

    # --- Side walls ---
    for x_val in [-2.0, 2.0]:
        n = 250
        points.append(np.column_stack([
            np.full(n, x_val) + rng.normal(0, 0.01, n),
            rng.uniform(-1.2, 1.2, n),
            rng.uniform(0.0, 0.6, n),
        ]))

    # --- Front / Back walls ---
    for y_val in [-1.2, 1.2]:
        n = 250
        points.append(np.column_stack([
            rng.uniform(-2.0, 2.0, n),
            np.full(n, y_val) + rng.normal(0, 0.01, n),
            rng.uniform(0.0, 0.6, n),
        ]))

    # --- Uneven load surface ---
    phase = time.time() * 0.3
    n_load = 1800
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

    return np.vstack(points).astype(np.float64)


# ─────────────────────────────────────────────
# 2. Open3D Processing Pipeline
# ─────────────────────────────────────────────

def process_pointcloud_open3d(raw_points: np.ndarray):
    """
    Use Open3D APIs for statistical outlier removal, downsampling,
    RANSAC plane segmentation, and Alpha Shape surface reconstruction.
    """
    # 1. Load into Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)

    # 2. Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)

    # 3. Voxel Downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    all_pts = np.asarray(pcd.points)

    # 4. RANSAC Plane Segmentation (find the floor)
    # Reduced iterations from 1000 to 200 for web responsiveness
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.04,
        ransac_n=3,
        num_iterations=200
    )
    
    # 5. Separate Floor vs Cargo
    load_cloud = pcd.select_by_index(inliers, invert=True)
    load_pts = np.asarray(load_cloud.points)

    # 6. Surface Mesh Generation (Alpha Shapes)
    verts, faces = [], []
    if len(load_pts) >= 4:
        try:
            # Estimate normals needed for surface reconstruction
            load_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
            )
            # Reconstruct surface using Alpha Shape
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                load_cloud, alpha=0.3
            )
            mesh.compute_vertex_normals()
            verts = np.asarray(mesh.vertices).tolist()
            faces = np.asarray(mesh.triangles).tolist()
        except Exception as e:
            print(f"Open3D Mesh Error: {e}")

    return all_pts, load_pts, verts, faces


# ─────────────────────────────────────────────
# 3. Volume estimation & Analytics
# ─────────────────────────────────────────────

def compute_volume(load_pts: np.ndarray) -> float:
    if len(load_pts) < 10:
        return 0.0

    x, y, z = load_pts[:, 0], load_pts[:, 1], load_pts[:, 2]
    grid_res = 20
    
    z_sum, x_edges, y_edges = np.histogram2d(x, y, bins=grid_res, weights=z)
    z_count, _, _ = np.histogram2d(x, y, bins=grid_res)

    mask = z_count > 0
    z_avg = np.zeros_like(z_sum)
    z_avg[mask] = z_sum[mask] / z_count[mask]

    dx = (x.max() - x.min()) / grid_res
    dy = (y.max() - y.min()) / grid_res

    volume = float(z_avg.sum() * dx * dy)
    return round(volume, 4)

def compute_load_balance(load_pts: np.ndarray) -> dict:
    if len(load_pts) < 5:
        return {"cx": 0.0, "cy": 0.0, "balance_x": "Centre", "balance_y": "Centre"}

    weights = np.maximum(load_pts[:, 2], 0.01)
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
# 4. Top-level pipeline
# ─────────────────────────────────────────────

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 3  

def run_pipeline() -> dict:
    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    t0 = time.perf_counter()

    # 1. Generate Points
    raw = generate_truck_bed_pointcloud()
    
    # 2. Run Open3D Core Logic
    all_pts, load_pts, verts, faces = process_pointcloud_open3d(raw)
    
    # 3. Calculate Stats
    volume = compute_volume(load_pts)
    balance = compute_load_balance(load_pts)

    # Subsample for browser UI
    if len(all_pts) > 3000:
        idx = np.random.choice(len(all_pts), 3000, replace=False)
        all_pts = all_pts[idx]

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "timestamp": now,
        "backend": "open3d",
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
    print(f"  Open3D Pipeline: {elapsed}ms | {len(all_pts)} pts | {len(faces)} faces")
    return result
