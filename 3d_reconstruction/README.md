# LiDAR Truck Bed Reconstruction

A realistic LiDAR-based truck bed reconstruction pipeline visualised in a
browser as a driver-assist system dashboard.

```
lidar-truck/
├── backend/
│   ├── app.py               ← Flask API server
│   ├── lidar_processing.py  ← Point cloud pipeline
│   └── requirements.txt
└── frontend/
    ├── index.html           ← Dashboard UI
    ├── script.js            ← Three.js renderer
    └── style.css            ← Industrial dark theme
```

---

## 1 — Install dependencies

```bash
cd backend

# Core (required)
pip install flask flask-cors numpy

# Optional — enables full Open3D pipeline (noise removal, RANSAC, Poisson mesh)
# Falls back to numpy-only if not installed
pip install open3d
```

---

## 2 — Start the Flask server

```bash
cd backend
python app.py
```

You should see:
```
=======================================================
  LiDAR Truck Bed Reconstruction  –  Flask API
=======================================================
  API  : http://localhost:5000/lidar-data
  UI   : http://localhost:5000/
=======================================================
```

---

## 3 — Open the browser

Navigate to:  **http://localhost:5000/**

The dashboard will:
- Load a fresh point cloud + mesh every **4 seconds** (simulated LiDAR stream)
- Render both a coloured point cloud and a reconstructed surface mesh
- Show volume, load balance, and point counts in the side panel
- Allow OrbitControls (left-drag = rotate, right-drag = pan, scroll = zoom)
- Let you toggle between BOTH / CLOUD / MESH views

---

## Pipeline overview

```
generate_truck_bed_pointcloud()
        │
        ▼
remove_statistical_outlier()   ← noise removal
        │
        ▼
voxel_down_sample()            ← downsampling
        │
        ▼
segment_plane() RANSAC          ← truck floor extraction
        │
        ├─► floor_cloud
        └─► load_cloud
                │
                ▼
        alpha_shape / grid mesh ← surface reconstruction
                │
                ▼
        volume integration      ← approx m³
                │
                ▼
        JSON via /lidar-data    ← Three.js renders it
```

---

## Tested with

- Python 3.9 +
- Flask 3.x
- open3d 0.18  (optional)
- Three.js r128 (loaded from CDN)
