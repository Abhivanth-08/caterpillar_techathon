"""
app.py  –  FastAPI server for LiDAR truck bed reconstruction
Run:  uvicorn app:app --host 0.0.0.0 --port 5000 --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from lidar_processing import run_pipeline

app = FastAPI(title="LiDAR Truck Bed Reconstruction")

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────

@app.get("/lidar-data")
async def lidar_data():
    """
    GET /lidar-data
    Returns fresh point cloud + mesh + stats as JSON.
    """
    try:
        payload = run_pipeline()
        return payload
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount static files AFTER the explicit routes
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  LiDAR Truck Bed Reconstruction  –  FastAPI")
    print("=" * 55)
    print("  API  : http://localhost:5000/lidar-data")
    print("  UI   : http://localhost:5000/")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=5000)
