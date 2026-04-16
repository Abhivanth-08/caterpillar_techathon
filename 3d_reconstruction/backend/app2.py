"""
app2.py  –  FastAPI server using Open3D pipeline
Run:  uvicorn app2:app --host 0.0.0.0 --port 5001 --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Import from the Open3D version
from lidar_processing2 import run_pipeline

app = FastAPI(title="LiDAR Truck Bed Reconstruction (Open3D)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/lidar-data")
async def lidar_data():
    try:
        payload = run_pipeline()
        return payload
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  LiDAR Truck Bed Reconstruction  –  Open3D Backend  ")
    print("=" * 55)
    print("  API  : http://localhost:5001/lidar-data")
    print("  UI   : http://localhost:5001/")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=5001)
