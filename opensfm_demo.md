# OpenSfM 3D reconstruction - one-sequence demo

This is the heavy 3D layer: a real photogrammetric reconstruction (sparse +
dense point cloud, optional mesh) of a portion of Beirut from the Mapillary
street-view images. We use [OpenSfM](https://github.com/mapillary/OpenSfM),
which is what Mapillary itself uses internally - so the image format and
metadata (GPS, heading, altitude) are already in the right shape.

The lightweight 3D layer (`view_3d.py`) is enough for camera-pose maps. Use
this OpenSfM path when you need actual building geometry - e.g., to estimate
volume of rubble at the port, or to compare façade depth pre/post explosion.

## Why a single sequence

Full-city SfM over 31k images would take days on CPU and tens of GB of disk.
Each sequence is a continuous capture path (one camera moving), which is the
unit OpenSfM reconstructs best. A 50–300 image sequence reconstructs in 20–90
min on CPU and produces a usable point cloud of one block / one street.

Pick a sequence that has good coverage near your area of interest - e.g.
near the port for the Aug 2020 explosion analysis.

## Choose a sequence

```python
import pandas as pd
df = pd.read_csv("beirut_metadata.csv.gz")
# sequences near the port, ordered by image count
PORT_LAT, PORT_LON = 33.9020, 35.5180
mask = (df["lat"].between(PORT_LAT - 0.005, PORT_LAT + 0.005) &
        df["lon"].between(PORT_LON - 0.005, PORT_LON + 0.005))
print(df[mask].groupby("sequence").size().sort_values(ascending=False).head(10))
```

Pick the sequence id with 80–300 images and copy those JPEGs into a working
folder (one folder per reconstruction).

```python
SEQ = "PASTE_SEQUENCE_ID_HERE"
import shutil
from pathlib import Path
work = Path("opensfm_work/images")
work.mkdir(parents=True, exist_ok=True)
for img_id in df[df["sequence"] == SEQ]["id"]:
    src = Path(r"E:\לימודים\Data Science\Projects\Mapillary_Beirut\images_2048") / f"{img_id}.jpg"
    if src.exists():
        shutil.copy(src, work / f"{img_id}.jpg")
print(f"copied {len(list(work.iterdir()))} images")
```

## Install OpenSfM (Docker is easiest on Windows)

OpenSfM has heavy C++ dependencies (Ceres, OpenCV, GFLAGS). The official
Docker image avoids the Windows build pain.

```powershell
docker pull mapillary/opensfm

# Mount the work folder and run reconstruction
docker run -it --rm -v "${PWD}/opensfm_work:/data" mapillary/opensfm `
    bin/opensfm_run_all /data
```

The whole pipeline runs: `extract_metadata`, `detect_features`,
`match_features`, `create_tracks`, `reconstruct`, `mesh`, `undistort`,
`compute_depthmaps`, `export_ply`.

## Inject the Mapillary EXIF / pose

OpenSfM auto-reads EXIF, but Mapillary thumbnails sometimes strip it. To
guarantee a clean reconstruction, write a `camera_models_overrides.json`
and `exif_overrides.json` in `opensfm_work/`:

```python
import json
from pathlib import Path
import pandas as pd

df = pd.read_csv("beirut_metadata.csv.gz")
df = df[df["sequence"] == SEQ].set_index("id")

exif_overrides = {}
for img_id, row in df.iterrows():
    exif_overrides[f"{img_id}.jpg"] = {
        "gps": {
            "latitude":  float(row["lat"]),
            "longitude": float(row["lon"]),
            "altitude":  float(row["altitude"] if pd.notna(row["altitude"]) else row["computed_altitude"]),
            "dop":       5.0,
        },
        "orientation": 1,
        "capture_time": int(row["captured_at"] / 1000),
        "camera": "perspective" if not row["is_pano"] else "spherical",
    }
Path("opensfm_work/exif_overrides.json").write_text(json.dumps(exif_overrides, indent=2))
```

For panoramic sequences OpenSfM handles equirectangular projection natively
(`projection_type: spherical`).

## Inspect the result

After `bin/opensfm_run_all` finishes, you'll have:

- `opensfm_work/reconstruction.json` - camera poses + sparse 3D points
- `opensfm_work/undistorted/depthmaps/` - per-image depth maps
- `opensfm_work/undistorted/depthmaps/merged.ply` - fused dense point cloud
- `opensfm_work/undistorted/openmvs/scene_dense_mesh.ply` - textured mesh

Open `merged.ply` in [MeshLab](https://www.meshlab.net) or
[CloudCompare](https://www.cloudcompare.org). For a Python-only viewer:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("opensfm_work/undistorted/depthmaps/merged.ply")
o3d.visualization.draw_geometries([pcd])
```

## Scaling up beyond one sequence

To stitch multiple sequences (e.g., reconstruct a neighborhood):
1. Reconstruct each sequence independently.
2. Use OpenSfM's `merge` command or align them in CloudCompare via GPS.
3. For deep-learning–accelerated reconstruction, swap to
   [hloc + COLMAP](https://github.com/cvg/Hierarchical-Localization) or
   [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Recommended pre/post-explosion split

```python
EXPLOSION = pd.Timestamp("2020-08-04")
pre  = df[df["captured_at_dt"] <  EXPLOSION]
post = df[df["captured_at_dt"] >= EXPLOSION]
```

Run reconstruction once on each side, then compare point clouds in
CloudCompare ("M3C2 distance" plugin) - surfaces that moved or disappeared
flag damage.
