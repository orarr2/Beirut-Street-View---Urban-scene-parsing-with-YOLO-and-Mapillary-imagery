# Beirut Street-View — Urban Scene Parsing with Mapillary

Single-notebook pipeline that takes [Mapillary](https://www.mapillary.com)
street-level imagery for Beirut, Lebanon and produces three things:

1. **Suspicious / damaged buildings** — supervised classifier trained on the
   Aug 4 2020 port explosion as weak supervision (stratified 80/20 train/test split,
   logistic regression on CLIP features)
2. **3D model of the city** — interactive 3D scene of every camera pose,
   colored by capture year, with the top suspicious images highlighted
3. **Closest places to a given picture** — CLIP + FAISS visual-similarity search

The dataset spans **2014–2024**, so it's useful for change-detection analyses
including the August 2020 port explosion.

## Files

| File | Purpose |
|---|---|
| `beirut_analysis.ipynb` | The entire project — fetch metadata, download images, embed, train, score, visualize |
| `opensfm_demo.md` | How to run real photogrammetric 3D reconstruction (Docker + OpenSfM) |
| `.gitignore` | Ignores the generated artifacts (metadata CSV, images, embeddings, models, HTML) |

Everything the project does happens in the notebook. The metadata CSV, the
image files, and the embedding index are all **generated** by the notebook —
nothing is checked in.

## Setup

```bash
pip install requests pandas numpy pillow matplotlib torch open_clip_torch faiss-cpu plotly scikit-learn
```

## Mapillary token

Get a free token (READ scope) at https://www.mapillary.com/dashboard/developers.
The notebook prompts for it in a **hidden input field** the first time you
run it — never typed in plaintext, never saved to disk.

## Running the notebook

Open `beirut_analysis.ipynb` in Jupyter / JupyterLab and run cells top to bottom.

```
Step 0 — Fetch image metadata from Mapillary  (~minutes, parallel API calls)
Step 1 — Download the actual JPEGs            (~hours first time, resumable)
Step 2 — CLIP image embeddings                (~hours on CPU, ~minutes on GPU; resumable)
Step 3 — Suspicious / damaged buildings       (train + test split + metrics)
Step 4 — 3D model of the city                 (interactive plotly HTML)
Step 5 — Closest places to a given picture    (FAISS similarity search)
```

The image path is set at the top of cell 2 — edit it to wherever you want
images stored:

```python
IMG_DIR = Path(r'E:\לימודים\Data Science\Projects\Mapillary_Beirut\images_2048')
```

Steps 1 and 2 are **resume-safe and incremental** — re-running picks up only
new images, and Step 2 saves after every batch so you can interrupt at any
time without losing work.

## BBox (central Beirut)

```
WEST = 35.470   SOUTH = 33.860   EAST = 35.570   NORTH = 33.920
```

Covers from the Mediterranean coast east through downtown, including the port,
Hamra, Achrafieh, and southern suburbs.

## Real photogrammetric 3D

The notebook's Step 4 gives a true-to-scale scene of every camera pose. For
**actual building geometry** — dense point clouds, mesh of facades, comparable
surfaces pre/post Aug 2020 — see [`opensfm_demo.md`](opensfm_demo.md):
Docker-based OpenSfM on one capture sequence at a time, ~30 min per sequence on CPU.

## License

Mapillary images are licensed under **CC-BY-SA 4.0**.
Attribution required when publishing: "Imagery © Mapillary contributors".
