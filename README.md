# Beirut Street-View - Urban Scene Parsing with Mapillary

Single-notebook pipeline that takes [Mapillary](https://www.mapillary.com)
street-level imagery for Beirut, Lebanon (2014-2024) and produces:

1. **CLIP embeddings** for every image - panoramas are **cubemap-unwrapped** into 4
   perspective views first so CLIP (which was trained on perspective imagery) can
   actually score them.
2. **Semantic segmentation** (SegFormer-Cityscapes) - per-image `vegetation_frac`,
   `sky_frac`, `road_frac`, `building_frac`, `terrain_frac`.
3. **Damage / suspicious buildings classifier** (Approach A) - CLIP + logistic
   regression, **sequence-aware 5-fold GroupKFold CV** (no near-duplicate frames
   leak across train/test), full ROC-AUC + KS test.
4. **Metadata-only Random Forest baseline** (Approach B) - lat / lon / altitude /
   time / multi-scale spatial density / sequence stats. **Sanity check: is CLIP
   actually adding signal over pure metadata?**
5. **Decision cell** - Pearson r + top-K overlap between Approach A and B, writes
   `outputs/decision.txt` with a plain-English verdict.
6. **3D model of the city** - interactive Plotly scene of every camera pose,
   damage-colored, top-500 suspicious highlighted.
7. **Greenery analysis** - city-wide green cover map, year-over-year mean
   `vegetation_frac`, and pre-vs-post-2020-08-04 histograms inside 1 km of the port.
8. **CLIP + FAISS visual-similarity search** - "find images like this one".
9. **3D dataset browser** - one self-contained HTML with 2D Leaflet map + 3D Plotly
   scene + sidebar search / filters + thumbnail gallery + CSV export button.
   No server, no install - double-click the HTML.

Everything auto-saves to `outputs/` (fig_NN.png for plots, individual CSVs and
TXTs for tables) so a runtime disconnect never costs you the results.

## Showcase

Real outputs from a full run of the notebook on 43,611 Mapillary street-view images of Beirut.

### Damage / suspicious buildings - CLIP + logistic regression

Sequence-aware 5-fold GroupKFold cross-validation on the CLIP-embedding classifier.

![Confusion matrix (5-fold OOF, group by sequence). AUC=0.901, KS=0.666](docs/screenshots/01_confusion_matrix.png)

Top-12 images ranked most damage-like by the classifier. Every one shows a real
Beirut building with visible damage, rubble, or post-blast state.

![Top-12 suspicious buildings](docs/screenshots/02_top12_damaged.png)

Geographic distribution: all 43,611 image poses (blue) with the top-500
damage-like frames overlaid in red. The cluster north-east of the port matches
the 4 August 2020 blast zone.

![Geographic distribution of suspicious images](docs/screenshots/03_damage_map.png)

### Greenery analysis - SegFormer-Cityscapes

Per-image green fraction (vegetation + terrain) rendered as a city-wide map,
with the top-15 greenest spots circled in black and the Beirut port marked with
a yellow star.

![Beirut greenery map (n=43,611 images, SegFormer-Cityscapes)](docs/screenshots/04_greenery_map.png)

Year-over-year mean and median of `green_frac` across the whole city.

![Beirut green cover over time (per-image mean)](docs/screenshots/05_green_cover_time.png)

`green_frac` distribution inside 1 km of the port, split at 4 August 2020 -
post-blast frames end up greener on average because rubble lots have been
partially re-vegetated.

![Green cover inside 1 km of Beirut port: pre vs post 2020-08-04](docs/screenshots/06_pre_post_port_histogram.png)

### 3D dataset browser (Step 7 - the final dashboard)

The last cell of the notebook produces a **self-contained HTML page**
(`outputs/beirut_browser.html`) that combines a 2D Leaflet map, a 3D Plotly
scene, a sidebar with live filters (text search on `image_id`/`sequence`, date
range, damage-score slider, green-fraction slider, panoramas-only toggle), a
right-side thumbnail gallery of the current filter, and an "Export filtered
-> CSV" button. It ships as one HTML + one JSON file next to `images_2048/`
and opens with a double-click - no server, no install. See
[Which dataset browser to use](#which-dataset-browser-to-use) below.

## Files

| File | Purpose |
|---|---|
| `beirut_analysis.ipynb` | The entire project - fetch, download, embed, segment, classify, compare, browse |
| `opensfm_demo.md` | Real photogrammetric 3D reconstruction (Docker + OpenSfM) for one sequence at a time |
| `.gitignore` | Ignores the generated artifacts (metadata CSV, images, embeddings, segmentation cache, outputs/, HTML) |

Everything the project does happens in the notebook. The metadata CSV, the
image files, the embedding index, the segmentation cache, and `outputs/` are
all **generated** - nothing is checked in.

## Setup

```bash
pip install requests pandas numpy pillow matplotlib tqdm torch open_clip_torch faiss-cpu plotly scikit-learn scipy transformers huggingface_hub
```

Colab already has most of these; the notebook `!pip install`s `open_clip_torch`
and `faiss-cpu` automatically on first use.

## Paths

The notebook now uses a **generic path** that works everywhere:

```python
IMG_DIR = Path(os.environ.get('IMG_DIR_OVERRIDE', 'images_2048')).resolve()
```

- On Colab: resolves to `/content/images_2048`
- On any other machine: `images_2048/` next to the notebook
- Override anywhere: `export IMG_DIR_OVERRIDE=/path/to/my/images` (or the PowerShell
  equivalent) before launching the kernel.

## Tokens (both mandatory, both hidden input)

The setup cell prompts for two tokens on first run. Both are read via `getpass`
(hidden input), cached in the kernel session, and never written to disk.

- **Mapillary token** (READ scope). Get one at
  https://www.mapillary.com/dashboard/developers. Used by Step 0 to fetch image
  metadata.
- **HuggingFace token** (READ scope). Get one at
  https://huggingface.co/settings/tokens. Used by Steps 2 and 2b when
  downloading `open_clip` and SegFormer weights. Anonymous HF downloads get
  rate-limited on shared IPs (Colab, university networks, VPNs), which silently
  stalls the run for hours - so the token is mandatory even for public models.

To skip the prompt in scripted runs, export them first:

```bash
export MAPILLARY_TOKEN='MLY|...'
export HF_TOKEN='hf_...'
```

## Persistence and skip-if-already-there

The setup cell picks the right base directory automatically and re-uses whatever
is already on disk. Nothing is downloaded, embedded, or segmented twice.

| Environment | Base dir | Survives runtime restart? |
|---|---|---|
| Local machine | current dir | yes (it's your disk) |
| Colab / Kaggle | `MyDrive/beirut_project/` (Drive auto-mounted) | yes (persisted to Drive) |
| Override anywhere | `export BEIRUT_BASE_DIR=/some/path` | yes |

On startup the setup cell prints exactly what is already cached
(`metadata`, `embeddings`, `segmentation`, JPEG count) and each heavy step
short-circuits on the same check:

- **Step 0** (metadata): skipped entirely if `beirut_metadata.csv.gz` and
  `embeddings.npy` both exist (re-fetching would drift IDs vs. the embeddings).
- **Step 1** (JPEGs): only downloads IDs not already in `images_2048/`.
- **Step 2** (CLIP embeddings): only embeds IDs not already in
  `embeddings_ids.txt`; saves after every batch.
- **Step 2b** (SegFormer): only segments IDs not already in
  `segmentation_scores.csv`; saves after every batch.
- **Model weights** (CLIP ~300 MB, SegFormer ~14 MB): HF and torch caches are
  redirected into the base dir, so on Colab they land on Drive and download
  once ever.

First full run on Colab T4: ~1 h. Every subsequent run: seconds to a few minutes
(only the classifier / plots / browser cells re-execute).

## Running the notebook

Open `beirut_analysis.ipynb` in Jupyter / Colab and run cells top to bottom.

```
Step 0  - Fetch image metadata from Mapillary    (~30 s, parallel API calls)
Step 1  - Download the actual JPEGs              (~15 min on Colab; resumable)
Step 2  - CLIP embeddings, panos cubemap-unwrap  (~15 min on T4; resumable)
Step 2b - SegFormer semantic segmentation        (~30 min on T4; resumable)
Step 3  - Approach A (CLIP + LR) with 5-fold OOF
Step 3b - Approach B (metadata-only RF) with 5-fold OOF
Step 3c - Decision (A vs B) → outputs/decision.txt
Step 4  - Interactive 3D Plotly scene            → beirut_3d.html
Step 5  - CLIP + FAISS visual-similarity search
Step 6  - Greenery analysis (map + time + pre/post)
Step 7  - 3D dataset browser                     → outputs/beirut_browser.html
```

Steps 1, 2, and 2b are **resume-safe and incremental**: re-running picks up only
new images, and each of them saves partial progress after every batch so you can
interrupt at any time without losing work.

## What ends up in `outputs/`

```
outputs/
├── fig_01.png … fig_NN.png     (every matplotlib figure, auto-saved)
├── classifier_metrics.txt      (accuracy, precision, recall, F1, AUC, KS)
├── confusion_matrix.csv
├── scored_train.csv            (per-image OOF CLIP damage score)
├── feature_importance.csv      (Approach B RF feature importances)
├── comparison_metrics.txt / .csv  (A vs B per-image scores)
├── decision.txt                (A vs B verdict)
├── top_15_greenest.csv         (top 15 greenest spots in Beirut)
├── beirut_dataset_index.csv    (Excel-friendly master index with file:// links)
├── browser_data.json           (payload for the standalone HTML browser)
└── beirut_browser.html         (drop next to images_2048/ and double-click)
```

## BBox (central Beirut)

```
WEST = 35.470   SOUTH = 33.860   EAST = 35.570   NORTH = 33.920
```

Covers from the Mediterranean coast east through downtown, including the port,
Hamra, Achrafieh, and southern suburbs.

## Real photogrammetric 3D

The notebook's Step 4 gives a true-to-scale scene of every camera pose. For
**actual building geometry** - dense point clouds, mesh of facades, comparable
surfaces pre/post Aug 2020 - see [`opensfm_demo.md`](opensfm_demo.md):
Docker-based OpenSfM on one capture sequence at a time, ~30 min per sequence on CPU.

## Which dataset browser to use

Step 7 writes both an HTML browser and a CSV index. They're complementary:

- **`beirut_browser.html`** - open in any browser (needs `browser_data.json` and
  the `images_2048/` folder next to it). Best for **exploration, demos, and
  hand-off to non-technical viewers**. Live filters, live thumbnail preview,
  live 3D scene, CSV export of whatever you're currently looking at.
- **`beirut_dataset_index.csv`** - open in Excel/LibreOffice/pandas. Best for
  **pivoting, joining with other data, or feeding another pipeline**. Has every
  column including clickable `file://` URLs.

Short recommendation: **use the HTML for browsing, use the CSV for analysis**.

## License

Mapillary images are licensed under **CC-BY-SA 4.0**.
Attribution required when publishing: "Imagery © Mapillary contributors".
