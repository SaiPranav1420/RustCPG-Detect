# 🚀 How to Access & Run RustCPG-Detect
### Complete Setup Guide — Kaggle, Colab, and Local

---

## Option 1 — Run on Kaggle (RECOMMENDED — Free GPU, Dataset Already There)

This is the easiest method. The dataset lives on Kaggle, so no downloads needed.

### Step 1 — Go to the Dataset
👉 **https://www.kaggle.com/datasets/kaarthikeyaganji/cid-i2**

### Step 2 — Open a Notebook on Kaggle
1. Click the blue **"New Notebook"** button on the dataset page
2. A Kaggle notebook editor opens with the dataset already attached

### Step 3 — Upload Your Notebook File
1. In the Kaggle notebook editor, click **File → Import Notebook**
2. Upload one of these `.ipynb` files from this repo:
   - `notebooks/03_rustcpg_detect_pipeline.ipynb` ← **Start here (main pipeline)**
   - `notebooks/02_baseline_rust_ir_bert.ipynb`   ← Base paper reproduction
   - `notebooks/04_gnn_ablation_study.ipynb`       ← GNN ablation
   - `notebooks/05_results_and_comparison.ipynb`   ← All results + plots

### Step 4 — Enable GPU
1. On the right sidebar, click **"Session options"**
2. Under Accelerator, select **GPU T4 x2** or **P100**
3. Click **Save**

### Step 5 — Fix the Dataset Path
In the first code cell that loads the dataset, the path should already be correct:
```python
dataset = torch.load(
    "/kaggle/input/cid-i2/embedded_dataset_balanced.pt",
    weights_only=False
)
```
If you get a path error, click the **"+ Add Data"** button on the right sidebar,
search for `cid-i2`, and add the dataset. Then check the exact path shown.

### Step 6 — Run All Cells
Click **Run All** (▶▶) or press `Shift+Enter` through each cell.

**Expected runtimes on Kaggle GPU:**
| Notebook | Runtime |
|---|---|
| 02 — Base Paper | ~20 minutes |
| 03 — Full Pipeline | ~45 minutes |
| 04 — GNN Ablation | ~60 minutes |
| 05 — Results | ~10 minutes |

---

## Option 2 — Run on Google Colab

### Step 1 — Mount Google Drive and Upload Dataset
1. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d kaarthikeyaganji/cid-i2
   ```
2. Unzip and upload `embedded_dataset_balanced.pt` to your Google Drive at:
   ```
   MyDrive/CompilerProject/embedded_dataset_balanced.pt
   ```

### Step 2 — Open Notebook in Colab
1. Go to **https://colab.research.google.com**
2. Click **File → Upload Notebook**
3. Upload your `.ipynb` file

### Step 3 — Enable GPU
1. Click **Runtime → Change Runtime Type**
2. Select **GPU** (T4 is free, A100 requires Colab Pro)
3. Click **Save**

### Step 4 — Fix Dataset Path
Change the dataset load path to:
```python
from google.colab import drive
drive.mount('/content/drive')

dataset = torch.load(
    "/content/drive/MyDrive/CompilerProject/embedded_dataset_balanced.pt",
    weights_only=False
)
```

### Step 5 — Install Dependencies
The first cell handles this:
```python
!pip install catboost torch-geometric torch-scatter -q
```

---

## Option 3 — Run Locally

### Requirements
- Python 3.10+
- CUDA GPU strongly recommended (CPU will be very slow for GNN training)
- ~4GB RAM minimum, 8GB+ recommended
- ~3GB disk space for dataset

### Step 1 — Clone the Repo
```bash
git clone https://github.com/KK-College/RustCPG-Detect.git
cd RustCPG-Detect
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```
For GPU (CUDA 11.8):
```bash
pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 3 — Download Dataset
```bash
# Install Kaggle CLI first
pip install kaggle

# Download (requires Kaggle account + API key in ~/.kaggle/kaggle.json)
kaggle datasets download -d kaarthikeyaganji/cid-i2
unzip cid-i2.zip -d data/
```

### Step 4 — Fix Dataset Path in Notebooks
Change the load path to:
```python
dataset = torch.load("data/embedded_dataset_balanced.pt", weights_only=False)
```

### Step 5 — Launch Jupyter
```bash
jupyter notebook
```
Open notebooks in order: 01 → 02 → 03 → 04 → 05

---

## Dataset Details

| Property | Value |
|---|---|
| File | `embedded_dataset_balanced.pt` |
| Format | PyTorch Geometric dataset (list of `Data` objects) |
| Size | ~2GB |
| Samples | 32,510 |
| Classes | 5 (Safe, UAF, DataRace, IntOverflow, MemCorrupt) |
| Per class | 6,502 (perfectly balanced) |
| Node features | 835-dim (768 BERT + 67 structural) |
| Kaggle URL | https://www.kaggle.com/datasets/kaarthikeyaganji/cid-i2 |

### How to Load the Dataset
```python
import torch
from collections import Counter

dataset = torch.load("embedded_dataset_balanced.pt", weights_only=False)

print(f"Total samples: {len(dataset)}")
print(f"Class distribution: {Counter([d.y.item() for d in dataset])}")
print(f"Node feature shape: {dataset[0].x.shape}")   # [num_blocks, 835]
print(f"Edge index shape:   {dataset[0].edge_index.shape}")
print(f"Edge attr shape:    {dataset[0].edge_attr.shape}")
print(f"Label:              {dataset[0].y.item()}")   # 0-4
```

---

## Notebook Order & Dependencies

```
01_dataset_generation         → No dependencies (standalone explainer)
        ↓
02_baseline_rust_ir_bert      → Needs: dataset loaded
        ↓
03_rustcpg_detect_pipeline    → Needs: dataset loaded
        ↓                       Produces: binary_model, best_t, top_k, X splits
04_gnn_ablation_study         → Needs: dataset loaded
        ↓                       Produces: gnn_results
05_results_and_comparison     → Needs: everything above
```

---

## Common Errors & Fixes

**Error:** `FileNotFoundError: embedded_dataset_balanced.pt`
**Fix:** Update the dataset path in the load cell (see path options above)

**Error:** `CUDA out of memory`
**Fix:** Reduce `batch_size` in the GNN training cell from 32 to 16

**Error:** `torch_geometric not found`
**Fix:** `!pip install torch-geometric torch-scatter`

**Error:** `catboost GPU not available`
**Fix:** Change `task_type='GPU'` to `task_type='CPU'` — will be slower but works

**Error:** `weights_only=False` warning on older PyTorch
**Fix:** Update PyTorch: `pip install torch --upgrade`

---

## Kaggle API Key Setup (for CLI download)

1. Go to **https://www.kaggle.com/account**
2. Scroll to **API** section → Click **"Create New API Token"**
3. Download `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json`
5. Run: `chmod 600 ~/.kaggle/kaggle.json`

---

*For questions, open an issue on the GitHub repo.*
