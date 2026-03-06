# Base Paper — Rust-IR-BERT: Full Implementation Guide
### Reproduced by: RustCPG-Detect Team | Amrita Vishwa Vidyapeetham, Amaravati Campus

---

## Files in This Folder

```
base_paper_implementation/
├── BASE_PAPER_EXPLAINER.md     ← This file — full explanation, context, results
└── base_paper_code.py          ← Complete runnable code (paste into Kaggle/Colab)
```

---

## 1. What Is the Base Paper?

**Title:** Rust-IR-BERT: Vulnerability Detection in Rust via LLVM IR and GraphCodeBERT  
**Published in:** Machine Learning and Knowledge Extraction, 2025, Volume 7, Article 79  
**DOI / Link:** https://doi.org/10.3390/make7030079  
**Our role:** We reproduced this exact pipeline on our 14× larger dataset, then built RustCPG-Detect on top of it

As described in our presentation (Slide 7):

> "This paper proposes Rust-IR-BERT, a novel ML pipeline for Rust vulnerability detection:
> it compiles code to LLVM IR, extracts 768D GraphCodeBERT embeddings (encoding
> data/control-flow semantics), and classifies via CatBoost. Trained on 2300+ labeled
> samples from RustSec/OSV/GitHub, it yields high accuracy/F1 on held-out tests
> (93.7% vulnerable recall), surpassing source-based baselines and tools like HALURust
> by reducing syntactic noise; it generalizes to unseen crates (95.5% accuracy),
> supporting early CI/CD scans."

---

## 2. The Base Paper Pipeline — Step by Step

```
STEP 1 — IR Generation
─────────────────────────────────────────────────────────────
  Rust source file (.rs)
        ↓
  rustc --emit=llvm-ir snippet.rs -o output.ll
        ↓
  LLVM Intermediate Representation (.ll file)

  WHY: LLVM IR strips away Rust's syntactic sugar and exposes
  raw memory operations. Every pointer dereference, free(),
  arithmetic op becomes explicit and unambiguous. The base paper
  shows this alone boosts accuracy from 80% → 98.1% vs raw Rust.


STEP 2 — Embedding Extraction
─────────────────────────────────────────────────────────────
  LLVM IR text
        ↓
  microsoft/graphcodebert-base tokenizer
        ↓
  12 Transformer layers with data-flow-aware attention
        ↓
  Mean pool across all token positions
        ↓
  768-dimensional semantic embedding vector per function

  WHY GraphCodeBERT (not regular BERT):
  - Pre-trained on source code (not English)
  - Data-flow aware: understands that %result in one instruction
    came from a previous defining instruction
  - Treats LLVM IR tokens (%5, alloca, getelementptr) correctly


STEP 3 — Preprocessing
─────────────────────────────────────────────────────────────
  768-dim embedding vector
        ↓
  StandardScaler (zero mean, unit variance per feature)
        ↓
  Normalized 768-dim vector


STEP 4 — CatBoost Classification
─────────────────────────────────────────────────────────────
  Normalized 768-dim vector
        ↓
  CatBoostClassifier:
    iterations          = 500
    learning_rate       = 0.05
    depth               = 6
    l2_leaf_reg         = 3
    loss_function       = Logloss
    eval_metric         = Accuracy
    early_stopping      = 50 rounds
        ↓
  Probability score: P(vulnerable) ∈ [0, 1]


STEP 5 — Threshold Optimisation (Listing 3 from paper)
─────────────────────────────────────────────────────────────
  P(vulnerable) from validation set
        ↓
  Sweep threshold t from 0.1 → 0.9 in steps of 0.01
  For each t: compute F1 on validation labels
  Select t* = argmax F1
        ↓
  t* = 0.35 (base paper) — applied to test set
        ↓
  Final prediction: VULNERABLE if P(vuln) ≥ 0.35 else SAFE
```

---

## 3. Key Contributions of the Base Paper (From Slide 7)

1. Compiles Rust source → LLVM IR — preserves data-flow semantics
2. Tokenizes IR using GraphCodeBERT's IR-aware tokenizer
3. Produces 768-dimensional semantic embeddings via mean pooling
4. CatBoost classifier with StandardScaler normalization (depth=6, L2=3)
5. Threshold tuning on validation set (optimal @ 0.35) for F1 maximization
6. 5-fold stratified CV: 0.982 ± 0.008 accuracy on 2,305 samples

---

## 4. Base Paper Results (On Their 2,305-Sample Dataset)

| Metric | Value |
|---|---|
| Test Accuracy | 98.11% |
| Macro F1 | 97.14% |
| Vulnerable (class 1) Precision | 0.9737 |
| Vulnerable (class 1) Recall | 0.9367 |
| Vulnerable F1 | 0.9548 |
| Safe (class 0) Precision | 0.9830 |
| Safe (class 0) Recall | 0.9931 |
| Safe F1 | 0.9880 |
| 5-Fold CV Accuracy | 0.982 ± 0.008 |
| Optimal Threshold | 0.35 |
| External test (230 unseen IR samples) | 95.5% accuracy |
| Dataset size | 2,305 samples |

---

## 5. Our Reproduction (On Our 32,510-Sample Dataset)

We ran the **exact same pipeline** on our larger, harder dataset:

| Metric | Base Paper (their data) | Our Reproduction (our data) |
|---|---|---|
| Accuracy (std threshold 0.50) | — | ~91.70% |
| Macro F1 (std threshold) | — | ~84.81% |
| Accuracy (tuned threshold) | 98.11% | ~91.70% |
| 5-Fold CV | 0.982 ± 0.008 | See notebook output |
| Dataset size | 2,305 | 32,510 (14× larger) |

### Why Is Our Reproduction Lower?

From our presentation (Slide 9) — our dataset is deliberately harder:

- 32,510 samples vs 2,305 — 14× more diverse code
- 5 balanced classes vs binary — drawn from broader CVE landscape
- More varied unsafe patterns, not just the most common CVEs
- Think of it as the difference between 10 practice questions vs 140 exam questions

Our CPG-enhanced extension (in the main notebooks) improves this to **93.49%**.

---

## 6. Limitations of the Base Paper (From Slide 8)

As we identified in our project:

1. **LLVM IR compilation overhead** — fails on non-compilable snippets without manual wrapping (we built automatic wrappers to handle this)
2. **Binary classification only** — no distinction between UAF, Data Race, Integer Overflow, Memory Corruption
3. **Fixed 768-dim embeddings** — GraphCodeBERT's pre-trained weights may not optimally represent all Rust unsafe patterns
4. **Computational cost** — embedding extraction is slow; less ideal for lightweight scanning
5. **Dataset bias** — heavy toward known RustSec/OSV advisories; rare vulnerability types underrepresented
6. **No structural graph features** — the CPG edge structure (control + data flow) is completely ignored

These limitations are exactly what RustCPG-Detect addresses.

---

## 7. How to Run the Code

### Option A — Kaggle (Recommended, Free GPU)

1. Go to: **https://www.kaggle.com/datasets/kaarthikeyaganji/cid-i2**
2. Click **New Notebook** (top right of dataset page)
3. In the new notebook editor: **File → Import Notebook** → upload `base_paper_code.py`
   - Kaggle will convert the .py to notebook cells automatically
   - Alternatively: create a new notebook and paste the code from `base_paper_code.py`
4. Enable GPU: right sidebar → **Session options** → Accelerator: **GPU T4 x2**
5. The dataset is already attached — use this path:
   ```python
   dataset = torch.load(
       "/kaggle/input/cid-i2/embedded_dataset_balanced.pt",
       weights_only=False
   )
   ```
6. Click **Run All** — takes approximately 20–25 minutes

### Option B — Google Colab

1. Open **https://colab.research.google.com**
2. **File → Upload Notebook** → upload `base_paper_code.py`
3. Enable GPU: **Runtime → Change Runtime Type → GPU (T4)**
4. Mount Drive and upload the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Upload embedded_dataset_balanced.pt to MyDrive/CompilerProject/
   ```
5. Change dataset path in the load cell:
   ```python
   dataset = torch.load(
       "/content/drive/MyDrive/CompilerProject/embedded_dataset_balanced.pt",
       weights_only=False
   )
   ```
6. Run all cells

### Dataset Download (if needed)
```bash
pip install kaggle
kaggle datasets download -d kaarthikeyaganji/cid-i2
# Kaggle API key must be set up at ~/.kaggle/kaggle.json
```

---

## 8. Understanding the Code Section by Section

The code in `base_paper_code.py` has these sections:

| Section | Lines | What it does |
|---|---|---|
| Setup | 1–25 | Install catboost, torch-geometric. Set paths and class names. |
| Load Dataset | 26–32 | Load the 32,510-sample CPG dataset (.pt file) |
| Feature Extraction | 33–55 | `extract_binary_features()` — mean-pool BERT cols (0:768) only |
| Data Split | 56–72 | 70/15/15 stratified split + StandardScaler (fit on train only) |
| Train CatBoost | 73–95 | Exact base paper hyperparams: depth=6, L2=3, 500 iters |
| Threshold Tuning | 96–108 | F1-sweep on validation set (Listing 3 from paper) |
| Evaluation | 109–140 | Standard (0.50) and tuned threshold results + classification report |
| Confusion Matrix | 141–155 | TN/FP/FN/TP + Safe Precision + Vuln Recall |
| Save Results | 156–170 | JSON output to Drive |
| 5-Fold CV | 171–205 | Full cross-validation matching base paper protocol |
| Final Metrics | 206–245 | ROC-AUC, Balanced Accuracy, Specificity, full report |

---

## 9. What GraphCodeBERT Produces — Intuition

The base paper's core insight is that LLVM IR text, when fed to GraphCodeBERT, produces an embedding vector where:

- Functions with **Use-After-Free** patterns cluster together in 768-dim space
- Functions with **integer overflow** patterns cluster elsewhere
- **Safe functions** form their own cluster

This happens because GraphCodeBERT was trained on millions of code snippets with data-flow graph supervision. It learned that patterns like:

```llvm
; This sequence has high UAF signal:
%ptr = call i8* @malloc(i64 16)
call void @free(i8* %ptr)        ; free
%val = load i32, i32* %ptr       ; use after free — flagged
```

...produce characteristic embedding signatures that CatBoost can separate from safe code.

The threshold of 0.35 (vs our 0.71 for CPG features) reflects that with only 768-dim BERT features and a small dataset, the model needs a lower confidence bar to call something vulnerable.

---

## 10. Comparison: Base Paper vs Our Extension

| Aspect | Base Paper | RustCPG-Detect (Ours) |
|---|---|---|
| Feature vector | 768-dim BERT mean pool | 5,093-dim (7 pooling strategies) |
| Graph structure used? | No | Yes — CPG (CFG + DFG edges) |
| Structural features? | No | Yes — 67-dim per BasicBlock |
| GNN model? | No | Yes — GCN ablation (4 variants) |
| Classification task | Binary only | Binary + 5-class hierarchical |
| Feature selection | None | RF top-2000 |
| Dataset | 2,305 samples | 32,510 samples (14×) |
| Optimal threshold | 0.35 | 0.71 |
| Binary accuracy | 98.11% (own data) | 93.49% (harder data) |

---

## 11. Citation

**Base Paper:**
```
Rust-IR-BERT: Vulnerability Detection in Rust via LLVM IR and GraphCodeBERT
Machine Learning and Knowledge Extraction, 2025, 7, 79
https://doi.org/10.3390/make7030079
```

**Our Extension:**
```
RustCPG-Detect: CPG-Enhanced Vulnerability Detection in Rust
Kaarthikeya Lakshman Ganji, Guditi Sai Kaushik, P.V.S Pranav
Amrita Vishwa Vidyapeetham, Amaravati Campus, 2025
GitHub  : https://github.com/KK-College/RustCPG-Detect
Dataset : https://www.kaggle.com/datasets/kaarthikeyaganji/cid-i2
```
