# =============================================================================
# BASE PAPER IMPLEMENTATION — Rust-IR-BERT
# Reproducing: "Vulnerability Detection in Rust via LLVM IR and GraphCodeBERT"
# Machine Learning and Knowledge Extraction, 2025, 7, 79
# https://doi.org/10.3390/make7030079
#
# Team: Kaarthikeya Lakshman Ganji, Guditi Sai Kaushik, P.V.S Pranav
# Institution: Amrita Vishwa Vidyapeetham, Amaravati Campus
#
# HOW TO RUN:
#   Kaggle  → New notebook on https://www.kaggle.com/datasets/kaarthikeyaganji/cid-i2
#             Enable GPU T4 · paste code · Run All
#   Colab   → Mount Drive · upload dataset · change DATASET_PATH below · Run All
#   Local   → pip install -r requirements.txt · set DATASET_PATH · python base_paper_code.py
#
# DATASET PATH — update this one line for your environment:
#   Kaggle : "/kaggle/input/cid-i2/embedded_dataset_balanced.pt"
#   Colab  : "/content/drive/MyDrive/CompilerProject/embedded_dataset_balanced.pt"
#   Local  : "data/embedded_dataset_balanced.pt"
# =============================================================================

DATASET_PATH = "/kaggle/input/cid-i2/embedded_dataset_balanced.pt"
RESULTS_DIR  = "/kaggle/working"   # change to your output dir

# =============================================================================
# SECTION 1 — INSTALL & IMPORTS
# =============================================================================

# Uncomment the lines below when running on Kaggle/Colab:
# !pip install catboost torch-geometric -q

# Uncomment for Colab only (mount Google Drive):
# from google.colab import drive
# drive.mount('/content/drive')

import torch
import numpy as np
import json, os
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score,
    precision_score, recall_score
)

from catboost import CatBoostClassifier, Pool
from torch_geometric.data import Data

os.makedirs(f"{RESULTS_DIR}/results", exist_ok=True)

CLASS_NAMES = {
    0: 'Safe',
    1: 'UAF',
    2: 'Data Race',
    3: 'Integer Overflow',
    4: 'Memory Corruption'
}

print("=" * 60)
print("  BASE PAPER — Rust-IR-BERT Reproduction")
print("=" * 60)
print("All imports successful ✅")


# =============================================================================
# SECTION 2 — LOAD DATASET
# Each sample is a PyTorch Geometric Data object with:
#   .x          → [num_blocks, 835]  node features (768 BERT + 67 structural)
#   .edge_index → [2, num_edges]     CPG edges
#   .edge_attr  → [num_edges]        edge types (0=CFG, 1=DFG)
#   .y          → [1]                label (0=Safe, 1=UAF, 2=DR, 3=IO, 4=MC)
# =============================================================================

print(f"\nLoading dataset from: {DATASET_PATH}")
dataset = torch.load(DATASET_PATH, weights_only=False)

counts = Counter([d.y.item() for d in dataset])
print(f"Loaded {len(dataset)} samples")
print("Class distribution:")
for cls, name in CLASS_NAMES.items():
    print(f"  {name:20s} : {counts[cls]}")

d0 = dataset[0]
print(f"\nNode feature shape : {d0.x.shape}  → using only cols 0:768 (BERT)")
print(f"Edge index shape   : {d0.edge_index.shape}")


# =============================================================================
# SECTION 3 — FEATURE EXTRACTION (BASE PAPER METHOD)
#
# KEY POINT: The base paper uses ONLY the 768-dim GraphCodeBERT embeddings.
# It ignores columns 768:835 (structural features) and all graph topology.
# This is the exact difference between base paper and our CPG extension.
# =============================================================================

def extract_binary_features(dataset):
    """
    Base paper feature extraction:
    - Mean pool the 768-dim BERT embedding cols across all BasicBlocks
    - Returns (X, y_binary, y_multiclass)
    """
    X_list, y_binary, y_multi = [], [], []

    for d in dataset:
        x = d.x.numpy()          # shape: (num_blocks, 835)

        # BASE PAPER: mean pool GraphCodeBERT embeddings → 768-dim
        # Only columns 0:768 — structural features (768:835) discarded
        bert_mean = x[:, :768].mean(0)

        X_list.append(bert_mean)
        label = d.y.item()
        y_binary.append(0 if label == 0 else 1)   # 0=Safe, 1=Any vuln
        y_multi.append(label)

    return np.stack(X_list), np.array(y_binary), np.array(y_multi)


print("\nExtracting features (base paper method: 768-dim BERT mean pool)...")
X, y_bin, y_multi = extract_binary_features(dataset)

print(f"Feature matrix shape : {X.shape}    ← 768-dim per function")
print(f"Binary labels        : Safe={( y_bin==0).sum()}, Vulnerable={( y_bin==1).sum()}")


# =============================================================================
# SECTION 4 — TRAIN / VAL / TEST SPLIT + NORMALIZATION
# Exactly replicating base paper protocol: 70/15/15 stratified split
# StandardScaler fit on train only, applied to val and test
# =============================================================================

# 70% train, 30% temp
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y_bin, test_size=0.30, stratify=y_bin, random_state=42
)

# Split temp → 50% val, 50% test → each is 15% of total
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)

# StandardScaler — EXACTLY as in base paper
# fit_transform on train only, transform on val/test
scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te  = scaler.transform(X_te)

print(f"\nTrain : {X_tr.shape}  |  Safe={( y_tr==0).sum()}, Vuln={( y_tr==1).sum()}")
print(f"Val   : {X_val.shape}  |  Safe={( y_val==0).sum()}, Vuln={( y_val==1).sum()}")
print(f"Test  : {X_te.shape}  |  Safe={( y_te==0).sum()}, Vuln={( y_te==1).sum()}")
print("Normalized ✅")


# =============================================================================
# SECTION 5 — TRAIN CATBOOST (EXACT BASE PAPER HYPERPARAMETERS)
# depth=6, L2=3, learning_rate=0.05, 500 iterations
# These values are taken directly from the base paper
# =============================================================================

model_binary = CatBoostClassifier(
    iterations          = 500,
    learning_rate       = 0.05,
    depth               = 6,
    l2_leaf_reg         = 3,
    loss_function       = 'Logloss',
    eval_metric         = 'Accuracy',
    random_seed         = 42,
    early_stopping_rounds = 50,
    verbose             = 100,
    task_type           = 'GPU'    # change to 'CPU' if no GPU available
)

train_pool = Pool(X_tr,  y_tr)
val_pool   = Pool(X_val, y_val)

print("\nTraining Binary CatBoost (base paper config: depth=6, L2=3, 500 iters)...")
model_binary.fit(train_pool, eval_set=val_pool, use_best_model=True)
print("Training complete ✅")


# =============================================================================
# SECTION 6 — THRESHOLD TUNING (LISTING 3 FROM BASE PAPER)
# Sweep threshold from 0.1 to 0.9 on validation set
# Select threshold that maximises F1 score
# Base paper found optimal @ 0.35
# =============================================================================

val_probs = model_binary.predict_proba(X_val)[:, 1]

best_threshold, best_f1 = 0.5, 0.0

for t in np.linspace(0.1, 0.9, 81):
    preds = (val_probs >= t).astype(int)
    f1    = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nOptimal Threshold : {best_threshold:.2f}  (base paper reports 0.35)")
print(f"Validation F1     : {best_f1:.4f}")


# =============================================================================
# SECTION 7 — EVALUATE ON TEST SET
# Show both standard (0.50) and tuned threshold results
# =============================================================================

test_probs   = model_binary.predict_proba(X_te)[:, 1]
y_pred_std   = (test_probs >= 0.50).astype(int)
y_pred_tuned = (test_probs >= best_threshold).astype(int)

print("\n" + "=" * 55)
print("  STANDARD THRESHOLD (0.50) — baseline comparison")
print("=" * 55)
acc = accuracy_score(y_te, y_pred_std)
f1  = f1_score(y_te, y_pred_std, average='macro', zero_division=0)
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)  |  Macro F1 : {f1:.4f}")
print(classification_report(
    y_te, y_pred_std,
    target_names=['Safe', 'Vulnerable'], zero_division=0
))

print("=" * 55)
print(f"  TUNED THRESHOLD ({best_threshold:.2f}) — base paper method")
print("=" * 55)
acc_t = accuracy_score(y_te, y_pred_tuned)
f1_t  = f1_score(y_te, y_pred_tuned, average='macro', zero_division=0)
print(f"Accuracy : {acc_t:.4f} ({acc_t*100:.2f}%)  |  Macro F1 : {f1_t:.4f}")
print(classification_report(
    y_te, y_pred_tuned,
    target_names=['Safe', 'Vulnerable'], zero_division=0
))


# =============================================================================
# SECTION 8 — CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_te, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (tuned threshold):")
print(f"  True Negatives  (Safe→Safe)  : {tn}")
print(f"  False Positives (Safe→Vuln)  : {fp}  ← false alarms")
print(f"  False Negatives (Vuln→Safe)  : {fn}  ← missed vulnerabilities")
print(f"  True Positives  (Vuln→Vuln)  : {tp}")
print(f"\n  Safe Precision  : {tn/(tn+fn):.4f}")
print(f"  Vuln Recall     : {tp/(tp+fn):.4f}   ← most important for security")
print(f"  Vuln Precision  : {tp/(tp+fp):.4f}")
print(f"  False Alarm Rate: {fp/(fp+tn):.4f}")


# =============================================================================
# SECTION 9 — SAVE RESULTS TO JSON
# =============================================================================

results = {
    'binary_standard': {
        'accuracy' : float(acc),
        'macro_f1' : float(f1),
        'threshold': 0.50
    },
    'binary_tuned': {
        'accuracy' : float(acc_t),
        'macro_f1' : float(f1_t),
        'threshold': float(best_threshold)
    },
    'base_paper_reference': {
        'accuracy' : 0.9811,
        'macro_f1' : 0.9714,
        'threshold': 0.35,
        'note'     : 'Base paper tested on their own 2305-sample dataset'
    }
}

results_path = f"{RESULTS_DIR}/results/results_base_paper.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path} ✅")


# =============================================================================
# SECTION 10 — 5-FOLD CROSS-VALIDATION (BASE PAPER PROTOCOL)
# Base paper reports: 0.982 ± 0.008 on their 2,305-sample dataset
# We run same protocol on our 32,510-sample dataset
# =============================================================================

print("\n" + "=" * 55)
print("  5-FOLD CROSS-VALIDATION")
print("=" * 55)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []

# Reconstruct unscaled full dataset (CV does its own scaling per fold)
X_full_raw = np.vstack([
    scaler.inverse_transform(X_tr),
    scaler.inverse_transform(X_val),
    scaler.inverse_transform(X_te)
])
y_full = np.concatenate([y_tr, y_val, y_te])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_raw, y_full)):
    sc      = StandardScaler()
    X_f_tr  = sc.fit_transform(X_full_raw[tr_idx])
    X_f_val = sc.transform(X_full_raw[val_idx])

    m = CatBoostClassifier(
        iterations    = 500,
        learning_rate = 0.05,
        depth         = 6,
        l2_leaf_reg   = 3,
        loss_function = 'Logloss',
        random_seed   = fold,
        verbose       = 0,
        task_type     = 'GPU'
    )
    m.fit(
        Pool(X_f_tr,  y_full[tr_idx]),
        eval_set=Pool(X_f_val, y_full[val_idx]),
        use_best_model=True
    )

    preds = (m.predict_proba(X_f_val)[:, 1] >= best_threshold).astype(int)
    acc   = accuracy_score(y_full[val_idx], preds)
    cv_acc.append(acc)
    print(f"  Fold {fold+1}: Acc = {acc:.4f}")

print(f"\n  5-Fold CV Accuracy : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"  Base paper reports : 0.982  ± 0.008  (on 2,305-sample dataset)")
print(f"\n  NOTE: Lower CV on our data is expected — 14× larger, more diverse dataset")


# =============================================================================
# SECTION 11 — FINAL COMPREHENSIVE METRICS
# =============================================================================

print("\n" + "=" * 60)
print("  FINAL METRICS ON TEST SET (tuned threshold)")
print("=" * 60)

accuracy      = accuracy_score(y_te, y_pred_tuned)
precision     = precision_score(y_te, y_pred_tuned, zero_division=0)
recall        = recall_score(y_te, y_pred_tuned, zero_division=0)
f1_score_bin  = f1_score(y_te, y_pred_tuned, zero_division=0)
macro_f1      = f1_score(y_te, y_pred_tuned, average='macro', zero_division=0)
weighted_f1   = f1_score(y_te, y_pred_tuned, average='weighted', zero_division=0)
roc_auc       = roc_auc_score(y_te, test_probs)
balanced_acc  = balanced_accuracy_score(y_te, y_pred_tuned)

cm2           = confusion_matrix(y_te, y_pred_tuned)
tn2, fp2, fn2, tp2 = cm2.ravel()
specificity   = tn2 / (tn2 + fp2)
sensitivity   = tp2 / (tp2 + fn2)

print(f"  Accuracy             : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision            : {precision:.4f}")
print(f"  Recall (Sensitivity) : {recall:.4f}")
print(f"  Specificity          : {specificity:.4f}")
print(f"  F1 Score (binary)    : {f1_score_bin:.4f}")
print(f"  Macro F1             : {macro_f1:.4f}")
print(f"  Weighted F1          : {weighted_f1:.4f}")
print(f"  ROC-AUC              : {roc_auc:.4f}")
print(f"  Balanced Accuracy    : {balanced_acc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {cm2}")
print(f"\nDetailed Classification Report:")
print(classification_report(
    y_te, y_pred_tuned,
    target_names=['Safe', 'Vulnerable'], zero_division=0
))

print("\n" + "=" * 60)
print("  BASE PAPER REPRODUCTION COMPLETE")
print("  See notebooks/03_rustcpg_detect_pipeline.ipynb for our")
print("  CPG-enhanced extension achieving 93.49% accuracy.")
print("=" * 60)
