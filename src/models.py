"""
models.py — GNN Model Definitions for RustCPG-Detect
Implements the 4 variants tested in the ablation study.

Variant A: StructuralOnlyGCN  — 66-dim structural features, pure GCN
Variant B: GCNWithBERT        — 835-dim features, GCN
Variant C: GCNWithBERT        — 835-dim features, GCN + full CPG edges (WINNER)
Variant D: GATFullCPG         — 835-dim features, GAT with learned attention

Usage:
    from src.models import create_model
    model = create_model('C', in_channels=835, num_classes=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool


# ─────────────────────────────────────────────────────────────────────
# Variant A — Structural GCN (66-dim)
# Baseline: pure graph structure + expert structural features, no BERT.
# Params: 8,706
# Binary accuracy: 89.85%  |  Macro F1: 0.8319
# ─────────────────────────────────────────────────────────────────────
class StructuralOnlyGCN(nn.Module):
    """
    Variant A: Uses only the 66-dim structural features (no BERT).
    Demonstrates the contribution of graph structure alone.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GCNConv(66, 64)
        self.bn1     = nn.BatchNorm1d(64)
        self.conv2   = GCNConv(64, 32)
        self.bn2     = nn.BatchNorm1d(32)
        # Global mean + max pool → 64-dim → MLP
        self.mlp = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = x[:, :66]  # Use only structural features
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        g = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)  # [B, 64]
        return self.mlp(g)


# ─────────────────────────────────────────────────────────────────────
# Variants B & C — GCN with full CPG features (835-dim)
# B uses basic CFG edges; C uses full CPG (CFG + DFG) edges.
# Same architecture — difference is in the graph structure passed in.
# Params: 264,450
# Variant C binary accuracy: 91.94%  |  Macro F1: 0.8691  ← WINNER
# ─────────────────────────────────────────────────────────────────────
class GCNWithBERT(nn.Module):
    """
    Variants B & C: 3-layer GCN on 835-dim fused CPG node features.
    Global mean + max pooling → 128-dim → classification head.

    For Variant C, pass the full CPG edge_index (CFG + DFG edges).
    For Variant B, pass only CFG edges.
    """
    def __init__(self, in_channels: int = 835, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GCNConv(in_channels, 256)
        self.bn1     = nn.BatchNorm1d(256)
        self.conv2   = GCNConv(256, 128)
        self.bn2     = nn.BatchNorm1d(128)
        self.conv3   = GCNConv(128, 64)
        self.bn3     = nn.BatchNorm1d(64)
        # Global mean + max pool → 128-dim → MLP
        self.mlp = nn.Sequential(
            nn.Linear(128, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        g = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)  # [B, 128]
        return self.mlp(g)


# ─────────────────────────────────────────────────────────────────────
# Variant D — GAT with full CPG features (835-dim)
# Uses Graph Attention Networks — each node learns to weight neighbors.
# Params: 574,530  (2.2× more than GCN)
# Binary accuracy: 90.71%  |  Macro F1: 0.8398  ← WORSE than GCN
# Conclusion: CPG edge structure is already expressive; attention adds noise.
# ─────────────────────────────────────────────────────────────────────
class GATFullCPG(nn.Module):
    """
    Variant D: 3-layer GAT on 835-dim CPG features with edge attributes.
    Multi-head attention: 4 heads in layers 1 & 2, 1 head in layer 3.
    """
    def __init__(self, in_channels: int = 835, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GATConv(in_channels, 128, heads=4, edge_dim=1, dropout=dropout)
        self.bn1     = nn.BatchNorm1d(512)   # 128 × 4 heads
        self.conv2   = GATConv(512, 64, heads=4, edge_dim=1, dropout=dropout)
        self.bn2     = nn.BatchNorm1d(256)   # 64 × 4 heads
        self.conv3   = GATConv(256, 32, heads=1, edge_dim=1, dropout=dropout)
        self.bn3     = nn.BatchNorm1d(32)
        # Global mean + max pool → 64-dim → MLP
        self.mlp = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr must be float and 2D for GATConv edge_dim
        ea = edge_attr.float()
        if ea.dim() == 1:
            ea = ea.unsqueeze(-1)

        x = F.elu(self.bn1(self.conv1(x, edge_index, ea)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index, ea)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index, ea)))
        g = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)  # [B, 64]
        return self.mlp(g)


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────
def create_model(variant: str, in_channels: int = 835,
                 num_classes: int = 2) -> nn.Module:
    """
    Create a GNN model by variant name.

    Args:
        variant     : 'A', 'B', 'C', or 'D'
        in_channels : node feature dimensionality (835 for our dataset)
        num_classes : 2 for binary, 5 for multi-class

    Returns:
        nn.Module ready for training
    """
    if variant == 'A':
        return StructuralOnlyGCN(num_classes=num_classes)
    elif variant in ('B', 'C'):
        return GCNWithBERT(in_channels=in_channels, num_classes=num_classes)
    elif variant == 'D':
        return GATFullCPG(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from A, B, C, D.")


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(variant: str) -> None:
    """Print parameter count and architecture summary for a variant."""
    model = create_model(variant)
    n     = count_parameters(model)
    print(f"Variant {variant}: {model.__class__.__name__}")
    print(f"  Trainable parameters: {n:,}")
    print(f"  Architecture:")
    for name, module in model.named_children():
        print(f"    {name}: {module.__class__.__name__}")


ABLATION_RESULTS = {
    'A': {'accuracy': 0.8985, 'macro_f1': 0.8319, 'params':   8_706},
    'B': {'accuracy': 0.9188, 'macro_f1': 0.8673, 'params': 264_450},
    'C': {'accuracy': 0.9194, 'macro_f1': 0.8691, 'params': 264_450},  # WINNER
    'D': {'accuracy': 0.9071, 'macro_f1': 0.8398, 'params': 574_530},
}
