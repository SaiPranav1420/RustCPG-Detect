"""
cpg_builder.py — Code Property Graph Builder for RustCPG-Detect
Converts a parsed Function into a PyTorch Geometric Data object.

Each node  = one BasicBlock  (835-dim feature vector)
Each edge  = control flow (type=0) or data flow (type=1)

Usage:
    from src.cpg_builder import build_cpg
    graph = build_cpg(function, bert_embeddings, struct_features, label=1)
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.parser import Function


def build_cpg(
    function: Function,
    bert_embeddings: list,
    struct_features: list,
    label: int
) -> Data:
    """
    Build a Code Property Graph for one Rust function.

    Args:
        function        : parsed Function object (from LLVMIRParser)
        bert_embeddings : list of np.ndarray (768,), one per BasicBlock
        struct_features : list of np.ndarray (67,),  one per BasicBlock
        label           : int — vulnerability class (0=Safe, 1=UAF,
                          2=DataRace, 3=IntOverflow, 4=MemCorrupt)

    Returns:
        torch_geometric.data.Data with:
            x          = [num_nodes, 835]  float tensor
            edge_index = [2, num_edges]    long tensor
            edge_attr  = [num_edges]       long tensor (0=CFG, 1=DFG)
            y          = [1]               long tensor
    """
    bbs = function.basic_blocks
    n   = len(bbs)

    if n == 0:
        return None

    # ── Node features: fuse BERT (768) + structural (67) = 835-dim ──
    node_feats = []
    for bert, struct in zip(bert_embeddings, struct_features):
        fused = np.concatenate([bert[:768], struct[:67]])  # enforce dims
        node_feats.append(fused.astype(np.float32))

    x = torch.tensor(np.stack(node_feats), dtype=torch.float)  # [n, 835]

    # ── Map block name → index ───────────────────────────────────────
    name_to_idx = {bb.name: i for i, bb in enumerate(bbs)}

    edge_src  = []
    edge_dst  = []
    edge_type = []
    seen_edges = set()

    def add_edge(s, d, t):
        key = (s, d, t)
        if key not in seen_edges and s != d:
            seen_edges.add(key)
            edge_src.append(s)
            edge_dst.append(d)
            edge_type.append(t)

    for i, bb in enumerate(bbs):
        # Collect variable definitions in this block (for data flow)
        defs_here = set()

        for instr in bb.instructions:
            op  = instr.opcode.lower()
            ops = instr.operands

            # ── Control flow edges ───────────────────────────────────
            if op in ('br', 'switch'):
                for name, j in name_to_idx.items():
                    # Check if block label referenced in branch instruction
                    if (f'%{name}' in ops or f'label %{name}' in ops
                            or f', {name}' in ops):
                        add_edge(i, j, 0)  # 0 = control flow

            # ── Collect definitions ──────────────────────────────────
            if instr.result:
                var = instr.result.lstrip('%').strip()
                defs_here.add(var)

        # ── Data flow edges: def in block i → use in block j ─────────
        for var in defs_here:
            for j, other_bb in enumerate(bbs):
                if j == i:
                    continue
                for other_instr in other_bb.instructions:
                    if f'%{var}' in other_instr.operands:
                        add_edge(i, j, 1)  # 1 = data flow
                        break  # one edge per (i, j, var) is enough

    # ── Sequential fallthrough edges (if no explicit control flow) ───
    # Many blocks fall through to the next block implicitly
    for i in range(n - 1):
        bb = bbs[i]
        last_op = bb.instructions[-1].opcode.lower() if bb.instructions else ''
        if last_op not in ('br', 'switch', 'ret', 'unreachable'):
            add_edge(i, i + 1, 0)

    # ── Build tensors ────────────────────────────────────────────────
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_type,             dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros(0,      dtype=torch.long)

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor([label], dtype=torch.long)
    )


def dataset_stats(graphs: list) -> dict:
    """Print summary statistics for a list of CPG Data objects."""
    from collections import Counter
    labels    = [g.y.item() for g in graphs]
    n_nodes   = [g.x.shape[0] for g in graphs]
    n_edges   = [g.edge_index.shape[1] for g in graphs]
    class_map = {0: 'Safe', 1: 'UAF', 2: 'DataRace',
                 3: 'IntOverflow', 4: 'MemCorrupt'}

    stats = {
        'total':     len(graphs),
        'classes':   {class_map.get(k, k): v for k, v in Counter(labels).items()},
        'avg_nodes': float(np.mean(n_nodes)),
        'avg_edges': float(np.mean(n_edges)),
        'feat_dim':  int(graphs[0].x.shape[1]) if graphs else 0,
    }
    return stats
