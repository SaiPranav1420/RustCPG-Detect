"""
features.py — Structural Feature Extractor for RustCPG-Detect
Extracts 67-dim handcrafted security features from each BasicBlock.

Usage:
    from src.features import extract_structural_features
    feat = extract_structural_features(basic_block)  # → np.ndarray (67,)
"""

import numpy as np
from src.parser import BasicBlock


# All LLVM IR opcodes we track in the histogram
OPCODES = [
    'load', 'store', 'call', 'br', 'ret', 'add', 'sub', 'mul',
    'sdiv', 'udiv', 'srem', 'urem', 'and', 'or', 'xor', 'shl',
    'lshr', 'ashr', 'icmp', 'fcmp', 'phi', 'select', 'alloca',
    'getelementptr', 'bitcast', 'inttoptr', 'ptrtoint', 'trunc',
    'zext', 'sext', 'fptoui', 'fptosi', 'uitofp', 'sitofp',
    'extractvalue', 'insertvalue', 'atomicrmw', 'cmpxchg',
    'fence', 'unreachable'
]  # 40 opcodes

OPCODE_IDX = {op: i for i, op in enumerate(OPCODES)}

# Deallocation function names
FREE_CALLS = {'free', 'drop_in_place', 'dealloc', '__rust_dealloc',
              'core::mem::drop', 'std::mem::drop'}

# Memory copy functions
MEMCPY_CALLS = {'memcpy', 'memmove', 'memset', 'llvm.memcpy', 'llvm.memmove'}


def extract_structural_features(bb: BasicBlock) -> np.ndarray:
    """
    Extract 67-dimensional structural feature vector from a BasicBlock.

    Feature breakdown:
        [  0: 39] Opcode histogram (40 dims) — count of each instruction type
        [ 40: 54] Binary security flags (15 dims) — dangerous operation indicators
        [ 55: 66] Block-level properties (12 dims) — size, complexity metrics

    Returns:
        np.ndarray of shape (67,), dtype float32
    """
    # ── Opcode histogram (40-dim) ────────────────────────────────────
    hist = np.zeros(len(OPCODES), dtype=np.float32)

    # ── Security flags ───────────────────────────────────────────────
    has_raw_ptr_ops     = 0  # getelementptr, inttoptr, ptrtoint
    has_unchecked_arith = 0  # add/sub/mul without nsw/nuw flags
    has_free_call       = 0  # deallocation call
    has_load_after_ops  = 0  # load instruction (potential UAF partner)
    has_atomic_ops      = 0  # atomicrmw, cmpxchg, fence
    has_alloca          = 0  # stack allocation
    has_memcpy          = 0  # bulk memory operation
    has_indirect_call   = 0  # call through pointer
    has_volatile        = 0  # volatile load/store
    has_unsafe_cast     = 0  # inttoptr or ptrtoint
    has_integer_arith   = 0  # any integer arithmetic
    has_float_arith     = 0  # floating point arithmetic
    has_branch          = 0  # conditional branch
    has_phi             = 0  # phi node (merge point)
    has_ret             = 0  # return instruction

    for instr in bb.instructions:
        op   = instr.opcode.lower()
        ops  = instr.operands.lower()

        # Histogram
        if op in OPCODE_IDX:
            hist[OPCODE_IDX[op]] += 1

        # Security flags
        if op in ('getelementptr', 'inttoptr', 'ptrtoint'):
            has_raw_ptr_ops = 1
        if op in ('inttoptr', 'ptrtoint'):
            has_unsafe_cast = 1
        if op in ('add', 'sub', 'mul') and 'nsw' not in ops and 'nuw' not in ops:
            has_unchecked_arith = 1
        if op == 'call':
            if any(f in ops for f in FREE_CALLS):
                has_free_call = 1
            if any(f in ops for f in MEMCPY_CALLS):
                has_memcpy = 1
            if '* %' in instr.operands or '*(' in instr.operands:
                has_indirect_call = 1
        if op == 'load':
            has_load_after_ops = 1
            if 'volatile' in ops:
                has_volatile = 1
        if op == 'store' and 'volatile' in ops:
            has_volatile = 1
        if op in ('atomicrmw', 'cmpxchg', 'fence'):
            has_atomic_ops = 1
        if op == 'alloca':
            has_alloca = 1
        if op in ('add', 'sub', 'mul', 'sdiv', 'udiv', 'srem', 'urem',
                  'and', 'or', 'xor', 'shl', 'lshr', 'ashr'):
            has_integer_arith = 1
        if op in ('fadd', 'fsub', 'fmul', 'fdiv', 'frem'):
            has_float_arith = 1
        if op == 'br':
            has_branch = 1
        if op == 'phi':
            has_phi = 1
        if op == 'ret':
            has_ret = 1

    flags = np.array([
        has_raw_ptr_ops, has_unchecked_arith, has_free_call,
        has_load_after_ops, has_atomic_ops, has_alloca, has_memcpy,
        has_indirect_call, has_volatile, has_unsafe_cast,
        has_integer_arith, has_float_arith, has_branch, has_phi, has_ret
    ], dtype=np.float32)  # 15 dims

    # ── Block-level properties (12-dim) ─────────────────────────────
    n = len(bb.instructions)
    block_props = np.array([
        float(n),               # raw instruction count
        float(min(n, 50)),      # capped count (outlier robust)
        float(n > 5),           # non-trivial block
        float(n > 10),
        float(n > 20),
        float(n > 50),          # very large block
        float(hist.sum()),      # total tracked instructions
        float(hist[OPCODE_IDX.get('load', 0)]),   # load count
        float(hist[OPCODE_IDX.get('store', 0)]),  # store count
        float(hist[OPCODE_IDX.get('call', 0)]),   # call count
        float(hist[OPCODE_IDX.get('alloca', 0)]), # alloca count
        float(n / 10.0),        # normalised size
    ], dtype=np.float32)  # 12 dims

    return np.concatenate([hist, flags, block_props])  # 40 + 15 + 12 = 67


def extract_all_features(function) -> list:
    """Extract structural features for all BasicBlocks in a function."""
    return [extract_structural_features(bb) for bb in function.basic_blocks]
