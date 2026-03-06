"""
RustCPG-Detect — CPG-Enhanced Vulnerability Detection in Rust
Amrita Vishwa Vidyapeetham, Amaravati Campus — Batch 01

Team:
    Kaarthikeya Lakshman Ganji  (AV.SC.U4CSE23118)
    Guditi Sai Kaushik          (AV.SC.U4CSE23109)
    P.V.S Pranav                (AV.SC.U4CSE23136)

Submitted to: Dr. K.S.L Prasanna, Dept. of CSE
"""

from src.parser    import LLVMIRParser, Function, BasicBlock, Instruction
from src.features  import extract_structural_features
from src.cpg_builder import build_cpg
from src.models    import create_model, count_parameters

__version__ = "1.0.0"
__all__ = [
    'LLVMIRParser', 'Function', 'BasicBlock', 'Instruction',
    'extract_structural_features', 'build_cpg',
    'create_model', 'count_parameters',
]
