"""
parser.py — LLVM IR Parser for RustCPG-Detect
Converts .ll files into Function → BasicBlock → Instruction hierarchy.

Usage:
    from src.parser import LLVMIRParser
    parser = LLVMIRParser()
    functions = parser.parse_file("snippet.ll")
    for fn in functions:
        for bb in fn.basic_blocks:
            print(bb.name, len(bb.instructions))
"""


class Instruction:
    """A single LLVM IR instruction."""
    def __init__(self, opcode: str, operands: str, result: str = None):
        self.opcode   = opcode
        self.operands = operands
        self.result   = result   # e.g. "%5" for "%5 = add i32 %a, %b"

    def __repr__(self):
        return f"Instruction({self.opcode}, result={self.result})"


class BasicBlock:
    """A maximal sequence of IR instructions with single entry and exit."""
    def __init__(self, name: str, instructions: list):
        self.name         = name
        self.instructions = instructions

    @property
    def text(self) -> str:
        """Raw IR text of the block (for BERT tokenization)."""
        return "\n".join(i.operands for i in self.instructions)

    def __repr__(self):
        return f"BasicBlock({self.name}, {len(self.instructions)} instrs)"


class Function:
    """A complete Rust function with its basic blocks."""
    def __init__(self, name: str, basic_blocks: list):
        self.name         = name
        self.basic_blocks = basic_blocks

    def __repr__(self):
        return f"Function({self.name}, {len(self.basic_blocks)} blocks)"


class LLVMIRParser:
    """
    Parses LLVM IR (.ll) files produced by:
        rustc --emit=llvm-ir snippet.rs

    Handles: define/}, BasicBlock labels, all standard IR instructions.
    """

    DANGEROUS_OPCODES = {
        'load', 'store', 'getelementptr', 'inttoptr', 'ptrtoint',
        'atomicrmw', 'cmpxchg', 'call', 'free', 'alloca'
    }

    def parse_file(self, filepath: str) -> list:
        """Parse a .ll file and return list of Function objects."""
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
        return self._parse_functions(lines)

    def parse_text(self, ir_text: str) -> list:
        """Parse raw IR text string."""
        return self._parse_functions(ir_text.splitlines(keepends=True))

    def _parse_functions(self, lines: list) -> list:
        functions    = []
        current_fn   = None
        current_bb   = None
        instrs       = []

        for line in lines:
            line = line.rstrip()

            if line.startswith('define '):
                fname      = self._extract_fn_name(line)
                current_fn = {'name': fname, 'blocks': []}

            elif line == '}' and current_fn is not None:
                if current_bb and instrs:
                    current_fn['blocks'].append(BasicBlock(current_bb, list(instrs)))
                functions.append(Function(current_fn['name'], current_fn['blocks']))
                current_fn, current_bb, instrs = None, None, []

            elif (line.endswith(':') and current_fn is not None
                  and not line.startswith(' ') and not line.startswith('\t')):
                if current_bb is not None:
                    current_fn['blocks'].append(BasicBlock(current_bb, list(instrs)))
                current_bb = line[:-1].strip()
                instrs     = []

            elif current_bb is not None and line.strip():
                instr = self._parse_instruction(line.strip())
                if instr:
                    instrs.append(instr)

        return functions

    def _parse_instruction(self, line: str) -> Instruction:
        """Parse one instruction line into an Instruction object."""
        result = None
        if '=' in line and not line.startswith('store') and not line.startswith('br'):
            parts  = line.split('=', 1)
            result = parts[0].strip()
            line   = parts[1].strip()

        tokens = line.split()
        if not tokens:
            return None

        opcode   = tokens[0].lower()
        operands = line
        return Instruction(opcode, operands, result)

    def _extract_fn_name(self, line: str) -> str:
        """Extract function name from 'define ... @name(...)' line."""
        try:
            return line.split('@')[1].split('(')[0]
        except IndexError:
            return 'unknown'
