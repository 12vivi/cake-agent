"""LIR: the low-level, address-explicit kernel form ("SimPTX").

Cake IR lowers to LIR, and the simulated GPU executes LIR.  The direct-code
control arm (paper Sec. 5: "authors CUDA C++ and inline PTX directly") writes
LIR by hand through :mod:`cake.lir.builder`, so it must get shared-memory byte
addresses, mbarrier parities, transaction byte counts, TMEM columns, swizzle
modes and warp predicates right itself -- exactly the decisions that Cake IR
derives during lowering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..dtypes import DType
from ..errors import Loc
from ..expr import Var

NOLOC = Loc()


class LOp:
    kind = "op"
    loc: Loc
    src: Any


@dataclass(eq=False)
class LArg:
    """Operand of a register op: register tile, broadcast vector, immediate or scalar expr."""
    kind: str  # "reg" | "row" | "col" | "imm"
    value: Any

    def __repr__(self) -> str:
        if self.kind == "reg":
            return self.value
        if self.kind in ("row", "col"):
            return f"{self.value}.{self.kind}()"
        return repr(self.value)


@dataclass(eq=False)
class LDesc:
    """Shared-memory operand descriptor of an MMA (address + layout commitments)."""
    addr: Any
    rows: int
    cols: int
    dtype: DType
    swizzle: str

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * self.dtype.nbytes


@dataclass(eq=False)
class LLoop(LOp):
    var: Var
    start: Any
    stop: Any
    step: Any
    body: List[LOp]
    loc: Loc = NOLOC
    src: Any = None
    kind = "loop"


@dataclass(eq=False)
class LSet(LOp):
    name: str
    value: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "set"


@dataclass(eq=False)
class LMbarInit(LOp):
    addr: Any
    count: int
    loc: Loc = NOLOC
    src: Any = None
    kind = "mbar_init"


@dataclass(eq=False)
class LMbarArrive(LOp):
    addr: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "mbar_arrive"


@dataclass(eq=False)
class LMbarExpectTx(LOp):
    """mbarrier.arrive.expect_tx: one arrival plus an expected transaction byte count."""
    addr: Any
    nbytes: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "mbar_expect_tx"


@dataclass(eq=False)
class LMbarWait(LOp):
    """mbarrier.try_wait.parity loop: passes once the phase with ``parity`` has completed."""
    addr: Any
    parity: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "mbar_wait"


@dataclass(eq=False)
class LFence(LOp):
    """fence.proxy.async.shared::cta"""
    loc: Loc = NOLOC
    src: Any = None
    kind = "fence"


@dataclass(eq=False)
class LTmaLoad(LOp):
    dst: Any            # smem byte address
    tensor: str
    coords: Tuple[Any, ...]
    box: Tuple[int, int]
    dtype: DType
    swizzle: str
    mbar: Any           # completion mbarrier address
    engine: str = "tma"  # "tma" | "cp.async"
    loc: Loc = NOLOC
    src: Any = None
    kind = "tma_load"

    @property
    def nbytes(self) -> int:
        return self.box[0] * self.box[1] * self.dtype.nbytes


@dataclass(eq=False)
class LUmma(LOp):
    """tcgen05.mma: D[tmem] (+)= A[smem] x B[smem]^T, asynchronous."""
    d_col: Any
    a: LDesc
    b: LDesc
    m: int
    n: int
    k: int
    accumulate: Any
    b_major: str = "K"
    loc: Loc = NOLOC
    src: Any = None
    kind = "umma"


@dataclass(eq=False)
class LMmaSync(LOp):
    """Warp-level MMA (mma.sync / wgmma + wait): register accumulator, synchronous."""
    acc: str
    a: LDesc
    b: LDesc
    m: int
    n: int
    k: int
    accumulate: Any
    b_major: str = "K"
    loc: Loc = NOLOC
    src: Any = None
    kind = "mma_sync"


@dataclass(eq=False)
class LUmmaCommit(LOp):
    """tcgen05.commit: arrive on ``mbar`` once all prior MMAs of this thread complete."""
    mbar: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "umma_commit"


@dataclass(eq=False)
class LTmemLoad(LOp):
    dst: str
    col: Any
    lanes: int
    ncols: int
    loc: Loc = NOLOC
    src: Any = None
    kind = "tmem_load"


@dataclass(eq=False)
class LLdGlobal(LOp):
    dst: str
    tensor: str
    coords: Tuple[Any, ...]
    loc: Loc = NOLOC
    src: Any = None
    kind = "ld_global"


@dataclass(eq=False)
class LStGlobal(LOp):
    tensor: str
    coords: Tuple[Any, ...]
    src_reg: str
    valid: Any = None    # number of valid leading elements along dim 0 (rows), or None
    loc: Loc = NOLOC
    src: Any = None
    kind = "st_global"


@dataclass(eq=False)
class LLdSmem(LOp):
    dst: str
    addr: Any
    dtype: DType
    swizzle: str
    loc: Loc = NOLOC
    src: Any = None
    kind = "ld_smem"


@dataclass(eq=False)
class LStSmem(LOp):
    addr: Any
    src_reg: str
    dtype: DType
    swizzle: str
    loc: Loc = NOLOC
    src: Any = None
    kind = "st_smem"


@dataclass(eq=False)
class LFill(LOp):
    dst: str
    value: Any
    loc: Loc = NOLOC
    src: Any = None
    kind = "fill"


@dataclass(eq=False)
class LEwise(LOp):
    dst: str
    op: str
    args: List[LArg]
    loc: Loc = NOLOC
    src: Any = None
    kind = "ewise"


@dataclass(eq=False)
class LReduce(LOp):
    dst: str
    src_reg: str
    op: str
    idx: Optional[str] = None
    index_offset: Any = 0
    valid: Any = None    # number of valid columns, or None
    loc: Loc = NOLOC
    src: Any = None
    kind = "reduce"


@dataclass(eq=False)
class LConvert(LOp):
    dst: str
    src_reg: str
    loc: Loc = NOLOC
    src: Any = None
    kind = "convert"


@dataclass(eq=False)
class LReg:
    name: str
    shape: Tuple[int, ...]
    dtype: DType


@dataclass(eq=False)
class LBlock:
    """Code executed by a set of warps (a warp-specialized role)."""
    name: str
    warps: Tuple[int, ...]
    body: List[LOp]
    regs: Dict[str, LReg] = field(default_factory=dict)


@dataclass(eq=False)
class LParam:
    name: str
    kind: str                    # "tma" | "gmem" | "scalar"
    dtype: DType
    shape: Tuple[int, ...] = ()


@dataclass(eq=False)
class LKernel:
    name: str
    params: List[LParam]
    grid: Tuple[int, int, int]
    warps: int
    smem_bytes: int
    tmem_cols: int
    prologue: List[LOp]
    blocks: List[LBlock]
    origin: str = "raw"          # "cake" | "raw"
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def param(self, name: str) -> LParam:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def num_ctas(self) -> int:
        g = self.grid
        return int(g[0]) * int(g[1]) * int(g[2])


def walk(ops: List[LOp]) -> Iterator[LOp]:
    for op in ops:
        yield op
        if isinstance(op, LLoop):
            yield from walk(op.body)


def all_ops(k: LKernel) -> Iterator[LOp]:
    yield from walk(k.prologue)
    for b in k.blocks:
        yield from walk(b.body)
