"""Cake IR data model: declared resources, warp roles, barriers and typed operations.

A program records *how the machine should be driven* (paper Sec. 2.2): which
warps take which roles, which buffers are staged how deeply, which barrier
gates which hand-off, which instruction form consumes which operand.  It does
not record barrier addresses, phase bits, transaction byte counts, TMEM
offsets or warp predicates -- those are derived by lowering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..dtypes import DType
from ..errors import Loc
from ..expr import Var, to_py

NOLOC = Loc()


# ----------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ParamKind:
    """Annotation for tensor parameters: ``X: LM.tma3d`` or ``OUT: LM.gmem``."""
    kind: str               # "tma" | "gmem"
    ndim: Optional[int]

    def __repr__(self) -> str:
        return f"LM.{self.kind}{self.ndim}d" if self.kind == "tma" else "LM.gmem"


@dataclass(eq=False)
class TensorParam:
    name: str
    kind: str
    shape: Tuple[int, ...]
    dtype: DType
    index: int

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class ScalarParam:
    name: str
    dtype: DType
    var: Var


# ----------------------------------------------------------------------------- resources
@dataclass(eq=False)
class SmemPool:
    name: str
    nbytes: int
    loc: Loc = NOLOC
    views: List["SmemView"] = field(default_factory=list)
    _lm: Any = None

    def view(self, offset, shape, dtype, stage=1, swizzle="128B", name=None) -> "SmemView":
        """Carve a ``stage``-deep 2-D tile out of this pool at byte ``offset`` (stage stride = tile bytes)."""
        return self._lm._view(self, offset, shape, dtype, stage, swizzle, name)


@dataclass(eq=False)
class SmemView:
    name: str
    pool: SmemPool
    offset: int
    shape: Tuple[int, int]
    dtype: DType
    stages: int
    swizzle: str
    loc: Loc = NOLOC
    _lm: Any = None

    @property
    def tile_bytes(self) -> int:
        return self.shape[0] * self.shape[1] * self.dtype.nbytes

    @property
    def total_bytes(self) -> int:
        return self.tile_bytes * self.stages

    def __getitem__(self, stage) -> "SmemRef":
        return self._lm._smem_ref(self, stage)

    def tma_load(self, src, coords, stage=0, barrier=None, slot=None):
        """Asynchronous TMA copy of one tile of ``src`` at ``coords`` into stage ``stage``.

        Completion is signalled on ``barrier`` (slot defaults to ``stage``); the
        transaction byte count is derived by lowering.
        """
        return self._lm._tma_load(self, src, coords, stage, barrier, slot, "tma")

    def async_load(self, src, coords, stage=0, barrier=None, slot=None):
        """cp.async (Ampere-style) copy into stage ``stage``; completion signalled on ``barrier``."""
        return self._lm._tma_load(self, src, coords, stage, barrier, slot, "cp.async")

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class SmemRef:
    view: SmemView
    stage: Any

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self.view.dtype

    def __repr__(self) -> str:
        return f"{self.view.name}[{to_py(self.stage)}]"


@dataclass(eq=False)
class TmemView:
    name: str
    col: int
    width: int
    shape: Tuple[int, int]
    dtype: DType
    stages: int
    loc: Loc = NOLOC
    _lm: Any = None

    def __getitem__(self, stage) -> "TmemRef":
        return self._lm._tmem_ref(self, stage)

    def part(self, start: int, width: int) -> "TmemRef":
        """Column sub-range of a single-stage accumulator (``acc[s].part`` for multi-stage ones)."""
        return self._lm._tmem_part(self._lm._as_tmem_ref(self, self._lm_loc()), start, width)

    def _lm_loc(self):
        from ..errors import capture_loc
        return capture_loc()

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class TmemRef:
    view: TmemView
    stage: Any
    cols: Optional[Tuple[int, int]] = None   # (start, width) sub-range within the stage

    @property
    def shape(self):
        if self.cols is None:
            return self.view.shape
        return (self.view.shape[0], self.cols[1])

    def part(self, start: int, width: int) -> "TmemRef":
        """Static column sub-range ``[start, start + width)`` of this accumulator stage."""
        return self.view._lm._tmem_part(self, start, width)

    def __repr__(self) -> str:
        s = f"{self.view.name}[{to_py(self.stage)}]"
        if self.cols is not None:
            s += f"[:, {self.cols[0]}:{self.cols[0] + self.cols[1]}]"
        return s


@dataclass(eq=False)
class RegTile:
    name: str
    shape: Tuple[int, ...]
    dtype: DType
    role: Any
    init: Any = None
    loc: Loc = NOLOC

    def row(self) -> "Bcast":
        """Broadcast this length-N vector along rows of an (M, N) tile (one value per column)."""
        return Bcast(self, "row")

    def col(self) -> "Bcast":
        """Broadcast this length-M vector along columns of an (M, N) tile (one value per row)."""
        return Bcast(self, "col")

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class Bcast:
    reg: RegTile
    axis: str

    def __repr__(self) -> str:
        return f"{self.reg.name}.{self.axis}()"


@dataclass(eq=False)
class Role:
    name: str
    warps: Tuple[int, ...]
    loc: Loc = NOLOC
    _lm: Any = None

    def __enter__(self):
        self._lm._enter_role(self)
        return self

    def __exit__(self, *exc):
        self._lm._exit_role(self)
        return False

    def __repr__(self) -> str:
        return self.name


@dataclass(eq=False)
class Pipeline:
    name: str
    stages: int
    loc: Loc = NOLOC


@dataclass(eq=False)
class Barrier:
    name: str
    count: int                      # number of slots (one mbarrier per slot)
    prod: Tuple[Role, ...]
    cons: Tuple[Role, ...]
    init_count: int                 # arrivals that complete one phase
    pipeline: Optional[Pipeline]
    released: bool                  # slots start released: the first wait on each slot passes
    loc: Loc = NOLOC

    def __repr__(self) -> str:
        return self.name


# ----------------------------------------------------------------------------- operations
class Op:
    kind = "op"
    role: Role
    loc: Loc


@dataclass(eq=False)
class Loop(Op):
    var: Var
    start: Any
    stop: Any
    step: Any
    body: List[Op]
    role: Any = None
    loc: Loc = NOLOC
    kind = "loop"


@dataclass(eq=False)
class TmaLoad(Op):
    dst: SmemRef
    src: TensorParam
    coords: Tuple[Any, ...]
    barrier: Barrier
    slot: Any
    engine: str = "tma"
    role: Any = None
    loc: Loc = NOLOC
    kind = "tma_load"

    @property
    def nbytes(self) -> int:
        return self.dst.view.tile_bytes


@dataclass(eq=False)
class Wait(Op):
    barrier: Barrier
    slot: Any
    role: Any = None
    loc: Loc = NOLOC
    kind = "wait"


@dataclass(eq=False)
class Arrive(Op):
    barrier: Barrier
    slot: Any
    role: Any = None
    loc: Loc = NOLOC
    kind = "arrive"


@dataclass(eq=False)
class Commit(Op):
    barrier: Barrier
    slot: Any
    role: Any = None
    loc: Loc = NOLOC
    kind = "commit"


@dataclass(eq=False)
class FenceProxy(Op):
    role: Any = None
    loc: Loc = NOLOC
    kind = "fence_proxy"


@dataclass(eq=False)
class Mma(Op):
    acc: Union[TmemRef, RegTile]
    a: SmemRef
    b: SmemRef
    init: Any
    b_major: str = "K"
    role: Any = None
    loc: Loc = NOLOC
    kind = "mma"

    @property
    def mnk(self) -> Tuple[int, int, int]:
        m, k = self.a.shape
        n = self.b.shape[0] if self.b_major == "K" else self.b.shape[1]
        return m, n, k


@dataclass(eq=False)
class TmemLoad(Op):
    dst: RegTile
    src: TmemRef
    role: Any = None
    loc: Loc = NOLOC
    kind = "tmem_load"


@dataclass(eq=False)
class LoadG(Op):
    dst: RegTile
    src: TensorParam
    coords: Tuple[Any, ...]
    role: Any = None
    loc: Loc = NOLOC
    kind = "load"


@dataclass(eq=False)
class StoreG(Op):
    dst: TensorParam
    coords: Tuple[Any, ...]
    src: RegTile
    valid: Any = None
    role: Any = None
    loc: Loc = NOLOC
    kind = "store"


@dataclass(eq=False)
class StSmem(Op):
    dst: SmemRef
    src: RegTile
    role: Any = None
    loc: Loc = NOLOC
    kind = "st_smem"


@dataclass(eq=False)
class LdSmem(Op):
    dst: RegTile
    src: SmemRef
    role: Any = None
    loc: Loc = NOLOC
    kind = "ld_smem"


@dataclass(eq=False)
class Fill(Op):
    dst: RegTile
    value: Any
    role: Any = None
    loc: Loc = NOLOC
    kind = "fill"


@dataclass(eq=False)
class Ewise(Op):
    dst: RegTile
    op: str
    args: List[Any]
    role: Any = None
    loc: Loc = NOLOC
    kind = "ewise"


@dataclass(eq=False)
class Reduce(Op):
    dst: RegTile
    src: RegTile
    op: str
    idx: Optional[RegTile] = None
    index_offset: Any = 0
    valid: Any = None
    role: Any = None
    loc: Loc = NOLOC
    kind = "reduce"


@dataclass(eq=False)
class Convert(Op):
    dst: RegTile
    src: RegTile
    role: Any = None
    loc: Loc = NOLOC
    kind = "convert"


SYNC_KINDS = ("wait", "arrive", "commit")


# ----------------------------------------------------------------------------- program
@dataclass(eq=False)
class Program:
    name: str
    target: str
    dims: Dict[str, int]
    params: List[Any]
    grid: Tuple[int, int, int]
    warps: int
    pools: List[SmemPool]
    views: List[SmemView]
    tmems: List[TmemView]
    regs: List[RegTile]
    roles: List[Role]
    pipelines: List[Pipeline]
    barriers: List[Barrier]
    body: Dict[Role, List[Op]]
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def tensor(self, name: str) -> TensorParam:
        for p in self.params:
            if isinstance(p, TensorParam) and p.name == name:
                return p
        raise KeyError(name)

    def ops(self, role: Optional[Role] = None) -> Iterator[Op]:
        roles = [role] if role is not None else self.roles
        for r in roles:
            yield from walk(self.body.get(r, []))

    @property
    def smem_bytes(self) -> int:
        return sum(p.nbytes for p in self.pools)

    @property
    def num_ctas(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]


def walk(ops: List[Op]) -> Iterator[Op]:
    for op in ops:
        yield op
        if isinstance(op, Loop):
            yield from walk(op.body)


def tma_groups(ops: List[Op]) -> Dict[int, Tuple[bool, int, int]]:
    """Transaction groups: consecutive TMA loads signalling the same barrier slot.

    Returns ``{id(op): (is_group_leader, group_bytes, group_size)}``.  A group
    performs one arrival with the summed transaction byte count (derived).
    """
    out: Dict[int, Tuple[bool, int, int]] = {}

    def scan(body: List[Op]) -> None:
        i = 0
        while i < len(body):
            op = body[i]
            if isinstance(op, Loop):
                scan(op.body)
                i += 1
                continue
            if isinstance(op, TmaLoad):
                j = i + 1
                while (j < len(body) and isinstance(body[j], TmaLoad) and body[j].barrier is op.barrier
                       and to_py(body[j].slot) == to_py(op.slot)):
                    j += 1
                group = body[i:j]
                total = sum(g.nbytes for g in group)
                for n, g in enumerate(group):
                    out[id(g)] = (n == 0, total, len(group))
                i = j
                continue
            i += 1

    scan(ops)
    return out
