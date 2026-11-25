"""Agent-facing Cake IR builder: the ``lm`` handle and the ``@cake.schedule()`` decorator.

Every method performs the *local* typing rules of its operation at
construction time (design principle P4) and raises :class:`CakeTypeError`
pointing at the offending source line.  Relational properties (deadlock,
races, hardware contracts, data-flow consistency) are left to the verifier.
"""
from __future__ import annotations

import inspect
import linecache
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import dtypes as D
from ..errors import CakeTypeError, ConstructionError, Loc, capture_loc, register_api_file
from ..expr import DimInt, Expr, Var, call, is_static, static, static_call, wrap
from ..targets import DEFAULT_TARGET, get_target
from .core import (Arrive, Barrier, Bcast, Commit, Convert, Ewise, FenceProxy, Fill, LdSmem, LoadG, Loop, Mma,
                   Op, ParamKind, Pipeline, Program, Reduce, RegTile, Role, ScalarParam, SmemPool, SmemRef,
                   SmemView, StoreG, StSmem, TensorParam, TmaLoad, TmemLoad, TmemRef, TmemView, Wait)

register_api_file(__file__)
register_api_file(__file__.replace("builder.py", "core.py"))

_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]*)?=(?!=)")
_FOR_RE = re.compile(r"^\s*for\s+([A-Za-z_]\w*)\s+in\b")

EWISE_ARITY = {"copy": 1, "neg": 1, "abs": 1, "add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2, "fma": 3}
REDUCE_OPS = ("sum", "min", "max", "argmin", "argmax")
SWIZZLES = ("none", "32B", "64B", "128B")


def _infer(loc: Loc, regex) -> Optional[str]:
    if not loc.lineno:
        return None
    for back in range(0, 4):
        text = loc.text if back == 0 else linecache.getline(loc.filename, loc.lineno - back)
        m = regex.match(text or "")
        if m:
            return m.group(1)
        if back == 0 and regex is _FOR_RE:
            break
    return None


def _static_int(x, what: str, loc: Loc) -> int:
    if isinstance(x, Expr) and not is_static(x):
        raise CakeTypeError(f"{what} must be a static integer, got symbolic `{x}`", loc=loc)
    v = static(x)
    if isinstance(v, bool) or not isinstance(v, int):
        raise CakeTypeError(f"{what} must be an integer, got {v!r}", loc=loc)
    return v


def _tile_compatible(reg_shape, view_shape) -> bool:
    """A register tile matches a smem tile of equal shape, or a 1-D (n,) tile matches a (1, n) view."""
    reg_shape, view_shape = tuple(reg_shape), tuple(view_shape)
    return reg_shape == view_shape or (len(reg_shape) == 1 and view_shape == (1, reg_shape[0]))


def _shape(shape, loc: Loc, what: str = "shape") -> Tuple[int, ...]:
    if isinstance(shape, int):
        shape = (shape,)
    try:
        dims = tuple(_static_int(s, what, loc) for s in shape)
    except TypeError:
        raise CakeTypeError(f"{what} must be a tuple of integers, got {shape!r}", loc=loc) from None
    if not dims or any(d <= 0 for d in dims):
        raise CakeTypeError(f"{what} must have positive extents, got {dims}", loc=loc)
    return dims


class LM:
    """Handle passed as the first argument of a ``@cake.schedule()`` function."""

    # element types
    bf16 = D.bf16
    f16 = D.f16
    f32 = D.f32
    i32 = D.i32
    fp8 = D.fp8
    # parameter annotations
    tma1d = ParamKind("tma", 1)
    tma2d = ParamKind("tma", 2)
    tma3d = ParamKind("tma", 3)
    tma4d = ParamKind("tma", 4)
    gmem = ParamKind("gmem", None)

    def __init__(self, name: str, target, dims: Dict[str, int], tensors: Dict[str, Tuple],
                 scalars: Optional[Dict[str, Any]] = None, source: str = ""):
        self._name = name
        self._target = get_target(target)
        self._dims = dict(dims)
        self._tensor_specs = dict(tensors)
        self._scalars = dict(scalars or {})
        self._source = source
        self._params: List[Any] = []
        self._grid: Optional[Tuple[int, int, int]] = None
        self._warps: Optional[int] = None
        self._pools: List[SmemPool] = []
        self._views: List[SmemView] = []
        self._tmems: List[TmemView] = []
        self._regs: List[RegTile] = []
        self._roles: List[Role] = []
        self._pipes: List[Pipeline] = []
        self._bars: List[Barrier] = []
        self._body: Dict[Role, List[Op]] = {}
        self._cur_role: Optional[Role] = None
        self._stack: List[List[Op]] = []
        self._names: set = set()
        self._vars: set = set()

    # ------------------------------------------------------------------ naming
    def _unique(self, name: str) -> str:
        base, i = name, 1
        while name in self._names:
            name = f"{base}_{i}"
            i += 1
        self._names.add(name)
        return name

    def _mkname(self, loc: Loc, given: Optional[str], prefix: str, n: int) -> str:
        if given is not None:
            if not isinstance(given, str) or not given.isidentifier():
                raise CakeTypeError(f"name must be an identifier, got {given!r}", loc=loc)
            return self._unique(given)
        return self._unique(_infer(loc, _ASSIGN_RE) or f"{prefix}{n}")

    # ------------------------------------------------------------------ static helpers
    def dims(self, *names: str):
        """Concrete problem dimensions from the workload contract, e.g. ``B, N, K, D = lm.dims("B","N","K","D")``."""
        loc = capture_loc()
        out = []
        for n in names:
            if n not in self._dims:
                raise ConstructionError(f"unknown dimension `{n}`", loc=loc,
                                        hint=f"available dimensions: {', '.join(self._dims)}")
            out.append(DimInt(self._dims[n], n))
        return out[0] if len(out) == 1 else tuple(out)

    @property
    def target(self) -> str:
        return self._target.name

    def num_sms(self) -> int:
        """Number of SMs of the target GPU (for persistent grids)."""
        return self._target.num_sms

    def cdiv(self, a, b):
        """Ceiling division (static ints fold to an int)."""
        if is_static(a) and is_static(b):
            return static_call("cdiv", (static(a), static(b)))
        return call("cdiv", a, b)

    def min(self, a, b):
        if is_static(a) and is_static(b):
            return static_call("min", (static(a), static(b)))
        return call("min", a, b)

    def max(self, a, b):
        if is_static(a) and is_static(b):
            return static_call("max", (static(a), static(b)))
        return call("max", a, b)

    def select(self, cond, a, b):
        """``a if cond else b`` for symbolic conditions."""
        r = call("select", cond, a, b)
        return r.value if is_static(r) and isinstance(r, Expr) else r

    def launch(self, grid, warps: int):
        """Launch configuration: ``grid`` (up to 3 static extents) and warps per CTA."""
        loc = capture_loc()
        if self._grid is not None:
            raise ConstructionError("lm.launch() called twice", loc=loc)
        if self._cur_role is not None:
            raise ConstructionError("lm.launch() must be called outside role blocks", loc=loc)
        if isinstance(grid, (int, Expr)):
            grid = (grid,)
        g = [_static_int(x, "grid extent", loc) for x in grid]
        if not 1 <= len(g) <= 3 or any(x < 1 for x in g):
            raise CakeTypeError(f"grid must have 1-3 positive extents, got {tuple(g)}", loc=loc)
        g += [1] * (3 - len(g))
        self._grid = (g[0], g[1], g[2])
        self._warps = _static_int(warps, "warps", loc)
        if self._warps < 1:
            raise CakeTypeError("warps must be >= 1", loc=loc)

    def block_idx(self, axis: int = 0) -> Var:
        """CTA index along ``axis`` (0=x, 1=y, 2=z) as a symbolic value."""
        return Var(f"blockIdx.{'xyz'[axis]}")

    def grid_dim(self, axis: int = 0) -> int:
        if self._grid is None:
            raise ConstructionError("lm.grid_dim() requires lm.launch() first", loc=capture_loc())
        return self._grid[axis]

    # ------------------------------------------------------------------ resources
    def smem(self, nbytes: int, name: Optional[str] = None) -> SmemPool:
        """Declare a shared-memory pool of ``nbytes`` bytes; carve tiles out of it with ``pool.view``."""
        loc = capture_loc()
        self._outside_role("lm.smem", loc)
        n = _static_int(nbytes, "smem size", loc)
        if n <= 0:
            raise CakeTypeError("smem size must be positive", loc=loc)
        pool = SmemPool(self._mkname(loc, name, "smem", len(self._pools)), n, loc)
        pool._lm = self
        self._pools.append(pool)
        return pool

    def _view(self, pool, offset, shape, dtype, stage, swizzle, name) -> SmemView:
        loc = capture_loc()
        self._outside_role("pool.view", loc)
        off = _static_int(offset, "view offset", loc)
        shp = _shape(shape, loc, "view shape")
        if len(shp) != 2:
            raise CakeTypeError(f"shared-memory views are 2-D tiles, got shape {shp}", loc=loc)
        if not isinstance(dtype, D.DType):
            raise CakeTypeError(f"dtype must be an lm dtype (lm.bf16, ...), got {dtype!r}", loc=loc)
        st = _static_int(stage, "stage count", loc)
        if st < 1:
            raise CakeTypeError("stage count must be >= 1", loc=loc)
        if swizzle not in SWIZZLES:
            raise CakeTypeError(f"swizzle must be one of {SWIZZLES}, got {swizzle!r}", loc=loc)
        if off < 0:
            raise CakeTypeError("view offset must be non-negative", loc=loc)
        v = SmemView(self._mkname(loc, name, "view", len(self._views)), pool, off, (shp[0], shp[1]), dtype, st,
                     swizzle, loc)
        v._lm = self
        pool.views.append(v)
        self._views.append(v)
        return v

    def tmem(self, cols: int, width: int, shape, dtype=D.f32, stage: int = 1,
             name: Optional[str] = None) -> TmemView:
        """Declare a tensor-memory accumulator region: columns ``[cols, cols + stage*width)``."""
        loc = capture_loc()
        self._outside_role("lm.tmem", loc)
        c = _static_int(cols, "TMEM column", loc)
        w = _static_int(width, "TMEM width", loc)
        shp = _shape(shape, loc, "TMEM shape")
        st = _static_int(stage, "stage count", loc)
        if len(shp) != 2:
            raise CakeTypeError("TMEM regions are 2-D (lanes, columns)", loc=loc)
        if shp[1] != w:
            raise CakeTypeError(f"TMEM shape {shp} must have {w} columns (= width)", loc=loc)
        if dtype not in (D.f32, D.i32):
            raise CakeTypeError(f"TMEM accumulators are 32-bit (lm.f32 / lm.i32), got {dtype!r}", loc=loc)
        if c < 0 or w < 1 or st < 1:
            raise CakeTypeError("TMEM column/width/stage must be positive", loc=loc)
        t = TmemView(self._mkname(loc, name, "tmem", len(self._tmems)), c, w, (shp[0], shp[1]), dtype, st, loc)
        t._lm = self
        self._tmems.append(t)
        return t

    def regs(self, shape, dtype, init=None, name: Optional[str] = None) -> RegTile:
        """Declare a register tile owned by the current role (``init`` emits a fill)."""
        loc = capture_loc()
        role = self._need_role("lm.regs", loc)
        shp = _shape(shape, loc, "register tile shape")
        if len(shp) > 2:
            raise CakeTypeError("register tiles are 1-D or 2-D", loc=loc)
        if dtype not in (D.f32, D.i32, D.bf16, D.f16):
            raise CakeTypeError(f"unsupported register dtype {dtype!r}", loc=loc)
        reg = RegTile(self._mkname(loc, name, "r", len(self._regs)), shp, dtype, role, init, loc)
        self._regs.append(reg)
        if init is not None:
            self._emit(Fill(reg, self._scalar(init, loc)), loc)
        return reg

    def role(self, warps: Sequence[int], name: Optional[str] = None) -> Role:
        """Declare a warp role: the listed warps execute the role's block as one instruction stream."""
        loc = capture_loc()
        self._outside_role("lm.role", loc)
        if isinstance(warps, int):
            warps = [warps]
        try:
            ws = tuple(_static_int(w, "warp id", loc) for w in warps)
        except TypeError:
            raise CakeTypeError(f"warps must be a list of warp ids, got {warps!r}", loc=loc) from None
        if not ws or any(w < 0 for w in ws) or len(set(ws)) != len(ws):
            raise CakeTypeError(f"warps must be distinct non-negative ids, got {list(ws)}", loc=loc)
        r = Role(self._mkname(loc, name, "role", len(self._roles)), ws, loc)
        r._lm = self
        self._roles.append(r)
        self._body[r] = []
        return r

    def pipeline(self, stages: int, name: Optional[str] = None) -> Pipeline:
        """Declare a software pipeline of ``stages`` stages (phase tracking is derived by lowering)."""
        loc = capture_loc()
        self._outside_role("lm.pipeline", loc)
        st = _static_int(stages, "pipeline stages", loc)
        if st < 1:
            raise CakeTypeError("pipeline stages must be >= 1", loc=loc)
        p = Pipeline(self._mkname(loc, name, "pipe", len(self._pipes)), st, loc)
        self._pipes.append(p)
        return p

    def barrier(self, count: int = 1, prod: Sequence[Role] = (), cons: Sequence[Role] = (),
                init_count: int = 1, pipeline: Optional[Pipeline] = None, released: bool = False,
                name: Optional[str] = None) -> Barrier:
        """Declare ``count`` mbarrier slots handing data from ``prod`` roles to ``cons`` roles.

        ``init_count`` arrivals complete a phase.  ``released=True`` makes every
        slot start released, so the first wait on each slot passes (use it for
        "slot is free" barriers that producers wait on before refilling).
        """
        loc = capture_loc()
        self._outside_role("lm.barrier", loc)
        n = _static_int(count, "barrier count", loc)
        ic = _static_int(init_count, "init_count", loc)
        if n < 1 or ic < 1:
            raise CakeTypeError("barrier count and init_count must be >= 1", loc=loc)
        prod = tuple(prod) if isinstance(prod, (list, tuple)) else (prod,)
        cons = tuple(cons) if isinstance(cons, (list, tuple)) else (cons,)
        for r in prod + cons:
            if not isinstance(r, Role):
                raise CakeTypeError(f"prod/cons must list roles, got {r!r}", loc=loc)
        if pipeline is not None and not isinstance(pipeline, Pipeline):
            raise CakeTypeError("pipeline= must be an lm.pipeline(...)", loc=loc)
        b = Barrier(self._mkname(loc, name, "bar", len(self._bars)), n, prod, cons, ic, pipeline, bool(released), loc)
        self._bars.append(b)
        return b

    # ------------------------------------------------------------------ control flow
    def range(self, start, stop=None, step=1):
        """Traced loop: ``for j in lm.range(0, n):`` records the body once with symbolic ``j``."""
        loc = capture_loc()
        self._need_role("lm.range", loc)
        if stop is None:
            start, stop = 0, start
        if is_static(step) and static(step) == 0:
            raise CakeTypeError("lm.range step must be non-zero", loc=loc)
        pyname = _infer(loc, _FOR_RE) or "i"
        name = pyname
        k = 1
        while name in self._vars or name.startswith("blockIdx"):
            name = f"{pyname}_{k}"
            k += 1
        self._vars.add(name)
        loop = Loop(Var(name), wrap(start), wrap(stop), wrap(step), [])
        self._emit(loop, loc)
        self._stack.append(loop.body)
        try:
            yield loop.var
        finally:
            if self._stack and self._stack[-1] is loop.body:
                self._stack.pop()

    # ------------------------------------------------------------------ synchronization
    def wait(self, barrier: Barrier, stage=0):
        """Block until the next phase of ``barrier`` slot ``stage`` completes (parity derived)."""
        loc = capture_loc()
        self._emit(Wait(self._bar(barrier, loc), self._slot(barrier, stage, loc)), loc)

    def arrive(self, barrier: Barrier, stage=0):
        """Arrive on ``barrier`` slot ``stage`` (counts one of ``init_count`` arrivals)."""
        loc = capture_loc()
        self._emit(Arrive(self._bar(barrier, loc), self._slot(barrier, stage, loc)), loc)

    def commit(self, barrier: Barrier, stage=0):
        """tcgen05.commit: arrive on ``barrier`` slot ``stage`` once all MMAs this role issued so far complete."""
        loc = capture_loc()
        self._emit(Commit(self._bar(barrier, loc), self._slot(barrier, stage, loc)), loc)

    def fence_proxy(self):
        """fence.proxy.async: order this role's generic shared-memory writes before async-proxy reads."""
        loc = capture_loc()
        self._emit(FenceProxy(), loc)

    # ------------------------------------------------------------------ data movement
    def _tma_load(self, view: SmemView, src, coords, stage, barrier, slot, engine: str):
        loc = capture_loc()
        self._need_role(f"{view.name}.tma_load", loc)
        if not isinstance(src, TensorParam):
            raise CakeTypeError(f"tma_load source must be a tensor parameter, got {src!r}", loc=loc)
        if engine == "tma" and src.kind != "tma":
            raise CakeTypeError(f"tensor `{src.name}` is declared LM.gmem; TMA needs a tensor descriptor",
                                loc=loc, hint=f"annotate it as LM.tma{src.ndim}d")
        if src.dtype is not view.dtype:
            raise CakeTypeError(f"TMA cannot convert types: tensor `{src.name}` is {src.dtype.name} but view "
                                f"`{view.name}` is {view.dtype.name}", loc=loc)
        coords = tuple(coords) if isinstance(coords, (list, tuple)) else (coords,)
        if len(coords) != src.ndim:
            raise CakeTypeError(f"`{src.name}` has rank {src.ndim}; got {len(coords)} coordinates", loc=loc)
        if not isinstance(barrier, Barrier):
            raise CakeTypeError("tma_load needs barrier=<completion barrier>", loc=loc)
        ref = self._smem_ref(view, stage, loc)
        if slot is None:
            slot = stage if barrier.count > 1 else 0
        self._emit(TmaLoad(ref, src, tuple(wrap(c) if isinstance(c, Expr) else c for c in coords), barrier,
                           self._slot(barrier, slot, loc), engine), loc)

    def load(self, dst: RegTile, src: TensorParam, coords):
        """Synchronous global load of a ``dst``-shaped box at ``coords`` (out-of-range elements read 0)."""
        loc = capture_loc()
        self._need_role("lm.load", loc)
        self._own(dst, loc)
        if not isinstance(src, TensorParam):
            raise CakeTypeError(f"lm.load source must be a tensor parameter, got {src!r}", loc=loc)
        coords = tuple(coords) if isinstance(coords, (list, tuple)) else (coords,)
        if len(coords) != src.ndim:
            raise CakeTypeError(f"`{src.name}` has rank {src.ndim}; got {len(coords)} coordinates", loc=loc)
        if len(dst.shape) > src.ndim:
            raise CakeTypeError(f"cannot load a rank-{len(dst.shape)} tile from rank-{src.ndim} `{src.name}`",
                                loc=loc)
        if (src.dtype.kind == "int") != (dst.dtype.kind == "int"):
            raise CakeTypeError(f"cannot load {src.dtype.name} data into {dst.dtype.name} registers", loc=loc)
        self._emit(LoadG(dst, src, coords), loc)

    def store(self, dst: TensorParam, coords, src: RegTile, valid=None):
        """Global store of register tile ``src`` at ``coords``; ``valid`` limits the stored leading rows."""
        loc = capture_loc()
        self._need_role("lm.store", loc)
        self._own(src, loc)
        if not isinstance(dst, TensorParam):
            raise CakeTypeError(f"lm.store destination must be a tensor parameter, got {dst!r}", loc=loc)
        coords = tuple(coords) if isinstance(coords, (list, tuple)) else (coords,)
        if len(coords) != dst.ndim:
            raise CakeTypeError(f"`{dst.name}` has rank {dst.ndim}; got {len(coords)} coordinates", loc=loc)
        if src.dtype is not dst.dtype:
            raise CakeTypeError(f"store of {src.dtype.name} registers into {dst.dtype.name} tensor `{dst.name}`",
                                loc=loc, hint="convert first with lm.convert(dst_regs, src_regs)")
        self._emit(StoreG(dst, coords, src, valid), loc)

    def st_smem(self, dst, src: RegTile):
        """Generic-proxy store of registers into a shared-memory tile (needs fence_proxy before async reads)."""
        loc = capture_loc()
        self._need_role("lm.st_smem", loc)
        self._own(src, loc)
        ref = self._as_smem_ref(dst, loc)
        if not _tile_compatible(src.shape, ref.shape):
            raise CakeTypeError(f"st_smem shape mismatch {src.shape} vs {ref.shape}", loc=loc)
        self._emit(StSmem(ref, src), loc)

    def ld_smem(self, dst: RegTile, src):
        """Load a shared-memory tile into registers."""
        loc = capture_loc()
        self._need_role("lm.ld_smem", loc)
        self._own(dst, loc)
        ref = self._as_smem_ref(src, loc)
        if not _tile_compatible(dst.shape, ref.shape):
            raise CakeTypeError(f"ld_smem shape mismatch {dst.shape} vs {ref.shape}", loc=loc)
        self._emit(LdSmem(dst, ref), loc)

    def tmem_load(self, dst: RegTile, src):
        """tcgen05.ld: copy an accumulator stage (or a ``.part``) from TMEM into registers."""
        loc = capture_loc()
        self._need_role("lm.tmem_load", loc)
        self._own(dst, loc)
        ref = self._as_tmem_ref(src, loc)
        if tuple(dst.shape) != tuple(ref.shape):
            raise CakeTypeError(f"tmem_load destination shape {dst.shape} != source shape {ref.shape}", loc=loc)
        if dst.dtype is not ref.view.dtype:
            raise CakeTypeError(f"tmem_load into {dst.dtype.name} registers from {ref.view.dtype.name} TMEM",
                                loc=loc)
        self._emit(TmemLoad(dst, ref), loc)

    # ------------------------------------------------------------------ compute
    def mma(self, acc, a, b, init=True, b_major: str = "K"):
        """Tensor-core MMA ``acc (+)= a @ b^T`` (``b_major="K"``: b is (N, K)) or ``a @ b`` (``"N"``).

        ``acc`` in TMEM issues tcgen05.mma (asynchronous; signal completion with
        ``lm.commit``); a register ``acc`` issues a synchronous warp-level MMA.
        ``init`` (bool or expression) overwrites instead of accumulating.
        """
        loc = capture_loc()
        self._need_role("lm.mma", loc)
        ra, rb = self._as_smem_ref(a, loc), self._as_smem_ref(b, loc)
        if ra.dtype is not rb.dtype:
            raise CakeTypeError(f"MMA operand types differ: {ra.dtype.name} vs {rb.dtype.name}", loc=loc)
        if ra.dtype.name not in ("bf16", "f16", "fp8"):
            raise CakeTypeError(f"tensor cores take bf16/f16/fp8 operands, got {ra.dtype.name}", loc=loc)
        if b_major not in ("K", "N"):
            raise CakeTypeError("b_major must be 'K' or 'N'", loc=loc)
        m, k = ra.shape
        if b_major == "K":
            n, kb = rb.shape
        else:
            kb, n = rb.shape
        if k != kb:
            raise CakeTypeError(f"MMA reduction extents differ: a is {ra.shape}, b is {rb.shape} "
                                f"({b_major}-major)", loc=loc)
        if isinstance(acc, RegTile):
            self._own(acc, loc)
            ashape, adt = acc.shape, acc.dtype
            acc_ref: Any = acc
        else:
            acc_ref = self._as_tmem_ref(acc, loc)
            ashape, adt = acc_ref.shape, acc_ref.view.dtype
        if tuple(ashape) != (m, n):
            raise CakeTypeError(f"accumulator shape {tuple(ashape)} != ({m}, {n}) = a{ra.shape} x "
                                f"b{rb.shape}", loc=loc)
        if adt is not D.f32:
            raise CakeTypeError(f"MMA accumulates in f32, accumulator is {adt.name}", loc=loc)
        self._emit(Mma(acc_ref, ra, rb, self._scalar(init, loc), b_major), loc)

    def fill(self, dst: RegTile, value):
        """Set every element of ``dst`` to ``value``."""
        loc = capture_loc()
        self._need_role("lm.fill", loc)
        self._own(dst, loc)
        self._emit(Fill(dst, self._scalar(value, loc)), loc)

    def ewise(self, dst: RegTile, op: str, a, b=None, c=None):
        """Elementwise ``dst = op(a, b, c)`` on CUDA cores.

        ops: copy/neg/abs (1 operand), add/sub/mul/min/max (2), fma (3: a*b+c).
        Operands: register tiles of dst's shape, scalars, or broadcast vectors
        ``v.row()`` / ``v.col()``.
        """
        loc = capture_loc()
        self._need_role("lm.ewise", loc)
        self._own(dst, loc)
        if op not in EWISE_ARITY:
            raise CakeTypeError(f"unknown elementwise op `{op}`; expected one of {list(EWISE_ARITY)}", loc=loc)
        args = [x for x in (a, b, c) if x is not None]
        if len(args) != EWISE_ARITY[op]:
            raise CakeTypeError(f"`{op}` takes {EWISE_ARITY[op]} operand(s), got {len(args)}", loc=loc)
        out = [self._operand(x, dst, loc) for x in args]
        self._emit(Ewise(dst, op, out), loc)

    def reduce(self, dst: RegTile, src: RegTile, op: str, idx: Optional[RegTile] = None, index_offset=0,
               valid=None):
        """Row reduction of an (M, N) tile *accumulated into* ``dst`` (M,).

        ``argmin``/``argmax`` also update ``idx`` (M,) i32 with column index +
        ``index_offset`` (strictly-better wins, so earlier columns win ties).
        ``valid`` masks columns ``>= valid`` (use it for ragged tails).
        """
        loc = capture_loc()
        self._need_role("lm.reduce", loc)
        self._own(dst, loc)
        self._own(src, loc)
        if op not in REDUCE_OPS:
            raise CakeTypeError(f"unknown reduction `{op}`; expected one of {REDUCE_OPS}", loc=loc)
        if len(src.shape) != 2:
            raise CakeTypeError(f"reduce source must be 2-D, got {src.shape}", loc=loc)
        if tuple(dst.shape) != (src.shape[0],):
            raise CakeTypeError(f"reduce destination must have shape ({src.shape[0]},), got {dst.shape}", loc=loc)
        if op in ("argmin", "argmax"):
            if idx is None:
                raise CakeTypeError(f"`{op}` needs idx=<(M,) i32 register tile>", loc=loc)
            self._own(idx, loc)
            if tuple(idx.shape) != (src.shape[0],) or idx.dtype is not D.i32:
                raise CakeTypeError(f"idx must be an ({src.shape[0]},) lm.i32 tile", loc=loc)
        self._emit(Reduce(dst, src, op, idx, self._scalar(index_offset, loc),
                          None if valid is None else self._scalar(valid, loc)), loc)

    def convert(self, dst: RegTile, src: RegTile):
        """Elementwise type conversion between register tiles of equal shape (rounds to dst precision)."""
        loc = capture_loc()
        self._need_role("lm.convert", loc)
        self._own(dst, loc)
        self._own(src, loc)
        if tuple(dst.shape) != tuple(src.shape):
            raise CakeTypeError(f"convert shape mismatch {dst.shape} vs {src.shape}", loc=loc)
        self._emit(Convert(dst, src), loc)

    # ------------------------------------------------------------------ internals
    def _outside_role(self, what: str, loc: Loc) -> None:
        if self._cur_role is not None:
            raise ConstructionError(f"{what} declares a resource and must be called outside role blocks", loc=loc)

    def _need_role(self, what: str, loc: Loc) -> Role:
        if self._cur_role is None:
            raise ConstructionError(f"{what} must be issued inside a role block (`with role:`)", loc=loc,
                                    hint="every operation is executed by a declared warp role")
        return self._cur_role

    def _emit(self, op: Op, loc: Loc) -> None:
        role = self._need_role(op.kind, loc)
        op.role = role
        op.loc = loc
        self._stack[-1].append(op)

    def _enter_role(self, role: Role) -> None:
        loc = capture_loc()
        if self._cur_role is not None:
            raise ConstructionError(f"role `{role.name}` entered inside role `{self._cur_role.name}`", loc=loc)
        self._cur_role = role
        self._stack = [self._body[role]]

    def _exit_role(self, role: Role) -> None:
        self._cur_role = None
        self._stack = []

    def _own(self, reg, loc: Loc) -> None:
        if not isinstance(reg, RegTile):
            raise CakeTypeError(f"expected a register tile (lm.regs), got {reg!r}", loc=loc)
        if reg.role is not self._cur_role:
            raise CakeTypeError(f"register tile `{reg.name}` belongs to role `{reg.role.name}` but is used in role "
                                f"`{self._cur_role.name}`", loc=loc,
                                hint="registers are private to a role; exchange data through shared or tensor memory")

    def _scalar(self, x, loc: Loc):
        if isinstance(x, (RegTile, Bcast, SmemView, SmemRef, TmemView, TmemRef)):
            raise CakeTypeError(f"expected a scalar, got {x!r}", loc=loc)
        if isinstance(x, Expr):
            return x
        if isinstance(x, (bool, int, float)):
            return x
        try:
            return wrap(x)
        except Exception:
            raise CakeTypeError(f"expected a scalar, got {x!r}", loc=loc) from None

    def _operand(self, x, dst: RegTile, loc: Loc):
        if isinstance(x, RegTile):
            self._own(x, loc)
            if tuple(x.shape) != tuple(dst.shape):
                if len(x.shape) == 1 and len(dst.shape) == 2:
                    raise CakeTypeError(f"ambiguous broadcast of `{x.name}` {x.shape} to {dst.shape}", loc=loc,
                                        hint=f"use {x.name}.row() (one value per column) or {x.name}.col() "
                                             f"(one value per row)")
                raise CakeTypeError(f"operand `{x.name}` shape {x.shape} != destination shape {dst.shape}", loc=loc)
            return x
        if isinstance(x, Bcast):
            self._own(x.reg, loc)
            if len(dst.shape) != 2 or len(x.reg.shape) != 1:
                raise CakeTypeError("row()/col() broadcast a 1-D tile over a 2-D destination", loc=loc)
            want = dst.shape[1] if x.axis == "row" else dst.shape[0]
            if x.reg.shape[0] != want:
                raise CakeTypeError(f"`{x}` has length {x.reg.shape[0]}, destination needs {want}", loc=loc)
            return x
        return self._scalar(x, loc)

    def _bar(self, b, loc: Loc) -> Barrier:
        if not isinstance(b, Barrier):
            raise CakeTypeError(f"expected a barrier, got {b!r}", loc=loc)
        return b

    def _slot(self, b: Barrier, stage, loc: Loc):
        if isinstance(stage, Expr) and not is_static(stage):
            return stage
        s = _static_int(stage, "barrier slot", loc)
        if not 0 <= s < b.count:
            raise CakeTypeError(f"slot {s} out of range for barrier `{b.name}` with {b.count} slot(s)", loc=loc)
        return s

    def _smem_ref(self, view: SmemView, stage, loc: Optional[Loc] = None) -> SmemRef:
        loc = loc or capture_loc()
        if isinstance(stage, Expr) and not is_static(stage):
            return SmemRef(view, stage)
        s = _static_int(stage, "stage index", loc)
        if not 0 <= s < view.stages:
            raise CakeTypeError(f"stage {s} out of range for view `{view.name}` with {view.stages} stage(s)", loc=loc)
        return SmemRef(view, s)

    def _as_smem_ref(self, x, loc: Loc) -> SmemRef:
        if isinstance(x, SmemRef):
            return x
        if isinstance(x, SmemView):
            if x.stages != 1:
                raise CakeTypeError(f"view `{x.name}` has {x.stages} stages; index it, e.g. {x.name}[stage]",
                                    loc=loc)
            return SmemRef(x, 0)
        raise CakeTypeError(f"expected a shared-memory tile, got {x!r}", loc=loc)

    def _tmem_ref(self, view: TmemView, stage, loc: Optional[Loc] = None) -> TmemRef:
        loc = loc or capture_loc()
        if isinstance(stage, Expr) and not is_static(stage):
            return TmemRef(view, stage)
        s = _static_int(stage, "stage index", loc)
        if not 0 <= s < view.stages:
            raise CakeTypeError(f"stage {s} out of range for TMEM region `{view.name}` with {view.stages} "
                                f"stage(s)", loc=loc)
        return TmemRef(view, s)

    def _tmem_part(self, ref: TmemRef, start: int, width: int) -> TmemRef:
        loc = capture_loc()
        s = _static_int(start, "column start", loc)
        w = _static_int(width, "column width", loc)
        base = ref.cols[0] if ref.cols else 0
        limit = ref.cols[1] if ref.cols else ref.view.width
        if s < 0 or w < 1 or s + w > limit:
            raise CakeTypeError(f"column range [{s}, {s + w}) outside the {limit}-column accumulator", loc=loc)
        return TmemRef(ref.view, ref.stage, (base + s, w))

    def _as_tmem_ref(self, x, loc: Loc) -> TmemRef:
        if isinstance(x, TmemRef):
            return x
        if isinstance(x, TmemView):
            if x.stages != 1:
                raise CakeTypeError(f"TMEM region `{x.name}` has {x.stages} stages; index it, e.g. "
                                    f"{x.name}[stage]", loc=loc)
            return TmemRef(x, 0)
        raise CakeTypeError(f"expected a TMEM accumulator, got {x!r}", loc=loc)

    def _finish(self, fn_name: str) -> Program:
        if self._cur_role is not None:
            raise ConstructionError(f"role block `{self._cur_role.name}` was not closed")
        if not self._roles:
            raise ConstructionError("the schedule declares no roles", hint="declare warp roles with lm.role(warps=[...])")
        if self._grid is None:
            self._grid = (1, 1, 1)
        if self._warps is None:
            self._warps = max(max(r.warps) for r in self._roles) + 1
        return Program(name=fn_name, target=self._target.name, dims=dict(self._dims), params=list(self._params),
                       grid=self._grid, warps=self._warps, pools=self._pools, views=self._views,
                       tmems=self._tmems, regs=self._regs, roles=self._roles, pipelines=self._pipes,
                       barriers=self._bars, body=self._body, source=self._source)


# ----------------------------------------------------------------------------- decorator
class Schedule:
    """A ``@cake.schedule()`` function; ``build`` instantiates it for concrete shapes."""

    def __init__(self, fn, target: Optional[str] = None, name: Optional[str] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.target = target
        try:
            self.source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            self.source = ""

    def build(self, dims: Dict[str, int], tensors: Dict[str, Tuple], target: Optional[str] = None,
              scalars: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> Program:
        target = target or self.target or DEFAULT_TARGET
        lm = LM(self.name, target, dims, tensors, scalars, source if source is not None else self.source)
        sig = inspect.signature(self.fn)
        ps = list(sig.parameters.values())
        if not ps:
            raise ConstructionError("a schedule function takes `lm` as its first parameter")
        args: List[Any] = [lm]
        for i, p in enumerate(ps[1:]):
            ann = p.annotation
            if isinstance(ann, str):
                try:
                    ann = eval(ann, {"LM": LM, **getattr(self.fn, "__globals__", {})})  # noqa: S307
                except Exception:
                    ann = None
            if isinstance(ann, ParamKind):
                spec = lm._tensor_specs.get(p.name)
                if spec is None:
                    raise ConstructionError(f"parameter `{p.name}` is not a tensor of the workload contract",
                                            hint=f"contract tensors: {', '.join(lm._tensor_specs)}")
                shape, dt = spec
                if ann.kind == "tma" and ann.ndim is not None and ann.ndim != len(shape):
                    raise CakeTypeError(f"`{p.name}` is annotated LM.tma{ann.ndim}d but the tensor has rank "
                                        f"{len(shape)}")
                tp = TensorParam(p.name, ann.kind, tuple(int(s) for s in shape), D.get(dt), i)
                lm._params.append(tp)
                args.append(tp)
            elif isinstance(ann, D.DType):
                sp = ScalarParam(p.name, ann, Var(p.name))
                lm._params.append(sp)
                args.append(sp.var)
            else:
                raise CakeTypeError(f"parameter `{p.name}` needs a Cake annotation",
                                    hint="LM.tma2d / LM.tma3d / LM.gmem for tensors, LM.i32 for scalars")
        self.fn(*args)
        prog = lm._finish(self.name)
        prog.meta["scalars"] = dict(scalars or {})
        return prog

    __call__ = build


def schedule(fn=None, *, target: Optional[str] = None, name: Optional[str] = None):
    """Decorator turning a Python function into a Cake IR schedule."""
    if fn is not None and callable(fn):
        return Schedule(fn, target, name)

    def deco(f):
        return Schedule(f, target, name)

    return deco
