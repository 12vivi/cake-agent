"""Direct-code authoring surface for the control arm (``@cake.raw_kernel()``).

This is the CUDA/PTX-level counterpart of Cake IR: the author manages shared
memory byte addresses, mbarrier objects and their phase parities, transaction
byte counts, TMEM columns, MMA descriptors (including swizzle modes) and warp
predicates by hand.  Only assembler-level checks run at build time; everything
else surfaces as runtime behaviour (hang, illegal address, wrong results).
"""
from __future__ import annotations

import inspect
import textwrap
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import dtypes as D
from ..errors import ConstructionError, capture_loc, register_api_file
from ..expr import DimInt, Expr, Var, call, is_static, static, static_call, wrap
from ..targets import DEFAULT_TARGET, get_target
from .core import (LArg, LBlock, LConvert, LDesc, LEwise, LFence, LFill, LKernel, LLdGlobal, LLdSmem, LLoop,
                   LMbarArrive, LMbarExpectTx, LMbarInit, LMbarWait, LMmaSync, LOp, LParam, LReduce, LReg, LSet,
                   LStGlobal, LStSmem, LTmaLoad, LTmemLoad, LUmma, LUmmaCommit)

register_api_file(__file__)

EWISE_ARITY = {"copy": 1, "neg": 1, "abs": 1, "add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2, "fma": 3}


class RawTensor:
    def __init__(self, name: str, kind: str, shape: Tuple[int, ...], dtype: D.DType):
        self.name, self.kind, self.shape, self.dtype = name, kind, shape, dtype

    def __repr__(self) -> str:
        return self.name


class RawReg:
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: D.DType, block: "LBlock"):
        self.name, self.shape, self.dtype, self.block = name, shape, dtype, block

    def row(self) -> LArg:
        return LArg("row", self.name)

    def col(self) -> LArg:
        return LArg("col", self.name)

    def __repr__(self) -> str:
        return self.name


class K:
    """Handle passed as the first argument of a ``@cake.raw_kernel()`` function."""

    bf16, f16, f32, i32, fp8 = D.bf16, D.f16, D.f32, D.i32, D.fp8

    def __init__(self, name: str, target, dims: Dict[str, int], tensors: Dict[str, Tuple], source: str = ""):
        self._name = name
        self._target = get_target(target)
        self._dims = dict(dims)
        self._tensor_specs = dict(tensors)
        self._source = source
        self._params: List[LParam] = []
        self._grid: Optional[Tuple[int, int, int]] = None
        self._warps = 0
        self._smem = 0
        self._tmem = 0
        self._prologue: List[LOp] = []
        self._blocks: List[LBlock] = []
        self._block: Optional[LBlock] = None
        self._stack: List[List[LOp]] = []
        self._vars: set = set()

    # ------------------------------------------------------------------ config
    def dims(self, *names: str):
        out = []
        for n in names:
            if n not in self._dims:
                raise ConstructionError(f"unknown dimension `{n}`", loc=capture_loc())
            out.append(DimInt(self._dims[n], n))
        return out[0] if len(out) == 1 else tuple(out)

    def num_sms(self) -> int:
        return self._target.num_sms

    def cdiv(self, a, b):
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

    def align(self, x, a):
        """Round ``x`` up to a multiple of ``a``."""
        if is_static(x) and is_static(a):
            return static_call("align", (static(x), static(a)))
        return call("align", x, a)

    def select(self, c, a, b):
        r = call("select", c, a, b)
        return r.value if is_static(r) and isinstance(r, Expr) else r

    def launch(self, grid, warps: int, smem_bytes: int, tmem_cols: int = 0):
        """Grid, warps per CTA, dynamic shared memory bytes and TMEM columns to allocate."""
        if isinstance(grid, (int, Expr)):
            grid = (grid,)
        g = [static(x) for x in grid] + [1] * (3 - len(tuple(grid)))
        self._grid = (g[0], g[1], g[2])
        self._warps = static(warps)
        self._smem = static(smem_bytes)
        self._tmem = static(tmem_cols)

    def block_idx(self, axis: int = 0) -> Var:
        return Var(f"blockIdx.{'xyz'[axis]}")

    def grid_dim(self, axis: int = 0) -> int:
        if self._grid is None:
            raise ConstructionError("k.grid_dim() before k.launch()", loc=capture_loc())
        return self._grid[axis]

    # ------------------------------------------------------------------ structure
    @contextmanager
    def prologue(self):
        """Code run once by warp 0 before __syncthreads (initialize mbarriers here)."""
        if self._block is not None:
            raise ConstructionError("prologue inside a warp block", loc=capture_loc())
        self._stack = [self._prologue]
        self._block = LBlock("prologue", (), self._prologue)
        try:
            yield
        finally:
            self._block = None
            self._stack = []

    @contextmanager
    def warps(self, ids: Sequence[int], name: Optional[str] = None):
        """``if (warp_id in ids) { ... }`` -- code executed by these warps."""
        loc = capture_loc()
        if self._block is not None:
            raise ConstructionError("nested warp blocks", loc=loc)
        if isinstance(ids, int):
            ids = [ids]
        ws = tuple(int(static(w)) for w in ids)
        for b in self._blocks:
            if set(b.warps) & set(ws):
                raise ConstructionError(f"warps {sorted(set(b.warps) & set(ws))} already have a block", loc=loc,
                                        hint="put all code of a warp set in one block")
        blk = LBlock(name or f"warps{'_'.join(map(str, ws))}", ws, [])
        self._blocks.append(blk)
        self._block = blk
        self._stack = [blk.body]
        try:
            yield blk
        finally:
            self._block = None
            self._stack = []

    def regs(self, shape, dtype, name: Optional[str] = None) -> RawReg:
        loc = capture_loc()
        blk = self._need_block(loc)
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(int(static(s)) for s in shape)
        nm = name or f"r{sum(len(b.regs) for b in self._blocks)}"
        if nm in blk.regs:
            raise ConstructionError(f"register `{nm}` declared twice", loc=loc)
        blk.regs[nm] = LReg(nm, shape, dtype)
        return RawReg(nm, shape, dtype, blk)

    def set(self, name: str, value):
        """Scalar register assignment (e.g. phase bits)."""
        self._emit(LSet(name, value))

    def reg(self, name: str) -> Var:
        return Var(name)

    def range(self, start, stop=None, step=1):
        loc = capture_loc()
        self._need_block(loc)
        if stop is None:
            start, stop = 0, start
        name = f"i{len(self._vars)}"
        import re
        m = re.match(r"^\s*for\s+([A-Za-z_]\w*)\s+in\b", loc.text or "")
        if m and m.group(1) not in self._vars:
            name = m.group(1)
        self._vars.add(name)
        loop = LLoop(Var(name), wrap(start), wrap(stop), wrap(step), [], loc=loc)
        self._stack[-1].append(loop)
        self._stack.append(loop.body)
        try:
            yield loop.var
        finally:
            if self._stack and self._stack[-1] is loop.body:
                self._stack.pop()

    # ------------------------------------------------------------------ sync
    def mbar_init(self, addr, count: int):
        self._emit(LMbarInit(addr, count))

    def mbar_arrive(self, addr):
        self._emit(LMbarArrive(addr))

    def mbar_expect_tx(self, addr, nbytes):
        """mbarrier.arrive.expect_tx: one arrival + ``nbytes`` expected transaction bytes."""
        self._emit(LMbarExpectTx(addr, nbytes))

    def mbar_wait(self, addr, parity):
        """Spin until the phase with ``parity`` (0/1) of the mbarrier at ``addr`` has completed."""
        self._emit(LMbarWait(addr, parity))

    def fence_proxy_async(self):
        self._emit(LFence())

    # ------------------------------------------------------------------ memory
    def tma_load(self, dst, tensor: RawTensor, coords, box, swizzle: str, mbar):
        """cp.async.bulk.tensor: copy ``box`` (rows, cols) of ``tensor`` at ``coords`` to smem ``dst``."""
        self._tensor(tensor)
        self._emit(LTmaLoad(dst, tensor.name, tuple(coords), (int(box[0]), int(box[1])), tensor.dtype, swizzle,
                            mbar, "tma"))

    def cp_async(self, dst, tensor: RawTensor, coords, box, swizzle: str, mbar):
        self._tensor(tensor)
        self._emit(LTmaLoad(dst, tensor.name, tuple(coords), (int(box[0]), int(box[1])), tensor.dtype, swizzle,
                            mbar, "cp.async"))

    def ld_global(self, dst: RawReg, tensor: RawTensor, coords):
        self._reg(dst)
        self._tensor(tensor)
        self._emit(LLdGlobal(dst.name, tensor.name, tuple(coords)))

    def st_global(self, tensor: RawTensor, coords, src: RawReg, valid=None):
        self._reg(src)
        self._tensor(tensor)
        if src.dtype is not tensor.dtype:
            raise ConstructionError(f"st_global of {src.dtype.name} registers to {tensor.dtype.name} tensor",
                                    loc=capture_loc())
        self._emit(LStGlobal(tensor.name, tuple(coords), src.name, valid))

    def ld_shared(self, dst: RawReg, addr, dtype, swizzle: str = "none"):
        self._reg(dst)
        self._emit(LLdSmem(dst.name, addr, dtype, swizzle))

    def st_shared(self, addr, src: RawReg, dtype, swizzle: str = "none"):
        self._reg(src)
        self._emit(LStSmem(addr, src.name, dtype, swizzle))

    def tmem_load(self, dst: RawReg, col, lanes: int, ncols: int):
        """tcgen05.ld + tcgen05.wait::ld of ``lanes`` x ``ncols`` fp32 values starting at TMEM column ``col``."""
        self._reg(dst)
        self._emit(LTmemLoad(dst.name, col, int(lanes), int(ncols)))

    # ------------------------------------------------------------------ compute
    def smem_desc(self, addr, rows: int, cols: int, dtype, swizzle: str) -> LDesc:
        """Shared-memory matrix descriptor (start address, extent, element type, swizzle mode)."""
        return LDesc(addr, int(rows), int(cols), dtype, swizzle)

    def umma(self, d_col, a: LDesc, b: LDesc, m: int, n: int, k: int, accumulate, b_major: str = "K"):
        """tcgen05.mma.cta_group::1: TMEM[d_col] (+)= A x B^T (asynchronous)."""
        self._emit(LUmma(d_col, a, b, int(m), int(n), int(k), accumulate, b_major))

    def umma_commit(self, mbar):
        """tcgen05.commit.mbarrier::arrive::one: arrive once prior MMAs of this thread complete."""
        self._emit(LUmmaCommit(mbar))

    def mma_sync(self, acc: RawReg, a: LDesc, b: LDesc, m: int, n: int, k: int, accumulate, b_major: str = "K"):
        self._reg(acc)
        self._emit(LMmaSync(acc.name, a, b, int(m), int(n), int(k), accumulate, b_major))

    def fill(self, dst: RawReg, value):
        self._reg(dst)
        self._emit(LFill(dst.name, value))

    def ewise(self, dst: RawReg, op: str, a, b=None, c=None):
        self._reg(dst)
        if op not in EWISE_ARITY:
            raise ConstructionError(f"unknown elementwise op `{op}`", loc=capture_loc())
        args = [x for x in (a, b, c) if x is not None]
        if len(args) != EWISE_ARITY[op]:
            raise ConstructionError(f"`{op}` takes {EWISE_ARITY[op]} operands", loc=capture_loc())
        out = []
        for x in args:
            if isinstance(x, RawReg):
                self._reg(x)
                out.append(LArg("reg", x.name))
            elif isinstance(x, LArg):
                out.append(x)
            else:
                out.append(LArg("imm", x))
        self._emit(LEwise(dst.name, op, out))

    def reduce(self, dst: RawReg, src: RawReg, op: str, idx: Optional[RawReg] = None, index_offset=0, valid=None):
        self._reg(dst)
        self._reg(src)
        if idx is not None:
            self._reg(idx)
        self._emit(LReduce(dst.name, src.name, op, idx.name if idx is not None else None, index_offset, valid))

    def convert(self, dst: RawReg, src: RawReg):
        self._reg(dst)
        self._reg(src)
        self._emit(LConvert(dst.name, src.name))

    # ------------------------------------------------------------------ internals
    def _need_block(self, loc) -> LBlock:
        if self._block is None:
            raise ConstructionError("instruction outside a warp block (`with k.warps([...]):`)", loc=loc)
        return self._block

    def _emit(self, op: LOp) -> None:
        loc = capture_loc()
        self._need_block(loc)
        op.loc = loc
        self._stack[-1].append(op)

    def _reg(self, r) -> None:
        loc = capture_loc()
        if not isinstance(r, RawReg):
            raise ConstructionError(f"expected a register tile, got {r!r}", loc=loc)
        if self._block is None or r.block is not self._block:
            raise ConstructionError(f"register `{r.name}` is not declared in this warp block", loc=loc)

    def _tensor(self, t) -> None:
        if not isinstance(t, RawTensor):
            raise ConstructionError(f"expected a tensor parameter, got {t!r}", loc=capture_loc())

    def _finish(self) -> LKernel:
        if self._grid is None:
            raise ConstructionError("k.launch(...) was never called")
        if not self._blocks:
            raise ConstructionError("no warp blocks")
        return LKernel(name=self._name, params=self._params, grid=self._grid, warps=self._warps,
                       smem_bytes=self._smem, tmem_cols=self._tmem, prologue=self._prologue, blocks=self._blocks,
                       origin="raw", source=self._source, meta={"dims": dict(self._dims)})


class RawKernelFn:
    def __init__(self, fn, target: Optional[str] = None, name: Optional[str] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.target = target
        try:
            self.source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            self.source = ""

    def build(self, dims: Dict[str, int], tensors: Dict[str, Tuple], target: Optional[str] = None,
              scalars=None, source: Optional[str] = None) -> LKernel:
        from ..ir.core import ParamKind
        k = K(self.name, target or self.target or DEFAULT_TARGET, dims, tensors,
              source if source is not None else self.source)
        ps = list(inspect.signature(self.fn).parameters.values())
        args: List[Any] = [k]
        for p in ps[1:]:
            ann = p.annotation
            if not isinstance(ann, ParamKind):
                raise ConstructionError(f"parameter `{p.name}` needs LM.tma2d/LM.tma3d/LM.gmem")
            spec = tensors.get(p.name)
            if spec is None:
                raise ConstructionError(f"parameter `{p.name}` is not a tensor of the workload contract")
            shape, dt = spec
            dt = D.get(dt)
            k._params.append(LParam(p.name, ann.kind, dt, tuple(int(s) for s in shape)))
            args.append(RawTensor(p.name, ann.kind, tuple(int(s) for s in shape), dt))
        self.fn(*args)
        return k._finish()

    __call__ = build


def raw_kernel(fn=None, *, target: Optional[str] = None, name: Optional[str] = None):
    if fn is not None and callable(fn):
        return RawKernelFn(fn, target, name)

    def deco(f):
        return RawKernelFn(f, target, name)

    return deco
