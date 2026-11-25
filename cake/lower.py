"""Lowering: Cake IR -> LIR.

"A schedule states what is to happen and lowering derives how: barrier
addresses, phase bits, TMEM offsets, descriptor encodings, and warp identity
are all computed from the declarations rather than written out by the agent"
(paper Sec. 2.2).  Concretely this pass derives

* the shared-memory layout (pool bases, stage strides, mbarrier slots),
* ``arrive.expect_tx`` byte counts of TMA transaction groups,
* per-role mbarrier phase state (a parity bitmask per barrier, flipped on wait),
* TMEM column numbers and the allocation size,
* MMA operand descriptors (address, extent, swizzle),
* the mbarrier initialization prologue.
"""
from __future__ import annotations

from typing import Dict, List

from .errors import LoweringError
from .expr import DimInt, Var, call, static_call, symbol, wrap
from .ir.core import (Arrive, Barrier, Bcast, Commit, Convert, Ewise, FenceProxy, Fill, LdSmem, LoadG, Loop, Mma,
                      Op, Program, Reduce, RegTile, ScalarParam, SmemRef, StoreG, StSmem, TensorParam, TmaLoad,
                      TmemLoad, TmemRef, Wait, tma_groups, walk)
from .lir.core import (LArg, LBlock, LConvert, LDesc, LEwise, LFence, LFill, LKernel, LLdGlobal, LLdSmem, LLoop,
                       LMbarArrive, LMbarExpectTx, LMbarInit, LMbarWait, LMmaSync, LOp, LParam, LReduce, LReg,
                       LSet, LStGlobal, LStSmem, LTmaLoad, LTmemLoad, LUmma, LUmmaCommit)
from .targets import get_target


def _align(x: int, a: int) -> int:
    if isinstance(x, DimInt):
        return static_call("align", (x, a))
    return (x + a - 1) // a * a


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


class _Lowerer:
    def __init__(self, prog: Program):
        self.p = prog
        cur = 0
        self.pool_base: Dict[int, int] = {}
        for pool in prog.pools:
            cur = _align(cur, 1024)
            self.pool_base[id(pool)] = cur
            cur += pool.nbytes
        cur = _align(cur, 8)
        self.consts: Dict[str, str] = {}
        if isinstance(cur, DimInt):
            # name the (shape-dependent) start of the mbarrier region for readable listings
            self.consts["MBAR_BASE"] = symbol(cur)
            cur = DimInt(int(cur), "MBAR_BASE")
        self.bar_base: Dict[int, int] = {}
        self.mbar_names: Dict[int, str] = {}
        for b in prog.barriers:
            self.bar_base[id(b)] = cur
            for s in range(b.count):
                self.mbar_names[cur + 8 * s] = f"{b.name}[{s}]" if b.count > 1 else b.name
            cur += 8 * b.count
        self.smem_bytes = _align(cur, 128) if cur else 0
        max_col = max((t.col + t.stages * t.width for t in prog.tmems), default=0)
        # TMEM is allocated in power-of-two column counts (>= 32); an oversized request is kept as-is
        # so the launch check (and HW.TMEM in the verifier) report it
        self.tmem_cols = 0 if max_col == 0 else max(32, _next_pow2(max_col))
        self.groups: Dict[int, tuple] = {}
        for role in prog.roles:
            self.groups.update(tma_groups(prog.body[role]))

    # addresses ---------------------------------------------------------------
    def smem_addr(self, ref: SmemRef):
        v = ref.view
        base = self.pool_base[id(v.pool)] + v.offset
        return wrap(base) + wrap(ref.stage) * v.tile_bytes if not isinstance(ref.stage, int) \
            else base + ref.stage * v.tile_bytes

    def mbar(self, b: Barrier, slot):
        base = self.bar_base[id(b)]
        if isinstance(slot, int):
            return base + 8 * slot
        return wrap(base) + wrap(slot) * 8

    def tmem_col(self, ref: TmemRef):
        v = ref.view
        off = ref.cols[0] if ref.cols else 0
        if isinstance(ref.stage, int):
            return v.col + ref.stage * v.width + off
        return wrap(v.col + off) + wrap(ref.stage) * v.width

    def desc(self, ref: SmemRef) -> LDesc:
        v = ref.view
        return LDesc(self.smem_addr(ref), v.shape[0], v.shape[1], v.dtype, v.swizzle)

    @staticmethod
    def phase_reg(b: Barrier) -> str:
        return f"ph_{b.name}"

    # ops ---------------------------------------------------------------------
    def arg(self, x) -> LArg:
        if isinstance(x, RegTile):
            return LArg("reg", x.name)
        if isinstance(x, Bcast):
            return LArg(x.axis, x.reg.name)
        return LArg("imm", x)

    def ops(self, body: List[Op]) -> List[LOp]:
        out: List[LOp] = []
        for op in body:
            out.extend(self.op(op))
        return out

    def op(self, op: Op) -> List[LOp]:
        kw = dict(loc=op.loc, src=op)
        if isinstance(op, Loop):
            return [LLoop(op.var, op.start, op.stop, op.step, self.ops(op.body), **kw)]
        if isinstance(op, TmaLoad):
            leader, total, _ = self.groups[id(op)]
            out: List[LOp] = []
            addr = self.mbar(op.barrier, op.slot)
            if leader:
                out.append(LMbarExpectTx(addr, total, **kw))
            v = op.dst.view
            out.append(LTmaLoad(self.smem_addr(op.dst), op.src.name, tuple(op.coords), v.shape, v.dtype, v.swizzle,
                                addr, op.engine, **kw))
            return out
        if isinstance(op, Wait):
            reg = self.phase_reg(op.barrier)
            parity = (Var(reg) >> wrap(op.slot)) & 1
            flip = Var(reg) ^ (wrap(1) << wrap(op.slot))
            return [LMbarWait(self.mbar(op.barrier, op.slot), parity, **kw), LSet(reg, flip, **kw)]
        if isinstance(op, Arrive):
            return [LMbarArrive(self.mbar(op.barrier, op.slot), **kw)]
        if isinstance(op, Commit):
            return [LUmmaCommit(self.mbar(op.barrier, op.slot), **kw)]
        if isinstance(op, FenceProxy):
            return [LFence(**kw)]
        if isinstance(op, Mma):
            m, n, k = op.mnk
            acc = (not op.init) if isinstance(op.init, bool) else call("not", op.init)
            if isinstance(op.acc, TmemRef):
                return [LUmma(self.tmem_col(op.acc), self.desc(op.a), self.desc(op.b), m, n, k, acc, op.b_major, **kw)]
            return [LMmaSync(op.acc.name, self.desc(op.a), self.desc(op.b), m, n, k, acc, op.b_major, **kw)]
        if isinstance(op, TmemLoad):
            shp = op.src.shape
            return [LTmemLoad(op.dst.name, self.tmem_col(op.src), shp[0], shp[1], **kw)]
        if isinstance(op, LoadG):
            return [LLdGlobal(op.dst.name, op.src.name, tuple(op.coords), **kw)]
        if isinstance(op, StoreG):
            return [LStGlobal(op.dst.name, tuple(op.coords), op.src.name, op.valid, **kw)]
        if isinstance(op, StSmem):
            v = op.dst.view
            return [LStSmem(self.smem_addr(op.dst), op.src.name, v.dtype, v.swizzle, **kw)]
        if isinstance(op, LdSmem):
            v = op.src.view
            return [LLdSmem(op.dst.name, self.smem_addr(op.src), v.dtype, v.swizzle, **kw)]
        if isinstance(op, Fill):
            return [LFill(op.dst.name, op.value, **kw)]
        if isinstance(op, Ewise):
            return [LEwise(op.dst.name, op.op, [self.arg(a) for a in op.args], **kw)]
        if isinstance(op, Reduce):
            return [LReduce(op.dst.name, op.src.name, op.op, op.idx.name if op.idx is not None else None,
                            op.index_offset, op.valid, **kw)]
        if isinstance(op, Convert):
            return [LConvert(op.dst.name, op.src.name, **kw)]
        raise LoweringError(f"cannot lower `{op.kind}`", loc=op.loc)

    def run(self) -> LKernel:
        p = self.p
        params = []
        for x in p.params:
            if isinstance(x, TensorParam):
                params.append(LParam(x.name, x.kind, x.dtype, x.shape))
            elif isinstance(x, ScalarParam):
                params.append(LParam(x.name, "scalar", x.dtype))
        blocks = []
        for role in p.roles:
            body: List[LOp] = []
            waited = []
            for op in walk(p.body[role]):
                if isinstance(op, Wait) and op.barrier not in waited:
                    waited.append(op.barrier)
            for b in waited:
                init = (1 << b.count) - 1 if b.released else 0
                body.append(LSet(self.phase_reg(b), init, loc=b.loc))
            body.extend(self.ops(p.body[role]))
            regs = {r.name: LReg(r.name, r.shape, r.dtype) for r in p.regs if r.role is role}
            blocks.append(LBlock(role.name, role.warps, body, regs))
        prologue: List[LOp] = []
        for b in p.barriers:
            for s in range(b.count):
                prologue.append(LMbarInit(self.bar_base[id(b)] + 8 * s, b.init_count, loc=b.loc))
        layout = {
            "pools": {pool.name: self.pool_base[id(pool)] for pool in p.pools},
            "views": {v.name: {"addr": self.pool_base[id(v.pool)] + v.offset, "stage_bytes": v.tile_bytes,
                               "stages": v.stages, "swizzle": v.swizzle} for v in p.views},
            "barriers": {b.name: self.bar_base[id(b)] for b in p.barriers},
            "tmem": {t.name: {"col": t.col, "width": t.width, "stages": t.stages} for t in p.tmems},
        }
        return LKernel(name=p.name, params=params, grid=p.grid, warps=p.warps, smem_bytes=self.smem_bytes,
                       tmem_cols=self.tmem_cols, prologue=prologue, blocks=blocks, origin="cake", source=p.source,
                       meta={"mbar_names": self.mbar_names, "layout": layout, "scalars": p.meta.get("scalars", {}),
                             "dims": dict(p.dims), "consts": dict(self.consts)})


def lower(prog: Program, target=None) -> LKernel:
    """Lower a Cake IR program to LIR for the simulated GPU."""
    if target is not None:
        get_target(target)
    return _Lowerer(prog).run()
