"""Print LIR as ``@cake.raw_kernel()`` source (the direct-code arm's representation).

Printing a lowered Cake program shows exactly what the agent would have to
write by hand without Cake IR: byte addresses, parity bookkeeping, expect-tx
byte counts, TMEM columns and descriptor swizzles.
"""
from __future__ import annotations

from typing import List

from ..expr import BinOp, Call, Const, Var, symbol
from .core import (LArg, LConvert, LDesc, LEwise, LFence, LFill, LKernel, LLdGlobal, LLdSmem, LLoop,
                   LMbarArrive, LMbarExpectTx, LMbarInit, LMbarWait, LMmaSync, LOp, LReduce, LSet, LStGlobal,
                   LStSmem, LTmaLoad, LTmemLoad, LUmma, LUmmaCommit)

_BIDX = {"blockIdx.x": "bx", "blockIdx.y": "by", "blockIdx.z": "bz"}


def _lit(v) -> str:
    return symbol(v)


def _tup(xs) -> str:
    xs = tuple(xs)
    return "(" + ", ".join(_lit(x) for x in xs) + ("," if len(xs) == 1 else "") + ")"


class _P:
    def __init__(self, k: LKernel):
        self.k = k
        self.loopvars: set = set()

    def e(self, x) -> str:
        if isinstance(x, Const):
            return _lit(x.value)
        if isinstance(x, Var):
            if x.name in _BIDX:
                return _BIDX[x.name]
            if x.name.startswith("gridDim."):
                return _lit(self.k.grid["xyz".index(x.name[-1])])
            if x.name in self.loopvars:
                return x.name
            return f'k.reg("{x.name}")'
        if isinstance(x, BinOp):
            op = {"and": "&", "or": "|"}.get(x.op, x.op)
            return f"({self.e(x.a)} {op} {self.e(x.b)})"
        if isinstance(x, Call):
            if x.fn == "not":
                return f"({self.e(x.args[0])} == 0)"
            return f"k.{x.fn}({', '.join(self.e(a) for a in x.args)})"
        return _lit(x)

    def dt(self, d) -> str:
        return f"k.{d.name}"

    def desc(self, d: LDesc) -> str:
        return (f'k.smem_desc({self.e(d.addr)}, {_lit(d.rows)}, {_lit(d.cols)}, {self.dt(d.dtype)}, '
                f'"{d.swizzle}")')

    def arg(self, a: LArg) -> str:
        if a.kind == "reg":
            return a.value
        if a.kind in ("row", "col"):
            return f"{a.value}.{a.kind}()"
        return self.e(a.value)

    def op(self, op: LOp, ind: str, out: List[str]) -> None:
        a = out.append
        if isinstance(op, LLoop):
            self.loopvars.add(op.var.name)
            a(f"{ind}for {op.var.name} in k.range({self.e(op.start)}, {self.e(op.stop)}, {self.e(op.step)}):")
            if not op.body:
                a(f"{ind}    pass")
            for o in op.body:
                self.op(o, ind + "    ", out)
        elif isinstance(op, LSet):
            a(f'{ind}k.set("{op.name}", {self.e(op.value)})')
        elif isinstance(op, LMbarInit):
            a(f"{ind}k.mbar_init({self.e(op.addr)}, {_lit(op.count)})")
        elif isinstance(op, LMbarArrive):
            a(f"{ind}k.mbar_arrive({self.e(op.addr)})")
        elif isinstance(op, LMbarExpectTx):
            a(f"{ind}k.mbar_expect_tx({self.e(op.addr)}, {self.e(op.nbytes)})")
        elif isinstance(op, LMbarWait):
            a(f"{ind}k.mbar_wait({self.e(op.addr)}, {self.e(op.parity)})")
        elif isinstance(op, LFence):
            a(f"{ind}k.fence_proxy_async()")
        elif isinstance(op, LTmaLoad):
            fn = "tma_load" if op.engine == "tma" else "cp_async"
            a(f"{ind}k.{fn}({self.e(op.dst)}, {op.tensor}, ({', '.join(self.e(c) for c in op.coords)},), "
              f'{_tup(op.box)}, "{op.swizzle}", {self.e(op.mbar)})')
        elif isinstance(op, LUmma):
            a(f"{ind}k.umma({self.e(op.d_col)}, {self.desc(op.a)}, {self.desc(op.b)}, {_lit(op.m)}, "
              f"{_lit(op.n)}, {_lit(op.k)}, "
              f'{self.e(op.accumulate)}, "{op.b_major}")')
        elif isinstance(op, LMmaSync):
            a(f"{ind}k.mma_sync({op.acc}, {self.desc(op.a)}, {self.desc(op.b)}, {_lit(op.m)}, "
              f"{_lit(op.n)}, {_lit(op.k)}, "
              f'{self.e(op.accumulate)}, "{op.b_major}")')
        elif isinstance(op, LUmmaCommit):
            a(f"{ind}k.umma_commit({self.e(op.mbar)})")
        elif isinstance(op, LTmemLoad):
            a(f"{ind}k.tmem_load({op.dst}, {self.e(op.col)}, {_lit(op.lanes)}, {_lit(op.ncols)})")
        elif isinstance(op, LLdGlobal):
            a(f"{ind}k.ld_global({op.dst}, {op.tensor}, ({', '.join(self.e(c) for c in op.coords)},))")
        elif isinstance(op, LStGlobal):
            v = "" if op.valid is None else f", valid={self.e(op.valid)}"
            a(f"{ind}k.st_global({op.tensor}, ({', '.join(self.e(c) for c in op.coords)},), {op.src_reg}{v})")
        elif isinstance(op, LLdSmem):
            a(f'{ind}k.ld_shared({op.dst}, {self.e(op.addr)}, {self.dt(op.dtype)}, "{op.swizzle}")')
        elif isinstance(op, LStSmem):
            a(f'{ind}k.st_shared({self.e(op.addr)}, {op.src_reg}, {self.dt(op.dtype)}, "{op.swizzle}")')
        elif isinstance(op, LFill):
            a(f"{ind}k.fill({op.dst}, {self.e(op.value)})")
        elif isinstance(op, LEwise):
            a(f'{ind}k.ewise({op.dst}, "{op.op}", {", ".join(self.arg(x) for x in op.args)})')
        elif isinstance(op, LReduce):
            extra = ""
            if op.idx is not None:
                extra += f", idx={op.idx}"
            extra += f", index_offset={self.e(op.index_offset)}"
            if op.valid is not None:
                extra += f", valid={self.e(op.valid)}"
            a(f'{ind}k.reduce({op.dst}, {op.src_reg}, "{op.op}"{extra})')
        elif isinstance(op, LConvert):
            a(f"{ind}k.convert({op.dst}, {op.src_reg})")
        else:  # pragma: no cover
            a(f"{ind}# <unprintable {op.kind}>")


def to_raw_source(k: LKernel, name: str = None) -> str:
    p = _P(k)
    name = name or k.name
    params = ", ".join(f"{x.name}: LM.{'gmem' if x.kind == 'gmem' else f'tma{len(x.shape)}d'}"
                       for x in k.params if x.kind != "scalar")
    out = ["@cake.raw_kernel()", f"def {name}(k, {params}):"]
    dims = k.meta.get("dims") or {}
    if dims:
        out.append(f"    {', '.join(dims)} = k.dims({', '.join(repr(d) for d in dims)})"
                   if len(dims) > 1 else f"    {list(dims)[0]} = k.dims({list(dims)[0]!r})")
    for cname, cexpr in (k.meta.get("consts") or {}).items():
        out.append(f"    {cname} = {cexpr}")
    out.append(f"    k.launch(grid={_tup(k.grid)}, warps={_lit(k.warps)}, smem_bytes={_lit(k.smem_bytes)}, "
               f"tmem_cols={_lit(k.tmem_cols)})")
    out.append("    bx, by, bz = k.block_idx(0), k.block_idx(1), k.block_idx(2)")
    names = k.meta.get("mbar_names", {})
    if names:
        out.append("    # mbarriers: " + ", ".join(f"{v}@{a:#x}" for a, v in sorted(names.items())))
    out.append("    with k.prologue():")
    for op in k.prologue:
        p.op(op, "        ", out)
    if not k.prologue:
        out.append("        pass")
    for b in k.blocks:
        out.append(f'    with k.warps({list(b.warps)}, "{b.name}"):')
        for r in b.regs.values():
            out.append(f'        {r.name} = k.regs({_tup(r.shape)}, k.{r.dtype.name}, name="{r.name}")')
        body_start = len(out)
        for op in b.body:
            p.op(op, "        ", out)
        if len(out) == body_start and not b.regs:
            out.append("        pass")
    return "\n".join(out) + "\n"
