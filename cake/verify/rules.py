"""Verifier rules (paper Table 1) and the rule registry.

Seed rules ship enabled.  ``LIBRARY`` rules are known-good analyses that are
*not* part of the seed harness: in offline mode the harness-evolution loop can
only enable them after recurring runtime failures justify them and corpus
gating passes (an LLM-backed run synthesizes new rule code instead).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional

from ..errors import Loc
from ..ir.core import (Arrive, Commit, Ewise, LoadG, Loop, Mma, Program, Reduce, TmaLoad, TmemLoad, TmemRef,
                       Wait, walk)
from ..sim.memory import SWIZZLE_ALIGN, SWIZZLE_SPAN
from ..targets import Target
from .facts import Facts, analyze, representative_ctas
from .findings import Finding


@dataclass
class Rule:
    id: str
    category: str
    disposition: str
    description: str
    check: Callable[["Ctx"], Iterable[Finding]]
    origin: str = "seed"          # "seed" | "library" | "evolved"
    source: str = ""              # source code for evolved rules

    def run(self, ctx: "Ctx") -> List[Finding]:
        return list(self.check(ctx) or [])


SEED: Dict[str, Rule] = {}
LIBRARY: Dict[str, Rule] = {}


def rule(id: str, category: str, disposition: str, description: str, library: bool = False):
    def deco(fn):
        (LIBRARY if library else SEED)[id] = Rule(id, category, disposition, description, fn,
                                                  "library" if library else "seed")
        return fn
    return deco


class Ctx:
    """What a rule sees: the program, the target and (lazily) abstract-execution facts."""

    def __init__(self, program: Program, target: Target, max_ops: int = 400_000):
        self.program = program
        self.target = target
        self.max_ops = max_ops
        self._rule: Optional[Rule] = None

    @cached_property
    def facts(self) -> List[Facts]:
        return [analyze(self.program, b, self.max_ops) for b in representative_ctas(self.program)]

    def all_facts(self, *kinds: str):
        seen = set()
        for fs in self.facts:
            for f in fs.of(*kinds):
                key = (f.kind, f.loc.lineno if f.loc else 0, f.role, f.message)
                if key not in seen:
                    seen.add(key)
                    yield f

    def ops(self):
        return self.program.ops()

    def finding(self, message: str, loc: Optional[Loc] = None, hint: Optional[str] = None, **subject) -> Finding:
        r = self._rule
        return Finding(r.id, r.category, r.disposition, message, loc, subject, hint)


def _role_of(op) -> str:
    return op.role.name if op.role is not None else "?"


# ============================================================================ schedule semantics
@rule("SEM.ROLE_OVERLAP", "schedule_semantics", "gate", "Every warp belongs to at most one role")
def _role_overlap(ctx: Ctx):
    owner = {}
    for r in ctx.program.roles:
        for w in r.warps:
            if w in owner:
                yield ctx.finding(f"warp {w} is assigned to both `{owner[w].name}` and `{r.name}`", r.loc,
                                  role=r.name, warp=w)
            owner[w] = r


@rule("SEM.WARP_RANGE", "schedule_semantics", "gate", "Role warps exist in the launch configuration")
def _warp_range(ctx: Ctx):
    W = ctx.program.warps
    for r in ctx.program.roles:
        bad = [w for w in r.warps if w >= W]
        if bad:
            yield ctx.finding(f"role `{r.name}` uses warps {bad} but the CTA launches {W} warps", r.loc, role=r.name,
                              hint=f"raise lm.launch(..., warps=...) to at least {max(r.warps) + 1}")


@rule("SEM.BARRIER_ROLE", "schedule_semantics", "gate",
      "Barriers are only signalled by declared producers and waited on by declared consumers")
def _barrier_role(ctx: Ctx):
    for op in ctx.ops():
        if isinstance(op, (TmaLoad, Arrive, Commit)):
            b = op.barrier
            if op.role not in b.prod:
                yield ctx.finding(f"role `{_role_of(op)}` signals `{b.name}`, whose producers are "
                                  f"{[r.name for r in b.prod]}", op.loc, barrier=b.name, role=_role_of(op),
                                  hint=f"add `{_role_of(op)}` to prod=[...] or signal another barrier")
        elif isinstance(op, Wait):
            b = op.barrier
            if op.role not in b.cons:
                yield ctx.finding(f"role `{_role_of(op)}` waits on `{b.name}`, whose consumers are "
                                  f"{[r.name for r in b.cons]}", op.loc, barrier=b.name, role=_role_of(op),
                                  hint=f"add `{_role_of(op)}` to cons=[...]")


@rule("SEM.BARRIER_UNSIGNALED", "schedule_semantics", "gate", "Every waited barrier is signalled somewhere")
def _unsignaled(ctx: Ctx):
    signalled, waited = set(), {}
    for op in ctx.ops():
        if isinstance(op, (TmaLoad, Arrive, Commit)):
            signalled.add(id(op.barrier))
        elif isinstance(op, Wait):
            waited.setdefault(id(op.barrier), op)
    for b in ctx.program.barriers:
        if id(b) in waited and id(b) not in signalled and not b.released:
            op = waited[id(b)]
            yield ctx.finding(f"`{b.name}` is waited on but no operation ever arrives on it", op.loc, barrier=b.name,
                              hint="add lm.arrive/lm.commit (or a tma_load with barrier=) in a producer role")


@rule("SEM.PIPELINE_COUNT", "schedule_semantics", "gate", "A pipelined barrier has one slot per pipeline stage")
def _pipeline_count(ctx: Ctx):
    for b in ctx.program.barriers:
        if b.pipeline is not None and b.count != b.pipeline.stages:
            yield ctx.finding(f"barrier `{b.name}` has {b.count} slot(s) but pipeline `{b.pipeline.name}` has "
                              f"{b.pipeline.stages} stages", b.loc, barrier=b.name)


@rule("SEM.INDEX_RANGE", "schedule_semantics", "gate", "Stage and slot indices stay within declared counts")
def _index_range(ctx: Ctx):
    for f in ctx.all_facts("slot_range", "stage_range", "eval_error"):
        yield ctx.finding(f.message, f.loc, role=f.role, **{k: v for k, v in f.subject.items()})


# ============================================================================ hardware conformance
@rule("HW.SMEM_CAPACITY", "hardware_conformance", "gate", "Shared memory fits the per-CTA limit")
def _smem_capacity(ctx: Ctx):
    p, t = ctx.program, ctx.target
    total = sum(pl.nbytes for pl in p.pools) + 8 * sum(b.count for b in p.barriers) + 1024 * max(0, len(p.pools) - 1)
    if total > t.smem_per_cta:
        loc = p.pools[0].loc if p.pools else None
        yield ctx.finding(f"{total} bytes of shared memory (pools + mbarriers) exceed the {t.smem_per_cta}-byte "
                          f"per-CTA limit of {t.name}", loc, bytes=total, limit=t.smem_per_cta,
                          hint="reduce pipeline stages or tile sizes")


@rule("HW.TMEM", "hardware_conformance", "gate", "Tensor memory exists on the target and fits 512 columns")
def _tmem(ctx: Ctx):
    t = ctx.target
    for tm in ctx.program.tmems:
        if not t.tmem:
            yield ctx.finding(f"TMEM region `{tm.name}` declared, but {t.name} has no tensor memory", tm.loc,
                              resource=tm.name, hint="accumulate in registers (lm.regs) with a warp-level MMA")
            continue
        end = tm.col + tm.stages * tm.width
        if end > t.tmem_cols:
            yield ctx.finding(f"TMEM region `{tm.name}` spans columns [{tm.col}, {end}) beyond the {t.tmem_cols} "
                              f"columns of {t.name}", tm.loc, resource=tm.name,
                              hint="reduce the accumulator width or the number of accumulator stages")
        if tm.shape[0] not in (64, 128):
            yield ctx.finding(f"TMEM region `{tm.name}` has {tm.shape[0]} lanes; tcgen05 accumulators use 64 or "
                              f"128 lanes", tm.loc, resource=tm.name)


@rule("HW.TMA", "hardware_conformance", "gate", "TMA is only used where the target has it, with legal boxes")
def _tma(ctx: Ctx):
    t = ctx.target
    for op in ctx.ops():
        if isinstance(op, TmaLoad):
            v = op.dst.view
            if op.engine == "tma" and not t.tma:
                yield ctx.finding(f"tma_load on {t.name}, which has no TMA engine", op.loc, resource=v.name,
                                  hint="use view.async_load(...) (cp.async) on this architecture")
            if op.engine == "cp.async" and not t.cp_async:
                yield ctx.finding(f"async_load on {t.name}, which has no cp.async", op.loc, resource=v.name)
            rows, cols = v.shape
            if rows > 256 or cols > 256 or (cols * v.dtype.nbytes) % 16:
                yield ctx.finding(f"TMA box {v.shape} of `{v.name}` is illegal: each box dimension must be <= 256 "
                                  f"elements and the inner dimension a multiple of 16 bytes", op.loc,
                                  resource=v.name)


@rule("HW.MMA", "hardware_conformance", "gate",
      "MMA instruction form, shape and operand types are supported by the target")
def _mma(ctx: Ctx):
    t = ctx.target
    for op in ctx.ops():
        if not isinstance(op, Mma):
            continue
        m, n, k = op.mnk
        dt = op.a.dtype.name
        if dt not in t.mma_dtypes:
            yield ctx.finding(f"{dt} tensor-core MMA is not supported on {t.name} (supported: "
                              f"{list(t.mma_dtypes)})", op.loc, dtype=dt)
        if isinstance(op.acc, TmemRef):
            if not t.tcgen05:
                yield ctx.finding(f"MMA into TMEM (tcgen05.mma) on {t.name}, which has no tcgen05", op.loc,
                                  hint="accumulate into a register tile instead")
            elif m not in (64, 128) or n % 16 or not 16 <= n <= 256 or k % 16:
                yield ctx.finding(f"tcgen05.mma shape M={m}, N={n}, K={k} is unsupported (M in {{64,128}}, "
                                  f"N a multiple of 16 in [16,256], K a multiple of 16)", op.loc, m=m, n=n, k=k)
        else:
            if m % 16 or n % 8 or k % 16:
                yield ctx.finding(f"warp-level MMA shape M={m}, N={n}, K={k} is not a multiple of the "
                                  f"m16n8k16 instruction", op.loc, m=m, n=n, k=k)
    for op in ctx.ops():
        if isinstance(op, Commit) and not t.tcgen05:
            yield ctx.finding(f"lm.commit (tcgen05.commit) on {t.name}", op.loc,
                              hint="synchronous MMAs need no commit; use lm.arrive")


@rule("HW.LAUNCH", "hardware_conformance", "gate", "Warps per CTA and the register file fit the target")
def _launch(ctx: Ctx):
    p, t = ctx.program, ctx.target
    if p.warps > t.max_warps_per_cta:
        yield ctx.finding(f"{p.warps} warps per CTA exceed the limit of {t.max_warps_per_cta}", None, warps=p.warps)
    total = 0
    for r in p.roles:
        total += min(_regs_per_thread(p, r), t.max_regs_per_thread) * 32 * len(r.warps)
    used = {w for r in p.roles for w in r.warps}
    total += (p.warps - len(used)) * 32 * 24
    if total > t.regs_per_sm:
        yield ctx.finding(f"the CTA needs {total} registers, more than the {t.regs_per_sm}-register file",
                          None, registers=total, hint="shrink register tiles (process accumulators in .part chunks)")


def _regs_per_thread(p: Program, role) -> int:
    regs = 24
    threads = 32 * len(role.warps)
    for rt in p.regs:
        if rt.role is role:
            n = 1
            for d in rt.shape:
                n *= d
            regs += -(-n // threads)
    return regs


@rule("HW.ALIGN", "hardware_conformance", "gate", "Shared-memory views satisfy TMA and swizzle-atom alignment")
def _align(ctx: Ctx):
    for v in ctx.program.views:
        need = SWIZZLE_ALIGN[v.swizzle]
        if v.offset % need:
            yield ctx.finding(f"view `{v.name}` starts at pool offset {v.offset}, not a multiple of {need} bytes "
                              f"required by swizzle {v.swizzle}", v.loc, resource=v.name,
                              hint=f"round the offset up to a multiple of {need}")
        elif v.stages > 1 and v.tile_bytes % need:
            yield ctx.finding(f"stage stride of `{v.name}` ({v.tile_bytes} bytes) breaks the {need}-byte alignment "
                              f"of later stages", v.loc, resource=v.name)


@rule("HW.SPILL", "performance_analysis", "report", "Register tiles fit in 255 registers per thread")
def _spill(ctx: Ctx):
    p, t = ctx.program, ctx.target
    for r in p.roles:
        regs = _regs_per_thread(p, r)
        if regs > t.max_regs_per_thread:
            yield ctx.finding(f"role `{r.name}` needs ~{regs} registers per thread (> {t.max_regs_per_thread}); "
                              f"the excess spills to local memory", r.loc, role=r.name, regs=regs,
                              hint="process the accumulator in column chunks with acc[...].part(start, width) or "
                                   "give the role more warps")


# ============================================================================ data consistency
@rule("DC.SWIZZLE", "data_consistency", "gate", "A view's swizzle mode is compatible with its row width")
def _swizzle(ctx: Ctx):
    for v in ctx.program.views:
        span = SWIZZLE_SPAN[v.swizzle]
        row = v.shape[1] * v.dtype.nbytes
        if span and row % span:
            yield ctx.finding(f"view `{v.name}` has {row}-byte rows, incompatible with swizzle {v.swizzle} "
                              f"(rows must be a multiple of {span} bytes)", v.loc, resource=v.name,
                              hint="pick a smaller swizzle or a wider tile")


@rule("DC.UNINIT", "data_consistency", "gate", "No value is read before it is produced")
def _uninit(ctx: Ctx):
    for f in ctx.all_facts("uninit_smem", "uninit_tmem", "uninit_reg"):
        yield ctx.finding(f.message, f.loc, role=f.role, **f.subject)


@rule("DC.TMEM_ALIAS", "data_consistency", "gate", "TMEM regions do not overlap")
def _tmem_alias(ctx: Ctx):
    ts = ctx.program.tmems
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            a0, a1 = a.col, a.col + a.stages * a.width
            b0, b1 = b.col, b.col + b.stages * b.width
            if a0 < b1 and b0 < a1:
                yield ctx.finding(f"TMEM regions `{a.name}` [{a0},{a1}) and `{b.name}` [{b0},{b1}) overlap", b.loc,
                                  resource=b.name)


@rule("DC.TMA_RANGE", "data_consistency", "gate", "TMA boxes intersect their tensors")
def _tma_range(ctx: Ctx):
    for f in ctx.all_facts("tma_oob"):
        yield ctx.finding(f.message, f.loc, role=f.role, hint="check the tile coordinates and grid mapping",
                          **f.subject)


# ============================================================================ program safety
@rule("SAFE.DEADLOCK", "program_safety", "gate", "Every wait can be satisfied (no synchronization deadlock)")
def _deadlock(ctx: Ctx):
    for f in ctx.all_facts("deadlock"):
        hint = f.subject.get("hint")
        subj = {k: v for k, v in f.subject.items() if k in ("barrier", "slot")}
        yield ctx.finding(f.message, f.loc, hint=hint, role=f.role, **subj)


@rule("SAFE.RACE", "program_safety", "gate", "Every cross-role hand-off is ordered by a barrier")
def _race(ctx: Ctx):
    for f in ctx.all_facts("raw_race", "war_race", "waw_race"):
        subj = {k: v for k, v in f.subject.items() if k in ("resource", "writer", "reader")}
        yield ctx.finding(f.message, f.loc, role=f.role, kind=f.kind, **subj)


@rule("SAFE.PHASE", "program_safety", "gate", "Producers never run two phases ahead of a waiting consumer")
def _phase(ctx: Ctx):
    for f in ctx.all_facts("phase_overrun"):
        yield ctx.finding(f.message, f.loc, role=f.role, **f.subject,
                          hint="add a 'free' barrier the producer waits on before reusing the slot")


@rule("SAFE.VIEW_BOUNDS", "program_safety", "gate", "Views lie inside their pools and do not overlap")
def _view_bounds(ctx: Ctx):
    for pool in ctx.program.pools:
        vs = pool.views
        for v in vs:
            if v.offset + v.total_bytes > pool.nbytes:
                yield ctx.finding(f"view `{v.name}` [{v.offset}, {v.offset + v.total_bytes}) exceeds pool "
                                  f"`{pool.name}` of {pool.nbytes} bytes", v.loc, resource=v.name,
                                  hint=f"grow lm.smem(...) to at least {v.offset + v.total_bytes} bytes")
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                a0, a1 = a.offset, a.offset + a.total_bytes
                b0, b1 = b.offset, b.offset + b.total_bytes
                if a0 < b1 and b0 < a1:
                    yield ctx.finding(f"views `{a.name}` [{a0},{a1}) and `{b.name}` [{b0},{b1}) overlap in pool "
                                      f"`{pool.name}`", b.loc, resource=b.name,
                                      hint=f"place `{b.name}` at offset >= {a1}" if b0 >= a0 else None)


@rule("SAFE.STORE_BOUNDS", "program_safety", "gate", "Global stores stay inside their tensors")
def _store_bounds(ctx: Ctx):
    for f in ctx.all_facts("store_oob"):
        yield ctx.finding(f.message, f.loc, role=f.role, hint="pass valid=<remaining rows> to lm.store",
                          **f.subject)


# ============================================================================ optimization guidance
@rule("HINT.PIPELINE", "optimization_guidance", "hint", "Asynchronous loads should be multi-buffered")
def _hint_pipeline(ctx: Ctx):
    for op in ctx.ops():
        if isinstance(op, TmaLoad) and op.dst.view.stages == 1 and _in_loop(ctx.program, op):
            yield ctx.finding(f"`{op.dst.view.name}` is single-buffered inside a loop: each load waits for the "
                              f"previous tile to be consumed", op.loc, resource=op.dst.view.name,
                              hint="give the view stage=2..4 and index stages with the loop counter")


@rule("HINT.SPECIALIZE", "optimization_guidance", "hint", "MMA issue and the epilogue should overlap")
def _hint_specialize(ctx: Ctx):
    for r in ctx.program.roles:
        ops = list(walk(ctx.program.body[r]))
        if any(isinstance(o, Mma) for o in ops) and any(isinstance(o, (Ewise, Reduce)) for o in ops):
            yield ctx.finding(f"role `{r.name}` both issues MMAs and runs the epilogue, so tensor-core work and "
                              f"CUDA-core work serialize", r.loc, role=r.name,
                              hint="move MMA issue to its own warp and double-buffer the accumulator "
                                   "(lm.tmem(..., stage=2)) so MMA j+1 overlaps the epilogue of j")


@rule("HINT.ACC_BUFFER", "optimization_guidance", "hint", "A specialized epilogue wants a double-buffered accumulator")
def _hint_acc(ctx: Ctx):
    mma_roles = {o.role for o in ctx.ops() if isinstance(o, Mma) and isinstance(o.acc, TmemRef)}
    for tm in ctx.program.tmems:
        readers = {o.role for o in ctx.ops() if isinstance(o, TmemLoad) and o.src.view is tm}
        if tm.stages == 1 and readers and not (readers & mma_roles):
            yield ctx.finding(f"accumulator `{tm.name}` has one stage: the MMA warp idles while the epilogue drains "
                              f"it", tm.loc, resource=tm.name, hint="declare stage=2 and alternate stages")


@rule("HINT.SWIZZLE", "optimization_guidance", "hint", "MMA operands should use a swizzled layout")
def _hint_swizzle(ctx: Ctx):
    for v in ctx.program.views:
        if v.swizzle in ("none", "32B") and any(isinstance(o, Mma) and (o.a.view is v or o.b.view is v)
                                               for o in ctx.ops()):
            yield ctx.finding(f"MMA operand view `{v.name}` uses swizzle {v.swizzle}, which causes shared-memory bank "
                              f"conflicts", v.loc, resource=v.name, hint="use swizzle=\"128B\" when rows are >= 128 bytes")


@rule("HINT.SYNC_LOAD", "optimization_guidance", "hint", "Avoid synchronous global loads in steady-state loops")
def _hint_sync_load(ctx: Ctx):
    for op in ctx.ops():
        if isinstance(op, LoadG) and _in_loop(ctx.program, op):
            yield ctx.finding(f"lm.load of `{op.src.name}` inside a loop exposes global-memory latency on the "
                              f"critical path of role `{_role_of(op)}`", op.loc, resource=op.src.name,
                              hint="prefetch it into shared memory with TMA (declare the tensor LM.tma2d) and read it "
                                   "with lm.ld_smem")


@rule("HINT.UNUSED", "optimization_guidance", "hint", "Declared resources are used")
def _hint_unused(ctx: Ctx):
    used = set()
    for op in ctx.ops():
        for attr in ("dst", "a", "b", "src"):
            x = getattr(op, attr, None)
            v = getattr(x, "view", None)
            if v is not None:
                used.add(id(v))
        if isinstance(op, (Wait, Arrive, Commit, TmaLoad)):
            used.add(id(op.barrier))
        if isinstance(op, Mma) and isinstance(op.acc, TmemRef):
            used.add(id(op.acc.view))
    for v in list(ctx.program.views) + list(ctx.program.tmems) + list(ctx.program.barriers):
        if id(v) not in used:
            yield ctx.finding(f"`{v.name}` is declared but never used", v.loc, resource=v.name)


def _in_loop(p: Program, target_op) -> bool:
    def visit(ops, depth):
        for o in ops:
            if o is target_op:
                return depth > 0
            if isinstance(o, Loop):
                r = visit(o.body, depth + 1)
                if r is not None:
                    return r
        return None
    for r in p.roles:
        res = visit(p.body[r], 0)
        if res is not None:
            return res
    return False


# ============================================================================ library (latent) rules
@rule("HW.TMEM_LANES", "hardware_conformance", "gate",
      "A role reading TMEM lanes covers the matching warp quadrants (warp w accesses lanes 32*(w%4)..+31)",
      library=True)
def _tmem_lanes(ctx: Ctx):
    for op in ctx.ops():
        if isinstance(op, TmemLoad):
            lanes = op.src.shape[0]
            need = set(range(min(4, max(1, lanes // 32))))
            have = {w % 4 for w in op.role.warps}
            missing = sorted(need - have)
            if missing:
                yield ctx.finding(f"role `{_role_of(op)}` (warps {list(op.role.warps)}) reads {lanes} TMEM lanes but "
                                  f"no warp covers lane quadrant(s) {missing}", op.loc, role=_role_of(op),
                                  hint="give the role 4 consecutive warps (or a multiple of 4) so warp_id % 4 "
                                       "covers every quadrant")


@rule("SAFE.PROXY_FENCE", "program_safety", "gate",
      "Generic-proxy shared-memory writes are fenced before async-proxy (MMA/TMA) reads", library=True)
def _proxy_fence(ctx: Ctx):
    for f in ctx.all_facts("proxy_unfenced"):
        yield ctx.finding(f.message, f.loc, role=f.role, hint="call lm.fence_proxy() after the st_smem and before "
                                                             "signalling the consumer", **f.subject)


@rule("DC.TAIL_MASK", "data_consistency", "gate",
      "Reductions over tiles that include zero-filled out-of-range data are masked", library=True)
def _tail_mask(ctx: Ctx):
    for f in ctx.all_facts("tail_taint"):
        yield ctx.finding(f.message, f.loc, role=f.role, hint="pass valid=<remaining columns> to lm.reduce",
                          **f.subject)


def seed_rules() -> List[Rule]:
    return list(SEED.values())
