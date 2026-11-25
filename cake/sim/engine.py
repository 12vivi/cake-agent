"""Discrete-event simulation of one CTA on the simulated GPU.

Each warp-specialized role is a sequential instruction stream; asynchronous
engines (TMA, tensor core) execute requests in the background and signal
mbarriers when they complete.  Data effects are applied at the simulated time
they happen, so a missing wait reads stale shared memory, a mis-derived
parity hangs, a swizzle mismatch scrambles operands -- the same symptoms real
hardware shows.  The simulator also keeps sanitizer-style notes (racecheck,
lane-access and proxy-fence violations) that only the *harness* sees.
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import dtypes as D
from ..errors import CakeError, Loc, SimError
from ..expr import ev
from ..lir.core import LArg, LDesc, LKernel, LOp
from ..targets import Target
from .memory import (MMA_SWIZZLE_FACTOR, SMEM_CONFLICT_FACTOR, SWIZZLE_ALIGN, read_box, read_tile,
                     write_tile)

LOOP_LIMIT = 1_000_000
ALU_WORK = {"fill": 0.25, "copy": 1.0, "add": 1.0, "sub": 1.0, "mul": 1.0, "fma": 1.0,
            "min": 1.0, "max": 1.0, "neg": 1.0, "abs": 1.0, "convert": 1.0}


def alu_efficiency(nwarps: int) -> float:
    """Fraction of the SM's CUDA-core lanes a role with ``nwarps`` warps keeps busy."""
    nw = max(1, nwarps)
    return (min(nw, 4) / 4.0) * (nw / (nw + 1.5))


@dataclass
class Rates:
    """Per-tensor memory throughput (bytes/clk available to this CTA) and latency."""
    bytes_per_clk: Dict[str, float] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    default_bpc: float = 32.0
    default_latency: float = 700.0

    def bpc(self, tensor: str) -> float:
        return self.bytes_per_clk.get(tensor, self.default_bpc)

    def lat(self, tensor: str) -> float:
        return self.latency.get(tensor, self.default_latency)


class Mbar:
    __slots__ = ("addr", "count", "pending", "tx", "phase", "waiters")

    def __init__(self, addr: int, count: int):
        self.addr, self.count = addr, count
        self.pending, self.tx, self.phase = count, 0, 0
        self.waiters: List[Tuple["Role", int, float]] = []


class Role:
    def __init__(self, idx: int, block, env: Dict[str, Any], target: Target, cta: "Cta"):
        self.idx = idx
        self.cta = cta
        self.block = block
        self.name = block.name
        self.warps = tuple(block.warps)
        self.nwarps = len(self.warps)
        self.threads = 32 * self.nwarps
        self.quadrants = {w % 4 for w in self.warps}
        self.env = dict(env)
        self.it = iter_ops(block.body, self.env)
        self.cur: Optional[LOp] = None
        self.t = 0.0
        self.finish = 0.0
        self.done = False
        self.blocked = None
        self.regs: Dict[str, np.ndarray] = {}
        self.reg_dtype: Dict[str, D.DType] = {}
        self.last_umma_done = 0.0
        self.busy: Dict[str, float] = defaultdict(float)
        self.waited: Dict[str, float] = defaultdict(float)
        self.unfenced: List[Tuple[int, int, np.ndarray]] = []
        regs = 24
        for r in block.regs.values():
            regs += math.ceil(int(np.prod(r.shape)) / max(1, self.threads))
        self.regs_per_thread = regs
        excess = max(0, regs - target.max_regs_per_thread)
        self.spill = 1.0 + 0.35 * (excess / 32.0)


class Cta:
    """Per-CTA state (shared memory, TMEM, mbarriers, in-flight async operations, roles)."""

    def __init__(self, idx: int, block_idx: Tuple[int, int, int], env0: Dict[str, Any]):
        self.idx = idx
        self.block_idx = block_idx
        self.env0 = env0
        self.smem: Optional[np.ndarray] = None
        self.written: Optional[np.ndarray] = None
        self.tmem: Optional[np.ndarray] = None
        self.mbars: Dict[int, Mbar] = {}
        self.inflight_tma: Dict[int, Tuple[int, int]] = {}
        self.inflight_tmem: Dict[int, Tuple[int, int, float]] = {}
        self.roles: List["Role"] = []


def iter_ops(ops, env):
    for op in ops:
        if op.kind == "loop":
            start, stop, step = ev(op.start, env), ev(op.stop, env), ev(op.step, env)
            if step == 0:
                raise SimError("illegal_instruction", "loop with zero step", op.loc)
            name = op.var.name
            i, n = start, 0
            while (i < stop) if step > 0 else (i > stop):
                env[name] = i
                yield from iter_ops(op.body, env)
                i += step
                n += 1
                if n > LOOP_LIMIT:
                    raise SimError("hang", "loop exceeded the iteration limit", op.loc)
        else:
            yield op


def _loc(op) -> Loc:
    src = getattr(op, "src", None)
    if src is not None and getattr(src, "loc", None) is not None:
        return src.loc
    return getattr(op, "loc", Loc())


@dataclass
class CtaResult:
    cycles: float
    roles: Dict[str, Dict[str, Any]]
    units: Dict[str, float]
    bytes_loaded: Dict[str, int]
    flops: float
    notes: List[Dict[str, Any]]
    events: int


class CtaSim:
    def __init__(self, kernel: LKernel, target: Target, gmem: Dict[str, np.ndarray],
                 block_idx, *, compute: bool = True,
                 rates: Optional[Rates] = None, seed: int = 0, scalars: Optional[Dict[str, Any]] = None,
                 max_events: int = 4_000_000):
        self.k = kernel
        self.T = target
        self.gmem = gmem
        self.compute = compute
        self.rates = rates or Rates()
        self.max_events = max_events
        rng = np.random.default_rng(seed)
        self.rng = rng
        n = kernel.smem_bytes
        self.smem_size = n
        self._ids = itertools.count()
        self.unit_free: Dict[str, float] = defaultdict(float)
        self.unit_busy: Dict[str, float] = defaultdict(float)
        self.bytes_loaded: Dict[str, int] = defaultdict(int)
        self.flops = 0.0
        self.heap: List = []
        self.seq = itertools.count()
        self.now = 0.0
        self.events = 0
        self.notes: List[Dict[str, Any]] = []
        self._note_keys = set()
        # one CTA, or several co-resident CTAs sharing the SM's units
        idxs = [tuple(block_idx)] if isinstance(block_idx[0], int) else [tuple(b) for b in block_idx]
        gx, gy, gz = kernel.grid
        self.ctas: List[Cta] = []
        self.roles: List[Role] = []
        for ci, (bx, by, bz) in enumerate(idxs):
            env = {"blockIdx.x": bx, "blockIdx.y": by, "blockIdx.z": bz,
                   "gridDim.x": gx, "gridDim.y": gy, "gridDim.z": gz}
            if scalars:
                env.update(scalars)
            cta = Cta(ci, (bx, by, bz), env)
            if compute:
                cta.smem = rng.integers(0, 256, size=max(n, 16), dtype=np.uint8)
                cta.written = np.zeros(max(n, 16), dtype=bool)
                cta.tmem = (rng.standard_normal((128, max(kernel.tmem_cols, 1))) * 37.0).astype(np.float32)
            cta.roles = [Role(i, b, env, target, cta) for i, b in enumerate(kernel.blocks)]
            self.ctas.append(cta)
            self.roles.extend(cta.roles)
        self.params = {p.name: p for p in kernel.params}
        self._h = {
            "set": self._op_set, "mbar_init": self._op_mbar_init, "mbar_arrive": self._op_arrive,
            "mbar_expect_tx": self._op_expect_tx, "mbar_wait": self._op_wait, "fence": self._op_fence,
            "tma_load": self._op_tma_load, "umma": self._op_umma, "umma_commit": self._op_commit,
            "mma_sync": self._op_mma_sync, "tmem_load": self._op_tmem_load,
            "ld_global": self._op_ld_global, "st_global": self._op_st_global,
            "ld_smem": self._op_ld_smem, "st_smem": self._op_st_smem, "fill": self._op_fill,
            "ewise": self._op_ewise, "reduce": self._op_reduce, "convert": self._op_convert,
        }
        if compute:
            for r in self.roles:
                for reg in r.block.regs.values():
                    r.regs[reg.name] = (rng.standard_normal(reg.shape) * 1e3).astype(reg.dtype.compute) \
                        if reg.dtype.kind == "float" else rng.integers(-9999, 9999, reg.shape).astype(np.int32)
                    r.reg_dtype[reg.name] = reg.dtype
        else:
            for r in self.roles:
                for reg in r.block.regs.values():
                    r.reg_dtype[reg.name] = reg.dtype

    # ------------------------------------------------------------------ utils
    def _push(self, t: float, kind: str, payload) -> None:
        heapq.heappush(self.heap, (t, next(self.seq), kind, payload))

    def note(self, kind: str, op, role: Optional[Role], **detail) -> None:
        loc = _loc(op)
        key = (kind, loc.lineno, role.name if role else "")
        if key in self._note_keys:
            return
        self._note_keys.add(key)
        self.notes.append({"kind": kind, "loc": loc.to_dict(), "role": role.name if role else None,
                           "detail": detail})

    def _err(self, fault: str, msg: str, op, role: Optional[Role] = None, **detail):
        d = dict(detail)
        if role is not None:
            d["role"] = role.name
        return SimError(fault, msg, loc=_loc(op), detail=d)

    def _mbar(self, addr: int, op, role) -> Mbar:
        m = role.cta.mbars.get(addr)
        if m is None:
            if addr % 8 or addr < 0 or addr + 8 > self.smem_size:
                raise self._err("illegal_address", f"mbarrier address {addr:#x} is invalid", op, role,
                                addr=addr)
            raise self._err("illegal_instruction",
                            f"mbarrier at smem {addr:#x} used before mbarrier.init", op, role, addr=addr)
        return m

    def _check_smem_range(self, lo: int, hi: int, op, role, what: str) -> None:
        if lo < 0 or hi > self.smem_size:
            raise self._err("illegal_address",
                            f"{what} smem range [{lo:#x}, {hi:#x}) exceeds the {self.smem_size}-byte "
                            f"shared-memory allocation", op, role, lo=lo, hi=hi)

    def _check_mbar_overlap(self, lo: int, hi: int, op, role, what: str) -> None:
        for a in role.cta.mbars:
            if lo < a + 8 and a < hi:
                raise self._err("illegal_address",
                                f"{what} [{lo:#x}, {hi:#x}) overwrites the mbarrier object at {a:#x}",
                                op, role, mbar=a)

    # ------------------------------------------------------------------ run
    def run(self) -> CtaResult:
        for cta in self.ctas:
            t = 0.0
            for op in self.k.prologue:
                if op.kind == "mbar_init":
                    self._op_mbar_init(None, op, cta)
                    t += 2
                elif op.kind == "set":
                    cta.env0[op.name] = ev(op.value, cta.env0)
                    for r in cta.roles:
                        r.env[op.name] = cta.env0[op.name]
                else:
                    raise self._err("illegal_instruction", f"`{op.kind}` is not allowed in the prologue", op)
            t += 20  # __syncthreads after barrier initialization
            for r in cta.roles:
                r.t = t
                self._push(t, "step", r)
        while self.heap:
            time, _, kind, payload = heapq.heappop(self.heap)
            self.now = time
            self.events += 1
            if self.events > self.max_events:
                raise SimError("hang", "watchdog: event budget exhausted (kernel does not terminate)",
                               detail={"events": self.events})
            if kind == "step":
                self._step(payload)
            elif kind == "tma_done":
                self._tma_done(payload)
            elif kind == "umma_start":
                self._umma_start(payload)
            elif kind == "umma_done":
                self._umma_done(payload)
            elif kind == "arrive":
                addr, op, role = payload
                m = role.cta.mbars.get(addr)
                if m is None:
                    raise self._err("illegal_instruction", "commit to an uninitialized mbarrier", op, role)
                self._arrive(m)
        stuck = [r for r in self.roles if not r.done]
        if stuck:
            blocked = []
            for r in stuck:
                if r.blocked is not None:
                    m, parity, since = r.blocked
                    blocked.append({"role": r.name, "mbar": m.addr, "parity": parity,
                                    "phase": m.phase, "pending": m.pending, "tx": m.tx,
                                    "loc": _loc(r.cur).to_dict(),
                                    "mbar_name": self.k.meta.get("mbar_names", {}).get(m.addr)})
            raise SimError("hang", "deadlock: every remaining role is blocked on an mbarrier wait",
                           loc=_loc(stuck[0].cur) if stuck[0].cur is not None else None,
                           detail={"blocked": blocked})
        cycles = max([r.finish for r in self.roles] + [0.0])
        roles = {r.name: {"busy": dict(r.busy), "waited": dict(r.waited), "finish": r.finish,
                          "regs_per_thread": r.regs_per_thread, "warps": list(r.warps)}
                 for r in self.ctas[0].roles}
        return CtaResult(cycles=cycles, roles=roles, units=dict(self.unit_busy),
                         bytes_loaded=dict(self.bytes_loaded), flops=self.flops, notes=self.notes,
                         events=self.events)

    def _step(self, r: Role) -> None:
        if r.done:
            return
        while True:
            if r.cur is None:
                r.cur = next(r.it, None)
                if r.cur is None:
                    r.done = True
                    r.finish = self.now
                    return
            op = r.cur
            try:
                res = self._h[op.kind](r, op)
            except SimError:
                raise
            except (CakeError, KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as e:
                msg = e.message if isinstance(e, CakeError) else f"{type(e).__name__}: {e}"
                raise self._err("illegal_instruction", f"invalid instruction: {msg}", op, r) from None
            if res is None:
                return
            r.cur = None
            if res > 0:
                r.t = self.now + res
                self._push(r.t, "step", r)
                return

    def _retry_if_busy(self, r: Role, unit: str) -> bool:
        free = self.unit_free[unit]
        if free > self.now + 1e-9:
            self._push(free, "step", r)
            return True
        return False

    # ------------------------------------------------------------------ sync
    def _op_set(self, r, op):
        r.env[op.name] = ev(op.value, r.env)
        return 0.0

    def _op_mbar_init(self, r, op, cta: Optional[Cta] = None):
        cta = cta if cta is not None else r.cta
        env = r.env if r is not None else cta.env0
        addr = int(ev(op.addr, env))
        if addr % 8:
            raise self._err("misaligned", f"mbarrier address {addr:#x} is not 8-byte aligned", op, r)
        self._check_smem_range(addr, addr + 8, op, r, "mbarrier")
        count = int(ev(op.count, env))
        if count < 1:
            raise self._err("illegal_instruction", "mbarrier.init with count < 1", op, r)
        cta.mbars[addr] = Mbar(addr, count)
        return 2.0

    def _arrive(self, m: Mbar, n: int = 1) -> None:
        m.pending -= n
        self._check_phase(m)

    def _check_phase(self, m: Mbar) -> None:
        guard = 0
        while m.pending <= 0 and m.tx <= 0:
            m.phase += 1
            m.pending += m.count
            m.tx = min(m.tx, 0)
            guard += 1
            keep = []
            for (r, parity, since) in m.waiters:
                if (m.phase & 1) != parity:
                    r.blocked = None
                    name = self.k.meta.get("mbar_names", {}).get(m.addr, f"{m.addr:#x}")
                    r.waited[name] += self.now - since
                    r.cur = None
                    r.t = self.now + self.T.wake_cycles
                    self._push(r.t, "step", r)
                else:
                    keep.append((r, parity, since))
            m.waiters = keep
            if guard > 4:
                break

    def _op_arrive(self, r, op):
        m = self._mbar(int(ev(op.addr, r.env)), op, r)
        self._arrive(m)
        return 2.0

    def _op_expect_tx(self, r, op):
        m = self._mbar(int(ev(op.addr, r.env)), op, r)
        m.tx += int(ev(op.nbytes, r.env))
        self._arrive(m)
        return 2.0

    def _op_wait(self, r, op):
        m = self._mbar(int(ev(op.addr, r.env)), op, r)
        parity = int(ev(op.parity, r.env)) & 1
        if (m.phase & 1) != parity:
            r.busy["sync"] += self.T.mbar_cycles
            return float(self.T.mbar_cycles)
        r.blocked = (m, parity, self.now)
        m.waiters.append((r, parity, self.now))
        return None

    def _op_fence(self, r, op):
        r.unfenced.clear()
        return 8.0

    # ------------------------------------------------------------------ TMA
    def _op_tma_load(self, r, op):
        T = self.T
        if op.engine == "tma" and not T.tma:
            raise self._err("unsupported", f"TMA (cp.async.bulk.tensor) is not available on {T.name}", op, r)
        if op.engine == "cp.async" and not T.cp_async:
            raise self._err("unsupported", f"cp.async is not available on {T.name}", op, r)
        p = self.params.get(op.tensor)
        if p is None:
            raise self._err("illegal_instruction", f"unknown tensor `{op.tensor}`", op, r)
        if op.engine == "tma" and p.kind != "tma":
            raise self._err("illegal_instruction",
                            f"tensor `{op.tensor}` has no TMA descriptor (declared as {p.kind})", op, r)
        dst = int(ev(op.dst, r.env))
        rows, cols = op.box
        nbytes = rows * cols * op.dtype.nbytes
        if rows > 256 or cols > 256 or (cols * op.dtype.nbytes) % 16:
            raise self._err("illegal_instruction",
                            f"invalid TMA box {op.box}: each dim must be <= 256 and the inner "
                            f"dimension a multiple of 16 bytes", op, r)
        if dst % SWIZZLE_ALIGN.get(op.swizzle, 128):
            raise self._err("misaligned", f"TMA destination {dst:#x} is not aligned to "
                            f"{SWIZZLE_ALIGN.get(op.swizzle, 128)} bytes (swizzle {op.swizzle})", op, r)
        self._check_smem_range(dst, dst + nbytes, op, r, "TMA destination")
        self._check_mbar_overlap(dst, dst + nbytes, op, r, "TMA destination")
        mb = int(ev(op.mbar, r.env))
        self._mbar(mb, op, r)
        coords = tuple(int(ev(c, r.env)) for c in op.coords)
        if len(coords) != len(p.shape):
            raise self._err("illegal_instruction",
                            f"TMA coordinate rank {len(coords)} != tensor rank {len(p.shape)}", op, r)
        if op.dtype.name != p.dtype.name:
            raise self._err("illegal_instruction",
                            f"TMA element type {op.dtype.name} != tensor `{op.tensor}` type {p.dtype.name}",
                            op, r)
        rates = self.rates
        if op.engine == "tma":
            issue = float(T.issue_cycles * 4)
            unit = "tma"
            occ = T.tma_overhead / 4.0 + nbytes / rates.bpc(op.tensor)
            lat = rates.lat(op.tensor) + T.tma_overhead
        else:
            issue = 4.0 + nbytes / (16.0 * r.threads)
            unit = "lsu"
            occ = nbytes / rates.bpc(op.tensor)
            lat = rates.lat(op.tensor)
        start = max(self.now + issue, self.unit_free[unit])
        self.unit_free[unit] = start + occ
        self.unit_busy[unit] += occ
        done = start + occ + lat
        tid = next(self._ids)
        r.cta.inflight_tma[tid] = (dst, dst + nbytes)
        self.bytes_loaded[op.tensor] += nbytes
        self._push(done, "tma_done", (tid, op, r, dst, coords, nbytes, mb))
        r.busy["tma_issue"] += issue
        return issue

    def _tma_done(self, payload) -> None:
        tid, op, r, dst, coords, nbytes, mb = payload
        cta = r.cta
        cta.inflight_tma.pop(tid, None)
        if self.compute:
            arr = self.gmem[op.tensor]
            box = (1,) * (arr.ndim - 2) + tuple(op.box)
            data, oob = read_box(arr, coords, box)
            write_tile(cta.smem, dst, data.reshape(op.box), op.swizzle)
            cta.written[dst:dst + nbytes] = True
            if oob:
                self.note("TMA_ZERO_FILL", op, r, tensor=op.tensor, coords=list(coords), box=list(op.box),
                          shape=list(arr.shape))
        m = cta.mbars.get(mb)
        if m is not None:
            m.tx -= nbytes
            self._check_phase(m)

    # ------------------------------------------------------------------ MMA
    def _check_desc(self, d: LDesc, r, op, what: str) -> int:
        addr = int(ev(d.addr, r.env))
        if addr % SWIZZLE_ALIGN.get(d.swizzle, 128):
            raise self._err("misaligned", f"MMA operand {what} at {addr:#x} not aligned for swizzle "
                            f"{d.swizzle}", op, r)
        self._check_smem_range(addr, addr + d.nbytes, op, r, f"MMA operand {what}")
        return addr

    def _read_operand(self, d: LDesc, addr: int, r, op) -> np.ndarray:
        lo, hi = addr, addr + d.nbytes
        cta = r.cta
        for (a, b) in cta.inflight_tma.values():
            if lo < b and a < hi:
                self.note("STALE_SMEM_READ", op, r, range=[lo, hi], reason="read overlaps an in-flight TMA write")
                break
        if not cta.written[lo:hi].all():
            self.note("UNINIT_SMEM_READ", op, r, range=[lo, hi])
        smem = cta.smem
        stale = []
        for role in cta.roles:
            for (a, b, old) in role.unfenced:
                if lo < b and a < hi:
                    stale.append((a, b, old))
                    self.note("PROXY_FENCE", op, r, writer=role.name, range=[a, b],
                              reason="async-proxy read of generic-proxy smem writes without fence.proxy.async")
        if stale:
            smem = cta.smem.copy()
            for (a, b, old) in stale:
                smem[a:b] = old
        tile = read_tile(smem, addr, (d.rows, d.cols), d.dtype.storage, d.swizzle)
        return D.decode(tile, d.dtype)

    def _mma_shape_ok(self, op) -> Optional[str]:
        a, b = op.a, op.b
        if (a.rows, a.cols) != (op.m, op.k):
            return f"A descriptor {a.rows}x{a.cols} does not match M={op.m}, K={op.k}"
        want = (op.n, op.k) if op.b_major == "K" else (op.k, op.n)
        if (b.rows, b.cols) != want:
            return f"B descriptor {b.rows}x{b.cols} does not match N={op.n}, K={op.k} ({op.b_major}-major)"
        if a.dtype.name != b.dtype.name:
            return "A and B operand types differ"
        return None

    def _op_umma(self, r, op):
        T = self.T
        if not T.tcgen05:
            raise self._err("unsupported", f"tcgen05.mma is not available on {T.name}", op, r)
        bad = self._mma_shape_ok(op)
        if bad is None and (op.m not in (64, 128) or op.n % 16 or not 16 <= op.n <= 256 or op.k % 16):
            bad = f"unsupported tcgen05.mma shape M={op.m} N={op.n} K={op.k}"
        if bad:
            raise self._err("illegal_instruction", f"invalid MMA instruction descriptor: {bad}", op, r)
        a_addr = self._check_desc(op.a, r, op, "A")
        b_addr = self._check_desc(op.b, r, op, "B")
        col = int(ev(op.d_col, r.env))
        if col < 0 or col + op.n > self.k.tmem_cols:
            raise self._err("illegal_address", f"TMEM columns [{col}, {col + op.n}) outside the "
                            f"{self.k.tmem_cols}-column allocation", op, r)
        acc = bool(ev(op.accumulate, r.env))
        issue = float(T.issue_cycles)
        factor = max(MMA_SWIZZLE_FACTOR[op.a.swizzle], MMA_SWIZZLE_FACTOR[op.b.swizzle])
        dur = (2.0 * 128 * op.n * op.k / T.tc_rate(op.a.dtype.name)) * factor + (op.k / 16) * 2
        start = max(self.now + issue, self.unit_free["tc"])
        self.unit_free["tc"] = start + dur
        self.unit_busy["tc"] += dur
        done = start + dur + T.umma_latency
        r.last_umma_done = max(r.last_umma_done, done)
        self.flops += 2.0 * op.m * op.n * op.k
        tid = next(self._ids)
        r.cta.inflight_tmem[tid] = (col, col + op.n, done)
        state = {"id": tid, "op": op, "r": r, "a": a_addr, "b": b_addr, "col": col, "acc": acc, "prod": None}
        self._push(start, "umma_start", state)
        self._push(done, "umma_done", state)
        r.busy["mma_issue"] += issue
        return issue

    def _umma_start(self, s) -> None:
        if not self.compute:
            return
        op, r = s["op"], s["r"]
        A = self._read_operand(op.a, s["a"], r, op)
        B = self._read_operand(op.b, s["b"], r, op)
        with np.errstate(all="ignore"):  # garbage operands (races) may overflow
            s["prod"] = A @ (B.T if op.b_major == "K" else B)

    def _umma_done(self, s) -> None:
        cta = s["r"].cta
        cta.inflight_tmem.pop(s["id"], None)
        if not self.compute:
            return
        op = s["op"]
        c0 = s["col"]
        if s["acc"]:
            with np.errstate(all="ignore"):
                cta.tmem[:op.m, c0:c0 + op.n] += s["prod"]
        else:
            cta.tmem[:op.m, c0:c0 + op.n] = s["prod"]

    def _op_commit(self, r, op):
        if not self.T.tcgen05:
            raise self._err("unsupported", f"tcgen05.commit is not available on {self.T.name}", op, r)
        addr = int(ev(op.mbar, r.env))
        self._mbar(addr, op, r)
        t = max(self.now, r.last_umma_done) + self.T.commit_cycles
        self._push(t, "arrive", (addr, op, r))
        return float(self.T.issue_cycles)

    def _op_mma_sync(self, r, op):
        T = self.T
        bad = self._mma_shape_ok(op)
        if bad is None and (op.m % 16 or op.n % 8 or op.k % 16):
            bad = f"unsupported mma shape M={op.m} N={op.n} K={op.k}"
        if bad:
            raise self._err("illegal_instruction", f"invalid MMA: {bad}", op, r)
        if self._retry_if_busy(r, "tc"):
            return None
        a_addr = self._check_desc(op.a, r, op, "A")
        b_addr = self._check_desc(op.b, r, op, "B")
        legacy = 4.0 if T.tcgen05 else (1.25 if T.wgmma else 1.0)
        factor = max(MMA_SWIZZLE_FACTOR[op.a.swizzle], MMA_SWIZZLE_FACTOR[op.b.swizzle])
        dur = (2.0 * op.m * op.n * op.k / T.tc_rate(op.a.dtype.name)) * factor * legacy + 8
        self.unit_free["tc"] = self.now + dur
        self.unit_busy["tc"] += dur
        self.flops += 2.0 * op.m * op.n * op.k
        if self.compute:
            A = self._read_operand(op.a, a_addr, r, op)
            B = self._read_operand(op.b, b_addr, r, op)
            with np.errstate(all="ignore"):
                prod = A @ (B.T if op.b_major == "K" else B)
            acc = r.regs[op.acc]
            if bool(ev(op.accumulate, r.env)):
                acc += prod
            else:
                acc[...] = prod
        r.busy["mma"] += dur
        return dur * r.spill

    # ------------------------------------------------------------------ TMEM / global / smem
    def _op_tmem_load(self, r, op):
        T = self.T
        if not T.tmem:
            raise self._err("unsupported", f"tcgen05.ld (TMEM) is not available on {T.name}", op, r)
        col = int(ev(op.col, r.env))
        if col < 0 or col + op.ncols > self.k.tmem_cols:
            raise self._err("illegal_address", f"TMEM columns [{col}, {col + op.ncols}) outside the "
                            f"{self.k.tmem_cols}-column allocation", op, r)
        if self._retry_if_busy(r, "tmem"):
            return None
        nbytes = op.lanes * op.ncols * 4
        occ = nbytes / T.tmem_bytes_per_clk
        self.unit_free["tmem"] = self.now + occ
        self.unit_busy["tmem"] += occ
        dur = (occ + T.tmem_ld_latency) * r.spill
        missing = [q for q in range(min(4, max(1, op.lanes // 32))) if q not in r.quadrants]
        if missing:
            self.note("TMEM_LANE_QUADRANT", op, r, warps=list(r.warps), missing_quadrants=missing,
                      reason="a warp can only access TMEM lanes 32*(warp_id % 4) .. +31")
        for (lo, hi, done) in r.cta.inflight_tmem.values():
            if col < hi and lo < col + op.ncols and done > self.now:
                self.note("TMEM_STALE_READ", op, r, cols=[col, col + op.ncols],
                          reason="tcgen05.ld overlaps an MMA that has not completed")
                break
        if self.compute:
            vals = r.cta.tmem[:op.lanes, col:col + op.ncols].copy()
            for q in missing:
                vals[q * 32:(q + 1) * 32] = self.rng.standard_normal((32, op.ncols)) * 1e3
            dst = r.regs[op.dst]
            if dst.shape != vals.shape:
                raise self._err("illegal_instruction",
                                f"tmem_load destination shape {dst.shape} != {vals.shape}", op, r)
            dst[...] = vals
        r.busy["tmem_ld"] += dur
        return dur

    def _gbox(self, r, op, tensor: str, coords, shape) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        p = self.params.get(tensor)
        if p is None:
            raise self._err("illegal_instruction", f"unknown tensor `{tensor}`", op, r)
        c = tuple(int(ev(x, r.env)) for x in coords)
        if len(c) != len(p.shape):
            raise self._err("illegal_instruction",
                            f"coordinate rank {len(c)} != rank {len(p.shape)} of `{tensor}`", op, r)
        box = (1,) * (len(p.shape) - len(shape)) + tuple(shape)
        return c, box

    def _op_ld_global(self, r, op):
        dt = r.reg_dtype[op.dst]
        shape = self.k.blocks[r.idx].regs[op.dst].shape
        coords, box = self._gbox(r, op, op.tensor, op.coords, shape)
        if self._retry_if_busy(r, "lsu"):
            return None
        p = self.params[op.tensor]
        nbytes = int(np.prod(shape)) * p.dtype.nbytes
        occ = nbytes / self.rates.bpc(op.tensor)
        self.unit_free["lsu"] = self.now + occ
        self.unit_busy["lsu"] += occ
        self.bytes_loaded[op.tensor] += nbytes
        dur = self.rates.lat(op.tensor) + occ
        if self.compute:
            data, _ = read_box(self.gmem[op.tensor], coords, box)
            r.regs[op.dst][...] = D.decode(data, p.dtype).reshape(shape).astype(dt.compute)
        r.busy["gmem_ld"] += dur
        return dur

    def _op_st_global(self, r, op):
        p = self.params.get(op.tensor)
        shape = self.k.blocks[r.idx].regs[op.src_reg].shape
        coords, box = self._gbox(r, op, op.tensor, op.coords, shape)
        n0 = shape[0]
        if op.valid is not None:
            n0 = max(0, min(n0, int(ev(op.valid, r.env))))
        store_box = list(box)
        store_box[len(box) - len(shape)] = n0
        for c, b, n in zip(coords, store_box, p.shape):
            if b > 0 and (c < 0 or c + b > n):
                raise self._err("illegal_address",
                                f"store to `{op.tensor}` at {list(coords)} with extent {store_box} is out of "
                                f"bounds for shape {list(p.shape)}", op, r, coords=list(coords),
                                shape=list(p.shape))
        nbytes = int(np.prod(shape)) * p.dtype.nbytes
        start = max(self.now, self.unit_free["lsu"])
        occ = nbytes / self.rates.bpc(op.tensor)
        self.unit_free["lsu"] = start + occ
        self.unit_busy["lsu"] += occ
        issue = 4.0 + int(np.prod(shape)) / max(1, r.threads)
        if self.compute and n0 > 0:
            src = r.regs[op.src_reg]
            if r.reg_dtype[op.src_reg].kind != p.dtype.kind:
                raise self._err("illegal_instruction", f"store of {r.reg_dtype[op.src_reg].name} registers "
                                f"to {p.dtype.name} tensor `{op.tensor}`", op, r)
            data = D.encode(src[:n0], p.dtype).reshape(tuple(store_box))
            sl = tuple(slice(c, c + b) for c, b in zip(coords, store_box))
            self.gmem[op.tensor][sl] = data
        r.busy["gmem_st"] += issue
        return issue

    def _op_ld_smem(self, r, op):
        shape = self.k.blocks[r.idx].regs[op.dst].shape
        addr = int(ev(op.addr, r.env))
        nbytes = int(np.prod(shape)) * op.dtype.nbytes
        if addr % 16:
            raise self._err("misaligned", f"shared load at {addr:#x} is not 16-byte aligned", op, r)
        self._check_smem_range(addr, addr + nbytes, op, r, "shared load")
        if self._retry_if_busy(r, "smem"):
            return None
        occ = nbytes / self.T.smem_bytes_per_clk * SMEM_CONFLICT_FACTOR[op.swizzle]
        self.unit_free["smem"] = self.now + occ
        self.unit_busy["smem"] += occ
        if self.compute:
            for (a, b) in r.cta.inflight_tma.values():
                if addr < b and a < addr + nbytes:
                    self.note("STALE_SMEM_READ", op, r, range=[addr, addr + nbytes],
                              reason="read overlaps an in-flight TMA write")
                    break
            tile = read_tile(r.cta.smem, addr, shape if len(shape) == 2 else (1, shape[0]),
                             op.dtype.storage, op.swizzle)
            r.regs[op.dst][...] = D.decode(tile, op.dtype).reshape(shape)
        dur = (occ + 30) * r.spill
        r.busy["smem"] += dur
        return dur

    def _op_st_smem(self, r, op):
        shape = self.k.blocks[r.idx].regs[op.src_reg].shape
        addr = int(ev(op.addr, r.env))
        nbytes = int(np.prod(shape)) * op.dtype.nbytes
        if addr % 16:
            raise self._err("misaligned", f"shared store at {addr:#x} is not 16-byte aligned", op, r)
        self._check_smem_range(addr, addr + nbytes, op, r, "shared store")
        self._check_mbar_overlap(addr, addr + nbytes, op, r, "shared store")
        if self._retry_if_busy(r, "smem"):
            return None
        occ = nbytes / self.T.smem_bytes_per_clk * SMEM_CONFLICT_FACTOR[op.swizzle]
        self.unit_free["smem"] = self.now + occ
        self.unit_busy["smem"] += occ
        if self.compute:
            old = r.cta.smem[addr:addr + nbytes].copy()
            tile = D.encode(r.regs[op.src_reg], op.dtype).reshape(shape if len(shape) == 2 else (1, shape[0]))
            write_tile(r.cta.smem, addr, tile, op.swizzle)
            r.cta.written[addr:addr + nbytes] = True
            r.unfenced.append((addr, addr + nbytes, old))
        dur = occ * r.spill
        r.busy["smem"] += dur
        return dur

    # ------------------------------------------------------------------ register ops (CUDA cores)
    def _alu(self, r, work: float):
        # The SM's CUDA cores are a shared pipe of ``alu_lanes`` lanes/clk.  A role
        # reserves its share (work / lanes) but, limited by its own warps' latency
        # hiding, takes work / (lanes * efficiency) -- so several roles (or
        # co-resident CTAs) can overlap until the pipe saturates.
        if self._retry_if_busy(r, "alu"):
            return None
        occ = work / self.T.alu_lanes * r.spill
        dur = work / (self.T.alu_lanes * alu_efficiency(r.nwarps)) * r.spill + 2.0
        self.unit_free["alu"] = self.now + occ
        self.unit_busy["alu"] += occ
        r.busy["alu"] += dur
        return dur

    def _arg(self, r, a: LArg):
        if a.kind == "reg":
            return r.regs[a.value]
        if a.kind == "row":
            return r.regs[a.value][None, :]
        if a.kind == "col":
            return r.regs[a.value][:, None]
        return ev(a.value, r.env)

    def _op_fill(self, r, op):
        n = int(np.prod(self.k.blocks[r.idx].regs[op.dst].shape))
        dur = self._alu(r, n * ALU_WORK["fill"])
        if dur is None:
            return None
        if self.compute:
            r.regs[op.dst][...] = ev(op.value, r.env)
        return dur

    def _op_ewise(self, r, op):
        shape = self.k.blocks[r.idx].regs[op.dst].shape
        n = int(np.prod(shape))
        dur = self._alu(r, n * ALU_WORK.get(op.op, 1.0))
        if dur is None:
            return None
        if self.compute:
            a = [self._arg(r, x) for x in op.args]
            f = op.op
            with np.errstate(all="ignore"):
                if f == "copy":
                    v = a[0]
                elif f == "add":
                    v = a[0] + a[1]
                elif f == "sub":
                    v = a[0] - a[1]
                elif f == "mul":
                    v = a[0] * a[1]
                elif f == "fma":
                    v = a[0] * a[1] + a[2]
                elif f == "min":
                    v = np.minimum(a[0], a[1])
                elif f == "max":
                    v = np.maximum(a[0], a[1])
                elif f == "neg":
                    v = -a[0]
                elif f == "abs":
                    v = np.abs(a[0])
                else:
                    raise self._err("illegal_instruction", f"unknown elementwise op `{f}`", op, r)
                dst = r.regs[op.dst]
                try:
                    dst[...] = np.broadcast_to(np.asarray(v), shape).astype(dst.dtype)
                except ValueError:
                    raise self._err("illegal_instruction", f"elementwise operands do not broadcast to "
                                    f"{shape}", op, r) from None
        return dur

    def _op_reduce(self, r, op):
        src_shape = self.k.blocks[r.idx].regs[op.src_reg].shape
        n = int(np.prod(src_shape))
        per = 2.0 if op.op in ("argmin", "argmax") else 1.0
        dur = self._alu(r, n * per + src_shape[0] * 3)
        if dur is None:
            return None
        if self.compute:
            s = r.regs[op.src_reg]
            if s.ndim != 2:
                raise self._err("illegal_instruction", "reduce expects a 2-D source tile", op, r)
            if op.valid is not None:
                v = int(ev(op.valid, r.env))
                if v < s.shape[1]:
                    fillv = {"min": np.inf, "argmin": np.inf, "max": -np.inf, "argmax": -np.inf,
                             "sum": 0.0}[op.op]
                    s = s.copy()
                    s[:, max(v, 0):] = fillv
            dst = r.regs[op.dst]
            off = int(ev(op.index_offset, r.env))
            with np.errstate(invalid="ignore"):
                if op.op == "sum":
                    dst += s.sum(axis=1)
                elif op.op == "min":
                    np.minimum(dst, s.min(axis=1), out=dst)
                elif op.op == "max":
                    np.maximum(dst, s.max(axis=1), out=dst)
                else:
                    if op.op == "argmin":
                        vals, arg = s.min(axis=1), s.argmin(axis=1)
                        better = vals < dst
                    else:
                        vals, arg = s.max(axis=1), s.argmax(axis=1)
                        better = vals > dst
                    dst[better] = vals[better]
                    idx = r.regs[op.idx]
                    idx[better] = (arg[better] + off).astype(np.int32)
        return dur

    def _op_convert(self, r, op):
        shape = self.k.blocks[r.idx].regs[op.dst].shape
        dur = self._alu(r, int(np.prod(shape)) * ALU_WORK["convert"])
        if dur is None:
            return None
        if self.compute:
            dst = r.regs[op.dst]
            dt = r.reg_dtype[op.dst]
            src = r.regs[op.src_reg]
            if dt.kind == "float" and dt.name != "f32":
                dst[...] = D.quantize(src, dt)
            else:
                dst[...] = src.astype(dst.dtype)
        return dur
