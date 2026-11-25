"""Abstract execution of a Cake IR schedule.

The analyzer interprets every role of one CTA with exact mbarrier phase
semantics and tracks happens-before with vector clocks (asynchronous TMA and
MMA operations get their own clock components).  Because ordering is decided
by vector clocks rather than by a particular timing, a missing or misplaced
wait is found regardless of how fast the hardware happens to be.

It produces *facts* (deadlocks, races, uninitialized reads, phase overruns,
out-of-range slots, zero-fill taint, proxy-fence violations, ...) that the
verifier rules turn into findings.  This mirrors the paper's split between an
analysis substrate (P5 "analysis-friendly") and the rule set that evolves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..errors import CakeError, Loc
from ..expr import ev
from ..ir.core import (Arrive, Barrier, Commit, Convert, Ewise, FenceProxy, Fill, LdSmem, LoadG, Loop, Mma, Op,
                       Program, Reduce, RegTile, SmemRef, StoreG, StSmem, TmaLoad, TmemLoad, TmemRef, Wait,
                       tma_groups)

VC = Dict[str, int]


def _join(a: VC, b: VC) -> None:
    for k, v in b.items():
        if a.get(k, 0) < v:
            a[k] = v


def _leq(a: VC, b: VC) -> bool:
    for k, v in a.items():
        if b.get(k, 0) < v:
            return False
    return True


@dataclass
class Fact:
    kind: str
    message: str
    loc: Optional[Loc]
    role: Optional[str]
    subject: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Facts:
    block_idx: Tuple[int, int, int]
    facts: List[Fact] = field(default_factory=list)
    ops_executed: int = 0
    truncated: bool = False
    tma_partial_oob: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def of(self, *kinds: str) -> List[Fact]:
        return [f for f in self.facts if f.kind in kinds]


class _Slot:
    __slots__ = ("completed", "pending", "tx", "phase_vc", "history", "init", "arrivals")

    def __init__(self, init: int):
        self.completed = 0
        self.pending = init
        self.tx = 0
        self.phase_vc: VC = {}
        self.history: List[VC] = []
        self.init = init
        self.arrivals = 0


class _Mem:
    __slots__ = ("write_vc", "write_loc", "writer", "reads", "zero_fill", "unfenced_by", "consumed")

    def __init__(self):
        self.write_vc: Optional[VC] = None
        self.write_loc: Optional[Loc] = None
        self.writer: Optional[str] = None
        self.reads: List[Tuple[VC, Loc, str]] = []
        self.zero_fill = False
        self.unfenced_by: Optional[str] = None
        self.consumed = False


class _Role:
    def __init__(self, role, ops, env):
        self.role = role
        self.name = role.name
        self.it = _gen(ops, env)
        self.env = env
        self.cur: Optional[Op] = None
        self.done = False
        self.blocked: Optional[Tuple] = None
        self.vc: VC = {f"R:{role.name}": 1}
        self.waits: Dict[Tuple[int, int], int] = {}
        self.mma_join: VC = {}
        self.last_mma: VC = {}
        self.regs_written: Set[str] = set()
        self.reg_taint: Dict[str, str] = {}
        self.unfenced: Set[Tuple] = set()

    def tick(self) -> None:
        k = f"R:{self.name}"
        self.vc[k] = self.vc.get(k, 0) + 1


def _gen(ops: List[Op], env: Dict[str, Any]) -> Iterator[Op]:
    for op in ops:
        if isinstance(op, Loop):
            start, stop, step = ev(op.start, env), ev(op.stop, env), ev(op.step, env)
            if step == 0:
                raise CakeError("loop with zero step", loc=op.loc)
            i = start
            while (i < stop) if step > 0 else (i > stop):
                env[op.var.name] = i
                yield from _gen(op.body, env)
                i += step
        else:
            yield op


class Analyzer:
    def __init__(self, prog: Program, block_idx: Tuple[int, int, int] = (0, 0, 0), max_ops: int = 400_000):
        self.p = prog
        self.block_idx = block_idx
        self.max_ops = max_ops
        self.facts = Facts(tuple(block_idx))
        self._seen: Set[Tuple] = set()
        self.groups = {}
        for r in prog.roles:
            self.groups.update(tma_groups(prog.body[r]))
        gx, gy, gz = prog.grid
        bx, by, bz = block_idx
        base_env = {"blockIdx.x": bx, "blockIdx.y": by, "blockIdx.z": bz,
                    "gridDim.x": gx, "gridDim.y": gy, "gridDim.z": gz}
        base_env.update(prog.meta.get("scalars", {}))
        self.roles = [_Role(r, prog.body[r], dict(base_env)) for r in prog.roles]
        self.by_name = {r.name: r for r in self.roles}
        self.slots: Dict[Tuple[int, int], _Slot] = {}
        for b in prog.barriers:
            for s in range(b.count):
                self.slots[(id(b), s)] = _Slot(b.init_count)
        self.mem: Dict[Tuple, _Mem] = {}
        self._nasync = 0
        self._ops = 0

    # ------------------------------------------------------------------ reporting
    def fact(self, kind: str, message: str, op: Optional[Op], role: Optional[_Role], **subject) -> None:
        loc = op.loc if op is not None else None
        key = (kind, loc.lineno if loc else 0, role.name if role else None, tuple(sorted(
            (k, str(v)) for k, v in subject.items() if k in ("barrier", "resource", "register", "slot"))))
        if key in self._seen:
            return
        self._seen.add(key)
        self.facts.facts.append(Fact(kind, message, loc, role.name if role else None, subject))

    # ------------------------------------------------------------------ helpers
    def _eval(self, x, r: _Role, op: Op):
        try:
            return ev(x, r.env)
        except CakeError as e:
            self.fact("eval_error", f"cannot evaluate `{x}`: {e.message}", op, r)
            raise _Abort()
        except ZeroDivisionError:
            self.fact("eval_error", f"division by zero evaluating `{x}`", op, r)
            raise _Abort()

    def _slot(self, b: Barrier, s_expr, r: _Role, op: Op) -> Tuple[Tuple[int, int], int]:
        s = int(self._eval(s_expr, r, op))
        if not 0 <= s < b.count:
            self.fact("slot_range", f"slot index {s} is out of range for barrier `{b.name}` with {b.count} "
                      f"slot(s)", op, r, barrier=b.name, slot=s)
            s %= b.count
        return (id(b), s), s

    def _skey(self, ref: SmemRef, r: _Role, op: Op) -> Tuple:
        st = int(self._eval(ref.stage, r, op))
        if not 0 <= st < ref.view.stages:
            self.fact("stage_range", f"stage index {st} is out of range for view `{ref.view.name}` with "
                      f"{ref.view.stages} stage(s)", op, r, resource=ref.view.name, stage=st)
            st %= ref.view.stages
        return ("S", ref.view.name, st)

    def _tkey(self, ref: TmemRef, r: _Role, op: Op) -> Tuple:
        st = int(self._eval(ref.stage, r, op))
        if not 0 <= st < ref.view.stages:
            self.fact("stage_range", f"stage index {st} is out of range for TMEM region `{ref.view.name}` with "
                      f"{ref.view.stages} stage(s)", op, r, resource=ref.view.name, stage=st)
            st %= ref.view.stages
        return ("T", ref.view.name, st)

    def _m(self, key: Tuple) -> _Mem:
        m = self.mem.get(key)
        if m is None:
            m = self.mem[key] = _Mem()
        return m

    @staticmethod
    def _kname(key: Tuple) -> str:
        return f"{key[1]}[{key[2]}]"

    def _new_async(self, r: _Role) -> Tuple[str, VC]:
        self._nasync += 1
        comp = f"A{self._nasync}"
        vc = dict(r.vc)
        vc[comp] = 1
        return comp, vc

    # memory access checks ----------------------------------------------------
    def _read(self, key: Tuple, vc: VC, r: _Role, op: Op, what: str, async_proxy: bool = False) -> _Mem:
        m = self._m(key)
        if m.write_vc is None:
            kind = "uninit_tmem" if key[0] == "T" else "uninit_smem"
            self.fact(kind, f"{what} reads `{self._kname(key)}`, which is never written before this point", op, r,
                      resource=self._kname(key))
        elif not _leq(m.write_vc, vc):
            self.fact("raw_race", f"{what} reads `{self._kname(key)}` without being ordered after the write at "
                      f"{m.write_loc} (by `{m.writer}`): a wait on the barrier that signals that write is missing "
                      f"or waits on the wrong slot/phase", op, r, resource=self._kname(key), writer=m.writer,
                      write_line=m.write_loc.lineno if m.write_loc else None)
        if async_proxy and m.unfenced_by is not None:
            self.fact("proxy_unfenced", f"{what} reads `{self._kname(key)}` through the async proxy, but role "
                      f"`{m.unfenced_by}` wrote it with generic stores and did not execute fence_proxy() before "
                      f"the hand-off", op, r, resource=self._kname(key), writer=m.unfenced_by)
        m.reads.append((dict(vc), op.loc, r.name))
        m.consumed = True
        return m

    def _write(self, key: Tuple, vc: VC, r: _Role, op: Op, what: str) -> _Mem:
        m = self._m(key)
        for (rvc, rloc, rrole) in m.reads:
            if not _leq(rvc, vc):
                self.fact("war_race", f"{what} overwrites `{self._kname(key)}` while the read at {rloc} (by "
                          f"`{rrole}`) is not ordered before it: the reader must release the buffer (arrive/"
                          f"commit on a 'free' barrier) and the writer must wait on that barrier first", op, r,
                          resource=self._kname(key), reader=rrole, read_line=rloc.lineno if rloc else None)
                break
        if m.write_vc is not None and not _leq(m.write_vc, vc) and m.writer != r.name:
            self.fact("waw_race", f"{what} writes `{self._kname(key)}` unordered with the previous write at "
                      f"{m.write_loc} (by `{m.writer}`)", op, r, resource=self._kname(key))
        if m.write_vc is not None and not m.consumed and key[0] == "S":
            self.fact("dead_load", f"`{self._kname(key)}` is overwritten before its previous contents were read",
                      op, r, resource=self._kname(key))
        m.write_vc = dict(vc)
        m.write_loc = op.loc
        m.writer = r.name
        m.reads = []
        m.consumed = False
        return m

    def _reg_read(self, reg: RegTile, r: _Role, op: Op) -> None:
        if reg.name not in r.regs_written:
            self.fact("uninit_reg", f"register tile `{reg.name}` is read before it is written (initialize it "
                      f"with lm.fill or a load)", op, r, register=reg.name)

    # barrier semantics -------------------------------------------------------
    def _arrive(self, key: Tuple[int, int], vc: VC, tx: int = 0) -> None:
        s = self.slots[key]
        s.pending -= 1
        s.arrivals += 1
        s.tx += tx
        _join(s.phase_vc, vc)
        self._maybe_complete(s)

    def _complete_tx(self, key: Tuple[int, int], vc: VC, nbytes: int) -> None:
        s = self.slots[key]
        s.tx -= nbytes
        _join(s.phase_vc, vc)
        self._maybe_complete(s)

    @staticmethod
    def _maybe_complete(s: _Slot) -> None:
        while s.pending <= 0 and s.tx <= 0:
            s.completed += 1
            s.history.append(s.phase_vc)
            s.phase_vc = {}
            s.pending += s.init
            s.tx = min(s.tx, 0)

    def _try_wait(self, r: _Role, op: Wait) -> bool:
        b = op.barrier
        key, slot = self._slot(b, op.slot, r, op)
        n = r.waits.get(key, 0)
        required = n + 1 - (1 if b.released else 0)
        s = self.slots[key]
        if required <= 0:
            r.waits[key] = n + 1
            return True
        if s.completed >= required:
            if s.completed > required:
                self.fact("phase_overrun", f"`{b.name}[{slot}]` completed phase {s.completed} while role "
                          f"`{r.name}` waits for phase {required}: the producer ran two phases ahead, so the "
                          f"parity-based hardware wait would miss a phase (missing back-pressure / free barrier)",
                          op, r, barrier=b.name, slot=slot)
            _join(r.vc, s.history[required - 1])
            r.waits[key] = n + 1
            return True
        r.blocked = (b, slot, required, s.completed, op)
        return False

    # ------------------------------------------------------------------ execution
    def _exec(self, r: _Role, op: Op) -> bool:
        """Execute ``op``; returns False if the role blocks."""
        if isinstance(op, Wait):
            return self._try_wait(r, op)
        if isinstance(op, Arrive):
            key, _ = self._slot(op.barrier, op.slot, r, op)
            self._arrive(key, r.vc)
            r.tick()
            return True
        if isinstance(op, Commit):
            key, _ = self._slot(op.barrier, op.slot, r, op)
            vc = dict(r.vc)
            _join(vc, r.mma_join)
            self._arrive(key, vc)
            r.tick()
            return True
        if isinstance(op, FenceProxy):
            for key in r.unfenced:
                m = self.mem.get(key)
                if m is not None and m.unfenced_by == r.name:
                    m.unfenced_by = None
            r.unfenced.clear()
            return True
        if isinstance(op, TmaLoad):
            key, slot = self._slot(op.barrier, op.slot, r, op)
            leader, total, _ = self.groups[id(op)]
            if leader:
                self._arrive(key, r.vc, tx=total)
                r.tick()
            comp, avc = self._new_async(r)
            skey = self._skey(op.dst, r, op)
            m = self._write(skey, avc, r, op, f"TMA load into `{op.dst.view.name}`")
            m.writer = f"{r.name}(TMA)"
            m.unfenced_by = None
            coords = [int(self._eval(c, r, op)) for c in op.coords]
            box = (1,) * (op.src.ndim - 2) + tuple(op.dst.view.shape)
            inb = [c >= 0 and c + b <= n for c, b, n in zip(coords, box, op.src.shape)]
            anyin = all(c + b > 0 and c < n for c, b, n in zip(coords, box, op.src.shape))
            m.zero_fill = not all(inb)
            if not anyin:
                self.fact("tma_oob", f"TMA box of `{op.src.name}` at {coords} lies entirely outside the tensor "
                          f"{list(op.src.shape)}", op, r, resource=op.src.name, coords=coords)
            elif m.zero_fill:
                self.facts.tma_partial_oob.append({"tensor": op.src.name, "coords": coords, "line": op.loc.lineno})
            self._complete_tx(key, avc, op.nbytes)
            return True
        if isinstance(op, Mma):
            comp, avc = self._new_async(r)
            _join(avc, r.last_mma)
            akey = self._skey(op.a, r, op)
            bkey = self._skey(op.b, r, op)
            self._read(akey, avc, r, op, "MMA operand A", async_proxy=True)
            mb = self._read(bkey, avc, r, op, "MMA operand B", async_proxy=True)
            taint = mb.zero_fill  # zero-filled B rows become accumulator columns (the reduced axis)
            if isinstance(op.acc, RegTile):
                if not (isinstance(op.init, bool) and op.init):
                    if not bool(self._eval(op.init, r, op)) and op.acc.name not in r.regs_written:
                        self._reg_read(op.acc, r, op)
                r.regs_written.add(op.acc.name)
                if taint:
                    r.reg_taint[op.acc.name] = op.b.view.name
                return True
            tkey = self._tkey(op.acc, r, op)
            init = bool(self._eval(op.init, r, op))
            if not init:
                mt = self._m(tkey)
                if mt.write_vc is None:
                    self.fact("uninit_tmem", f"MMA accumulates into `{self._kname(tkey)}` before it was "
                              f"initialized (use init=True on the first MMA)", op, r, resource=self._kname(tkey))
            mt = self._write(tkey, avc, r, op, "MMA")
            mt.zero_fill = taint or (mt.zero_fill and not init)
            r.last_mma = avc
            _join(r.mma_join, avc)
            return True
        if isinstance(op, TmemLoad):
            tkey = self._tkey(op.src, r, op)
            m = self._read(tkey, r.vc, r, op, "tmem_load")
            r.regs_written.add(op.dst.name)
            if m.zero_fill:
                r.reg_taint[op.dst.name] = "tmem"
            else:
                r.reg_taint.pop(op.dst.name, None)
            return True
        if isinstance(op, StSmem):
            self._reg_read(op.src, r, op)
            key = self._skey(op.dst, r, op)
            m = self._write(key, r.vc, r, op, "st_smem")
            m.unfenced_by = r.name
            r.unfenced.add(key)
            r.tick()
            return True
        if isinstance(op, LdSmem):
            key = self._skey(op.src, r, op)
            self._read(key, r.vc, r, op, "ld_smem")
            r.regs_written.add(op.dst.name)
            r.reg_taint.pop(op.dst.name, None)
            if self.mem[key].zero_fill:
                r.reg_taint[op.dst.name] = op.src.view.name
            return True
        if isinstance(op, LoadG):
            coords = [int(self._eval(c, r, op)) for c in op.coords]
            box = (1,) * (op.src.ndim - len(op.dst.shape)) + tuple(op.dst.shape)
            r.regs_written.add(op.dst.name)
            if not all(c >= 0 and c + b <= n for c, b, n in zip(coords, box, op.src.shape)):
                r.reg_taint[op.dst.name] = op.src.name
            else:
                r.reg_taint.pop(op.dst.name, None)
            return True
        if isinstance(op, StoreG):
            self._reg_read(op.src, r, op)
            coords = [int(self._eval(c, r, op)) for c in op.coords]
            box = list((1,) * (op.dst.ndim - len(op.src.shape)) + tuple(op.src.shape))
            lead = len(box) - len(op.src.shape)
            if op.valid is not None:
                v = int(self._eval(op.valid, r, op))
                box[lead] = max(0, min(box[lead], v))
            for c, b, n in zip(coords, box, op.dst.shape):
                if b > 0 and (c < 0 or c + b > n):
                    self.fact("store_oob", f"store of `{op.src.name}` to `{op.dst.name}` at {coords} with extent "
                              f"{box} exceeds the tensor shape {list(op.dst.shape)}", op, r,
                              resource=op.dst.name, coords=coords)
                    break
            return True
        if isinstance(op, Fill):
            r.regs_written.add(op.dst.name)
            r.reg_taint.pop(op.dst.name, None)
            return True
        if isinstance(op, Ewise):
            taint = None
            for a in op.args:
                reg = a if isinstance(a, RegTile) else getattr(a, "reg", None)
                if isinstance(reg, RegTile):
                    self._reg_read(reg, r, op)
                    taint = taint or r.reg_taint.get(reg.name)
            r.regs_written.add(op.dst.name)
            if taint:
                r.reg_taint[op.dst.name] = taint
            else:
                r.reg_taint.pop(op.dst.name, None)
            return True
        if isinstance(op, Reduce):
            self._reg_read(op.src, r, op)
            self._reg_read(op.dst, r, op)
            if op.idx is not None:
                self._reg_read(op.idx, r, op)
                r.regs_written.add(op.idx.name)
            r.regs_written.add(op.dst.name)
            taint = r.reg_taint.get(op.src.name)
            if taint and op.valid is None and op.op != "sum":
                self.fact("tail_taint", f"`{op.op}` over `{op.src.name}`, whose columns include zero-filled "
                          f"out-of-range data (from `{taint}`), without a `valid=` mask", op, r,
                          register=op.src.name, source=taint)
            return True
        if isinstance(op, Convert):
            self._reg_read(op.src, r, op)
            r.regs_written.add(op.dst.name)
            return True
        return True

    def _gc(self) -> None:
        """Drop async clock components every live role has already observed."""
        live = [r for r in self.roles if not r.done]
        if not live:
            return
        comps = [k for k in live[0].vc if k.startswith("A") and all(rr.vc.get(k, 0) >= 1 for rr in live[1:])]
        if not comps:
            return
        drop = set(comps)

        def clean(vc: VC) -> None:
            for k in drop:
                vc.pop(k, None)

        for r in self.roles:
            clean(r.vc)
            clean(r.mma_join)
            clean(r.last_mma)
        for s in self.slots.values():
            clean(s.phase_vc)
            for h in s.history:
                clean(h)
        for m in self.mem.values():
            if m.write_vc is not None:
                clean(m.write_vc)
            for (vc, _, _) in m.reads:
                clean(vc)

    def run(self) -> Facts:
        ops = 0
        try:
            while True:
                progressed = False
                for r in self.roles:
                    if r.done:
                        continue
                    if r.blocked is not None:
                        b, slot, required, _, op = r.blocked
                        r.blocked = None
                        if not self._try_wait(r, op):
                            continue
                        r.cur = None
                        progressed = True
                    while True:
                        if r.cur is None:
                            r.cur = next(r.it, None)
                            if r.cur is None:
                                r.done = True
                                break
                        if not self._exec(r, r.cur):
                            break
                        r.cur = None
                        progressed = True
                        ops += 1
                        if ops % 512 == 0:
                            self._gc()
                        if ops >= self.max_ops:
                            self.facts.truncated = True
                            raise _Abort()
                if all(r.done for r in self.roles):
                    break
                if not progressed:
                    self._deadlock()
                    break
        except _Abort:
            pass
        except CakeError as e:
            self.fact("eval_error", e.message, None, None)
        self.facts.ops_executed = ops
        for key, m in self.mem.items():
            if key[0] == "S" and m.write_vc is not None and not m.consumed and m.writer and "(TMA)" in m.writer:
                self.facts.counts["unconsumed_loads"] = self.facts.counts.get("unconsumed_loads", 0) + 1
        return self.facts

    def _deadlock(self) -> None:
        blocked = [r for r in self.roles if r.blocked is not None]
        desc = []
        for r in blocked:
            b, slot, required, completed, op = r.blocked
            s = self.slots[(id(b), slot)]
            prods = []
            for pr in b.prod:
                st = self.by_name[pr.name]
                if st.done:
                    prods.append(f"{pr.name} (finished)")
                elif st.blocked is not None:
                    pb = st.blocked
                    prods.append(f"{pr.name} (blocked on {pb[0].name}[{pb[1]}])")
                else:
                    prods.append(pr.name)
            desc.append({"role": r.name, "barrier": b.name, "slot": slot, "waiting_for_phase": required,
                         "completed_phases": completed, "pending_arrivals": s.pending,
                         "producers": prods, "line": op.loc.lineno})
        first = blocked[0] if blocked else None
        if first is None:
            return
        b, slot, required, completed, op = first.blocked
        cyc = "; ".join(f"`{d['role']}` waits on `{d['barrier']}[{d['slot']}]` for phase {d['waiting_for_phase']} "
                        f"({d['completed_phases']} completed; producers: {', '.join(d['producers']) or 'none'})"
                        for d in desc)
        hint = None
        first_waits = [d for d in desc if d["waiting_for_phase"] == 1 and d["completed_phases"] == 0]
        for d in first_waits:
            bar = next(x for x in self.p.barriers if x.name == d["barrier"])
            if not bar.released and any("blocked" in p for p in d["producers"]):
                hint = (f"`{bar.name}` is waited on before anything can signal it; if it marks free buffer slots, "
                        f"declare it with released=True so the first wait on each slot passes")
                break
        if hint is None and any("finished" in p for d in desc for p in d["producers"]):
            hint = "a producer role finished without enough arrive/commit operations for the phases consumers wait for"
        self.fact("deadlock", "deadlock: " + cyc, op, first, barrier=b.name, slot=slot, blocked=desc,
                  hint=hint)


class _Abort(Exception):
    pass


def analyze(prog: Program, block_idx: Tuple[int, int, int] = (0, 0, 0), max_ops: int = 400_000) -> Facts:
    return Analyzer(prog, block_idx, max_ops).run()


def representative_ctas(prog: Program) -> List[Tuple[int, int, int]]:
    gx, gy, gz = prog.grid
    first = (0, 0, 0)
    last = (gx - 1, gy - 1, gz - 1)
    return [first] if first == last else [first, last]
