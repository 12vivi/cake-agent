"""The evaluation harness: the environment an agent's candidate goes through.

Cake arm (paper Sec. 4): construct (typed IR checks) -> verifier hard gates ->
cost-model estimate -> lower -> numerical validation against the oracle on the
validation suite -> timing at the target shape -> profile.  Every failure is
returned as a *localized* diagnostic.

Direct-code arm (control): construct (assembler checks) -> run -> correctness ->
timing; failures surface only as compiler errors, coarse runtime errors,
correctness outcomes and end-to-end timing.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..cost.model import estimate, featurize
from ..errors import CakeError, SimError
from ..evolution.state import HarnessState
from ..ir.core import Program
from ..lower import lower
from ..sim import device
from ..targets import get_target
from ..verify import verify
from ..workloads.base import Case, Workload
from . import sandbox


def source_hash(source: str) -> str:
    return hashlib.sha1(source.strip().encode()).hexdigest()[:12]


@dataclass
class EvalRecord:
    id: str
    arm: str
    source: str
    source_hash: str
    stage: str = "new"   # construct | verify | filtered | runtime | correctness | timed
    ok: bool = False
    construct_error: Optional[Dict[str, Any]] = None
    findings: List[Dict[str, Any]] = field(default_factory=list)
    coverage: List[str] = field(default_factory=list)
    estimate: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    runtime_error: Optional[Dict[str, Any]] = None
    cases: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    timing: Optional[Dict[str, Any]] = None
    speedup: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0

    @property
    def predicted_us(self) -> Optional[float]:
        return (self.estimate or {}).get("us")

    @property
    def measured_us(self) -> Optional[float]:
        return (self.timing or {}).get("median_us")

    @property
    def failure_class(self) -> Optional[str]:
        if self.construct_error:
            return f"construct:{self.construct_error.get('kind')}"
        if self.stage == "verify":
            gates = [f["rule"] for f in self.findings if f["disposition"] == "gate"]
            return f"verify:{gates[0] if gates else '?'}"
        if self.runtime_error:
            return f"runtime:{self.runtime_error.get('fault')}"
        if self.stage == "correctness" and not self.ok:
            return "mismatch"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalRecord":
        r = EvalRecord(d["id"], d["arm"], d["source"], d["source_hash"])
        r.__dict__.update(d)
        return r


class Harness:
    def __init__(self, workload: Workload, target: Optional[str] = None, state: Optional[HarnessState] = None,
                 arm: str = "cake", samples: int = 5, raw_feedback: str = "profile", seed: int = 0):
        self.w = workload
        self.target = get_target(target or workload.target)
        self.state = state or HarnessState()
        self.arm = arm
        self.samples = samples
        self.raw_feedback = raw_feedback
        self.seed = seed
        self._cases: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = {}
        self._baseline: Optional[float] = None
        self._rules = None
        self._counter = 0

    # ------------------------------------------------------------------ helpers
    @property
    def kind(self) -> str:
        return "schedule" if self.arm == "cake" else "raw"

    def rules(self):
        if self._rules is None:
            self._rules = self.state.rules()
        return self._rules

    def refresh(self) -> None:
        """Pick up harness-state changes (new rules, calibrations)."""
        self._rules = None

    def calibration(self):
        return self.state.calibrations.get(self.target.name)

    def build(self, source: str, dims: Dict[str, int]):
        return sandbox.build(source, self.w, dims, self.target.name, kind=self.kind)

    def kernel(self, built):
        return lower(built) if isinstance(built, Program) else built

    def case_data(self, case: Case):
        if case.name not in self._cases:
            dims = case.dim_dict
            inp = self.w.make_inputs(dims, case.dist, case.seed)
            self._cases[case.name] = (inp, self.w.oracle(inp, dims))
        return self._cases[case.name]

    def baseline_us(self) -> float:
        if self._baseline is None:
            prog = sandbox.build(self.w.baseline_source, self.w, self.w.target_dims, self.target.name)
            self._baseline = device.measure(lower(prog), self.target, samples=self.samples, seed=self.seed).median_us
        return self._baseline

    def new_id(self, prefix: str = "c") -> str:
        self._counter += 1
        return f"{prefix}{self._counter:04d}"

    # ------------------------------------------------------------------ stages
    def precheck(self, source: str, meta: Optional[Dict[str, Any]] = None, rid: Optional[str] = None) -> EvalRecord:
        """Stage 2 of the workflow: IR construction checks, verifier gates and cost-model ranking."""
        t0 = time.time()
        rec = EvalRecord(rid or self.new_id(), self.arm, source, source_hash(source), meta=dict(meta or {}))
        try:
            built = self.build(source, self.w.target_dims)
        except CakeError as e:
            rec.stage = "construct"
            rec.construct_error = e.to_dict()
            rec.wall_s = time.time() - t0
            return rec
        rec._built = built  # type: ignore[attr-defined]
        if self.arm == "cake":
            report = verify(built, self.target, rules=self.rules())
            rec.findings = [f.to_dict() for f in report.findings]
            rec.coverage = report.coverage
            if report.blocked:
                rec.stage = "verify"
                rec.wall_s = time.time() - t0
                return rec
            est = estimate(built, self.target, calibration=self.calibration())
            rec.estimate = est.to_dict()
            if est.features is not None:
                rec.features = est.features
            elif est.covered is False:
                try:
                    rec.features = featurize(built, self.target)
                except CakeError:
                    pass
        rec.stage = "filtered"
        rec.wall_s = time.time() - t0
        return rec

    def device_eval(self, rec: EvalRecord) -> EvalRecord:
        """Stage 3: numerical validation on the validation suite, then timing at the target shape."""
        t0 = time.time()
        notes: List[Dict[str, Any]] = []
        for case in self.w.cases:
            dims = case.dim_dict
            try:
                built = self.build(rec.source, dims)
                k = self.kernel(built)
                inp, expected = self.case_data(case)
                gmem = {**inp, **self.w.make_outputs(dims)}
                ctas = device.sample_ctas(k, case.sample_ctas) if case.sample_ctas else None
                run = device.run(k, self.target, gmem, ctas=ctas, seed=self.seed)
                notes.extend(n for n in run.notes if n not in notes)
            except SimError as e:
                rec.stage = "runtime"
                rec.runtime_error = {**e.to_dict(), "case": case.name, "coarse": e.coarse()}
                rec.notes = notes
                rec.wall_s += time.time() - t0
                return rec
            except CakeError as e:
                rec.stage = "construct"
                rec.construct_error = {**e.to_dict(), "case": case.name}
                rec.wall_s += time.time() - t0
                return rec
            outs = {t.name: gmem[t.name] for t in self.w.tensors if t.role == "output"}
            res = self.w.check(outs, expected, inp, dims, full_coverage=not case.sample_ctas)
            rec.cases.append({"case": case.name, **res.to_dict()})
            if not res.ok:
                rec.stage = "correctness"
                rec.notes = notes
                rec.wall_s += time.time() - t0
                return rec
        rec.notes = notes
        try:
            built = getattr(rec, "_built", None) or self.build(rec.source, self.w.target_dims)
            tm = device.measure(self.kernel(built), self.target, samples=self.samples, seed=self.seed)
        except SimError as e:
            rec.stage = "runtime"
            rec.runtime_error = {**e.to_dict(), "case": "timing", "coarse": e.coarse()}
            rec.wall_s += time.time() - t0
            return rec
        rec.timing = tm.to_dict()
        rec.speedup = self.baseline_us() / tm.median_us
        rec.stage = "timed"
        rec.ok = True
        rec.wall_s += time.time() - t0
        return rec

    def evaluate(self, source: str, meta: Optional[Dict[str, Any]] = None, run_device: bool = True) -> EvalRecord:
        rec = self.precheck(source, meta)
        if rec.stage == "filtered" and run_device:
            self.device_eval(rec)
        return rec

    # ------------------------------------------------------------------ feedback
    def feedback(self, rec: EvalRecord, max_findings: int = 8) -> str:
        """Agent-visible feedback; the Cake arm gets localized diagnostics, the direct arm coarse outcomes."""
        if self.arm == "cake":
            return self._feedback_cake(rec, max_findings)
        return self._feedback_raw(rec)

    @staticmethod
    def _loc(d: Optional[Dict[str, Any]]) -> str:
        if not d or not d.get("line"):
            return ""
        return f" (line {d['line']}: `{d.get('text', '').strip()}`)"

    def _feedback_cake(self, rec: EvalRecord, max_findings: int) -> str:
        L: List[str] = []
        if rec.construct_error:
            e = rec.construct_error
            L.append(f"IR construction failed [{e.get('kind')}]: {e.get('message')}{self._loc(e.get('loc'))}")
            if e.get("hint"):
                L.append(f"  hint: {e['hint']}")
            return "\n".join(L)
        gates = [f for f in rec.findings if f["disposition"] == "gate"]
        others = [f for f in rec.findings if f["disposition"] != "gate"]
        if gates:
            L.append(f"Verifier rejected the schedule before compilation ({len(gates)} blocking finding(s)):")
            for f in gates[:max_findings]:
                L.append(f"- [{f['rule']}] {f['message']}{self._loc(f.get('loc'))}")
                if f.get("hint"):
                    L.append(f"    hint: {f['hint']}")
            return "\n".join(L)
        if rec.runtime_error:
            e = rec.runtime_error
            L.append(f"Runtime failure on case `{e.get('case')}` ({e.get('fault')}): {e.get('message')}"
                     f"{self._loc(e.get('loc'))}")
            for b in (e.get("detail") or {}).get("blocked", [])[:4]:
                L.append(f"  blocked: role `{b.get('role')}` on `{b.get('mbar_name')}`{self._loc(b.get('loc'))}")
        bad = [c for c in rec.cases if not c["ok"]]
        if bad:
            c = bad[0]
            L.append(f"Numerical validation failed on case `{c['case']}`: {c['summary']}")
            for ex in c.get("examples", [])[:3]:
                L.append(f"  e.g. {ex}")
        if (rec.runtime_error or bad) and rec.notes:
            L.append("Sanitizer evidence:")
            for n in rec.notes[:6]:
                reason = n["detail"].get("reason", "")
                L.append(f"  - {n['kind']} in role `{n['role']}`{self._loc(n.get('loc'))} {reason}")
        if rec.ok:
            L.append(f"Correct on all validation cases. Measured {rec.measured_us:.1f} us at the target shape "
                     f"= {rec.speedup:.3f}x the baseline ({self.baseline_us():.1f} us).")
            if rec.estimate and rec.estimate.get("covered"):
                L.append(f"Cost model predicted {rec.estimate['us']:.1f} us; bottleneck: {rec.estimate['bottleneck']}. "
                         f"{rec.estimate['attribution']}")
                for h in rec.estimate.get("hints", [])[:3]:
                    L.append(f"  model hint: {h}")
            if rec.timing:
                L.append("Profile: " + device.summarize_profile(rec.timing["profile"]).replace("\n", "\n  "))
        for f in others[:max_findings]:
            L.append(f"- {f['disposition'].upper()} [{f['rule']}] {f['message']}{self._loc(f.get('loc'))}")
            if f.get("hint"):
                L.append(f"    hint: {f['hint']}")
        return "\n".join(L) if L else "no feedback"

    def _feedback_raw(self, rec: EvalRecord) -> str:
        if rec.construct_error:
            e = rec.construct_error
            return f"compile error: {e.get('message')}{self._loc(e.get('loc'))}"
        if rec.runtime_error:
            return f"{rec.runtime_error['coarse']} (case `{rec.runtime_error.get('case')}`)"
        bad = [c for c in rec.cases if not c["ok"]]
        if bad:
            c = bad[0]
            ex = "; ".join(str(x) for x in c.get("examples", [])[:3])
            return f"wrong results on case `{c['case']}`: {c['summary']}" + (f" (e.g. {ex})" if ex else "")
        if rec.ok:
            s = (f"correct; measured {rec.measured_us:.1f} us = {rec.speedup:.3f}x the baseline "
                 f"({self.baseline_us():.1f} us)")
            if self.raw_feedback == "profile" and rec.timing:
                util = rec.timing["profile"].get("unit_utilization", {})
                s += "; unit utilization: " + ", ".join(f"{u} {v:.0%}" for u, v in
                                                       sorted(util.items(), key=lambda x: -x[1]))
            return s
        return "no result"
