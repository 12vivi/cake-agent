"""The kernel-evolution loop (paper Sec. 4).

A run has four stages per round:

1. **generate** structurally distinct candidates (LLM call);
2. **filter** them with IR construction checks, verifier hard gates and
   cost-model ranking before spending device time;
3. **evaluate** survivors against the external oracle with benchmarking and
   profiler evidence;
4. **route** the resulting evidence to the candidate (feedback), the verifier
   (runtime failures it missed), the cost model (calibration samples), the IR
   vocabulary (capability requests) and the tactic library.

The workload contract fixes shapes, oracle, tolerances, hardware and permitted
references; every record and every piece of routed evidence is retained.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..evolution.state import HarnessState
from ..harness import sandbox
from ..harness.evaluate import EvalRecord, Harness
from ..lir.printer import to_raw_source
from ..lower import lower
from ..workloads import get_workload
from .archive import Archive
from .llm import LLM, LLMError, make_llm
from .prompts import GenContext, parse_candidates, system_prompt, user_prompt

_REQUEST = re.compile(r"#\s*CAKE-REQUEST:\s*(.+)")


@dataclass
class EvolveConfig:
    workload: str = "kmeans_assign"
    arm: str = "cake"                     # "cake" (Cake IR) or "raw" (direct CUDA/PTX-level)
    llm: Union[str, LLM] = "scripted"
    target: Optional[str] = None
    budget_tokens: int = 2_000_000
    candidates_per_round: int = 4
    device_evals_per_round: int = 2
    max_rounds: int = 100_000
    plateau_window: float = 0.25          # fraction of the budget
    plateau_eps: float = 0.01             # relative improvement that resets the plateau clock
    seed: int = 0
    out_dir: Optional[str] = None
    state: Optional[HarnessState] = None
    co_evolve: bool = False               # run harness evolution every `evolve_every` rounds
    evolve_every: int = 8
    auto_approve: bool = True             # merge gate for harness changes (False = ask a human)
    samples: int = 5
    raw_feedback: str = "profile"
    verbose: bool = True


@dataclass
class RunResult:
    summary: Dict[str, Any]
    archive: Archive
    harness: Harness
    harness_reports: List[Dict[str, Any]] = field(default_factory=list)


def route(rec: EvalRecord, parent_speedup: Optional[float]) -> List[Dict[str, Any]]:
    """Stage 4: send each piece of evidence where it can improve the system."""
    base = {"record": rec.id, "round": rec.meta.get("round"), "arm": rec.arm}
    out: List[Dict[str, Any]] = []
    if rec.stage in ("runtime", "correctness"):
        out.append({**base, "kind": "verifier_gap" if rec.arm == "cake" else "runtime_failure",
                    "failure": rec.failure_class, "notes": rec.notes, "runtime_error": rec.runtime_error,
                    "cases": [c for c in rec.cases if not c["ok"]], "source": rec.source,
                    "rules_active": rec.meta.get("rules_active")})
    if rec.stage == "verify":
        out.append({**base, "kind": "verifier_catch",
                    "rules": sorted({f["rule"] for f in rec.findings if f["disposition"] == "gate"})})
    if rec.ok and rec.features is not None and rec.measured_us:
        out.append({**base, "kind": "calibration_sample", "features": rec.features, "measured_us": rec.measured_us,
                    "predicted_us": rec.predicted_us})
    reqs = _REQUEST.findall(rec.source)
    if reqs or (rec.construct_error and "has no attribute" in (rec.construct_error.get("message") or "")):
        out.append({**base, "kind": "ir_gap", "requests": reqs,
                    "error": (rec.construct_error or {}).get("message"), "source": rec.source})
    if rec.ok and parent_speedup and rec.speedup and rec.speedup > parent_speedup * 1.03:
        out.append({**base, "kind": "tactic", "idea": rec.meta.get("idea", ""),
                    "gain": rec.speedup / parent_speedup, "speedup": rec.speedup, "source": rec.source,
                    "parent": rec.meta.get("parent")})
    return out


def _plateau(checkpoints: List[Dict[str, Any]], tokens: int, budget: int, window: float, eps: float) -> bool:
    w = window * budget
    if tokens < w or not checkpoints:
        return False
    best_now = checkpoints[-1]["best_speedup"]
    earlier = [c for c in checkpoints if c["tokens"] <= tokens - w]
    if not earlier:
        return False
    return best_now <= earlier[-1]["best_speedup"] * (1 + eps) and best_now > 0


def evolve(cfg: EvolveConfig) -> RunResult:
    w = get_workload(cfg.workload)
    state = cfg.state or HarnessState()
    harness = Harness(w, cfg.target, state, cfg.arm, samples=cfg.samples, raw_feedback=cfg.raw_feedback,
                      seed=cfg.seed)
    llm = make_llm(cfg.llm, seed=cfg.seed) if isinstance(cfg.llm, str) else cfg.llm
    rng = random.Random(cfg.seed)
    archive = Archive(cfg.out_dir)
    t_start = time.time()

    if cfg.arm == "cake":
        baseline_src = w.baseline_source
    else:
        prog = sandbox.build(w.baseline_source, w, w.target_dims, harness.target.name)
        baseline_src = to_raw_source(lower(prog), prog.name)
    from ..agent.scripted import _templates
    try:
        base_params = _templates(w.name)[2]
    except KeyError:
        base_params = {}
    base_rec = harness.evaluate(baseline_src, meta={"idea": "baseline", "round": 0, "tokens_at": 0,
                                                    "scripted": {"params": dict(base_params), "bugs": []}})
    feedback: Dict[str, str] = {base_rec.id: harness.feedback(base_rec)}
    base_rec.meta["feedback"] = feedback[base_rec.id]
    archive.add(base_rec)
    if not base_rec.ok:
        raise RuntimeError(f"the baseline does not pass the harness: {feedback[base_rec.id]}")
    system = system_prompt(w, harness.target.summary(), cfg.arm, cfg.candidates_per_round, baseline_src)
    tokens, rnd = 0, 0
    archive.checkpoint(0, 0.0, 0)
    plateau_at: Optional[int] = None
    reports: List[Dict[str, Any]] = []
    stop_reason = "budget"
    while tokens < cfg.budget_tokens and rnd < cfg.max_rounds:
        rnd += 1
        parents = archive.select_parents(2, rng)
        failures = archive.recent_failures(3)
        ctx = GenContext(cfg.arm, w, cfg.candidates_per_round, rnd, parents, failures, feedback, state.tactics,
                         harness.baseline_us(), rng, harness)
        prompt = user_prompt(ctx)
        try:
            resp = llm.generate(system, prompt, ctx)
        except LLMError as e:
            stop_reason = f"llm error: {e}"
            break
        tokens += resp.total_tokens
        cands = parse_candidates(resp.text, cfg.candidates_per_round)
        metas = getattr(llm, "last_meta", None) or []
        recs: List[EvalRecord] = []
        for i, (idea, src) in enumerate(cands):
            meta: Dict[str, Any] = {"idea": idea, "round": rnd, "tokens_at": tokens,
                                    "rules_active": len(harness.rules()) if cfg.arm == "cake" else 0}
            if i < len(metas):
                meta["scripted"] = metas[i]
                meta["parent"] = metas[i].get("parent")
            elif parents:
                meta["parent"] = parents[0].id
            recs.append(harness.precheck(src, meta))
        # stage 2: rank the survivors of the static gates and spend device time on the best ones
        eligible = [r for r in recs if r.stage == "filtered"]
        if cfg.arm == "cake":
            eligible.sort(key=lambda r: r.predicted_us if r.predicted_us is not None else math.inf)
        for r in eligible[:cfg.device_evals_per_round]:
            harness.device_eval(r)
        speed = {r.id: r.speedup for r in archive.records}
        for r in recs:
            fb = harness.feedback(r)
            if r.stage == "filtered":
                pred = f" (predicted {r.predicted_us:.1f} us)" if r.predicted_us else ""
                fb = (f"Not run on the device: ranked below this round's other candidates{pred}.\n" + fb).strip()
            feedback[r.id] = fb
            r.meta["feedback"] = fb
            for item in route(r, speed.get(r.meta.get("parent"))):
                archive.add_evidence(item)
            archive.add(r)
        archive.checkpoint(tokens, time.time() - t_start, rnd)
        if plateau_at is None and _plateau(archive.checkpoints, tokens, cfg.budget_tokens, cfg.plateau_window,
                                           cfg.plateau_eps):
            plateau_at = tokens
        if cfg.co_evolve and rnd % cfg.evolve_every == 0:
            from ..evolution.loop import harness_step
            rep = harness_step(harness, archive, llm=llm, auto_approve=cfg.auto_approve, verbose=cfg.verbose)
            reports.append(rep)
            harness.refresh()
        if cfg.verbose:
            stages = ",".join(r.stage[0] + ("+" if r.ok else "") for r in recs)
            print(f"[{cfg.arm}] round {rnd:3d} tokens {tokens / 1e6:6.2f}M best {archive.best_speedup():.3f}x "
                  f"[{stages}]", flush=True)
    else:
        stop_reason = "budget" if tokens >= cfg.budget_tokens else "max_rounds"
    best = archive.best(1)[0]
    recs = archive.records[1:]
    summary = {
        "workload": w.name, "arm": cfg.arm, "llm": getattr(llm, "name", str(llm)), "target": harness.target.name,
        "seed": cfg.seed, "budget_tokens": cfg.budget_tokens, "tokens_used": tokens, "rounds": rnd,
        "stop_reason": stop_reason,
        "candidates": len(recs),
        "construct_errors": sum(1 for r in recs if r.stage == "construct"),
        "verifier_rejects": sum(1 for r in recs if r.stage == "verify"),
        "ranked_out": sum(1 for r in recs if r.stage == "filtered"),
        "device_evals": sum(1 for r in recs if r.stage in ("runtime", "correctness", "timed")),
        "device_failures": sum(1 for r in recs if r.stage in ("runtime", "correctness")),
        "best_speedup": best.speedup, "best_us": best.measured_us, "baseline_us": harness.baseline_us(),
        "best_id": best.id, "best_idea": best.meta.get("idea"),
        "plateau_tokens": plateau_at, "active_evolve_s": round(time.time() - t_start, 1),
        "harness_version": state.version, "active_rules": [r.id for r in harness.rules()],
    }
    archive.save_summary(summary)
    return RunResult(summary, archive, harness, reports)
