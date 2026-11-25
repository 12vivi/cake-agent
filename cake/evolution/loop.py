"""Evidence-driven compiler-harness evolution (paper Sec. 3.2, Figure 4).

One ``harness_step`` consumes the evidence routed by kernel evolution and
proposes harness changes, each of which must pass its gates before merging:

* **verifier rules** from recurring runtime failures the verifier missed
  ("an opaque runtime crash becomes a verifier rule").  A proposed rule must
  flag the failing programs (true positives), must not flag any corpus kernel
  or any candidate that passed numerical validation (no false positives), and
  must pass the merge gate (auto-approve or a human);
* **cost-model calibration** when predictions are systematically off ("a
  systematic misprediction becomes a calibration target"), gated on held-out
  error and rank correlation;
* **tactics** distilled from large improvements, injected into later prompts;
* **IR-vocabulary proposals** from capability requests, written for human review
  against the design principles P1-P8.

With an LLM backend, rule code is synthesized by the model; offline, proposals
come from a library of rule templates keyed by sanitizer signatures (including
a deliberately over-broad variant, to exercise the false-positive gate).
"""
from __future__ import annotations

import inspect
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..corpus import build_entry, corpus
from ..cost.calibrate import calibrate, metrics
from ..errors import CakeError
from ..harness import sandbox
from ..verify import verify
from ..verify.rules import LIBRARY, SEED, Rule
from .state import compile_rule

# sanitizer signature -> library rules to try (in order)
SIGNATURE_RULES = {
    "TMEM_LANE_QUADRANT": ["HW.TMEM_LANES_STRICT", "HW.TMEM_LANES"],
    "PROXY_FENCE": ["SAFE.PROXY_FENCE"],
    "TMA_ZERO_FILL": ["DC.TAIL_MASK"],
}

# an over-broad rule an agent might plausibly propose: "epilogue roles must have exactly 4 warps"
_STRICT_LANES = '''def check(ctx):
    for op in ctx.ops():
        if isinstance(op, TmemLoad) and len(op.role.warps) != 4:
            yield ctx.finding("role `" + op.role.name + "` reads TMEM with " + str(len(op.role.warps)) +
                              " warps; TMEM loads require exactly 4 warps", op.loc)
'''

RULE_API = """
Rule API: write `def check(ctx):` as a generator that yields `ctx.finding(message, loc, hint=None, **subject)`.
- `ctx.program`: the Cake IR program (`.roles`, `.views`, `.tmems`, `.barriers`, `.regs`, `.warps`, `.grid`,
  `.body[role]`); `ctx.ops()` iterates every operation (each has `.role` with `.warps`/`.name` and `.loc`).
- Operation classes in scope: TmaLoad (dst SmemRef, src, coords, barrier, slot), Wait/Arrive/Commit (barrier, slot),
  Mma (acc, a, b, init), TmemLoad (dst RegTile, src TmemRef; `src.shape`), LoadG, StoreG, StSmem, LdSmem, Fill,
  Ewise, Reduce (dst, src, op, idx, valid), Convert, FenceProxy, Loop (var, start, stop, step, body).
- `ctx.target`: target info (`.name`, `.tmem`, `.tcgen05`, ...).
- `ctx.all_facts(*kinds)`: facts from abstract execution (kinds include deadlock, raw_race, war_race, waw_race,
  uninit_smem, uninit_tmem, uninit_reg, phase_overrun, slot_range, stage_range, proxy_unfenced, tail_taint,
  tma_oob, store_oob); each fact has `.message`, `.loc`, `.role`, `.subject`.
No imports. Only flag what is certainly a contract violation: the rule must not reject valid kernels.
"""


def _signature(item: Dict[str, Any]) -> str:
    kinds = sorted({n["kind"] for n in item.get("notes") or [] if n["kind"] not in ("UNINIT_SMEM_READ",)})
    if kinds:
        return "+".join(kinds)
    err = item.get("runtime_error") or {}
    return f"fault:{err.get('fault')}" if err else "mismatch"


def _rule_flags(rule: Rule, prog, target) -> bool:
    rep = verify(prog, target, rules=[rule])
    return any(f.rule == rule.id for f in rep.findings)


def gate_rule(rule: Rule, failing_sources: List[str], harness, passing_sources: List[str],
              min_recall: float = 0.8) -> Dict[str, Any]:
    """Run the three automatic gates for a proposed verifier rule."""
    w, t = harness.w, harness.target
    tp = 0
    built = 0
    for src in failing_sources:
        try:
            prog = sandbox.build(src, w, w.target_dims, t.name)
        except CakeError:
            continue
        built += 1
        if _rule_flags(rule, prog, t):
            tp += 1
    recall = tp / built if built else 0.0
    corpus_fp: List[str] = []
    for e in corpus():
        try:
            prog = build_entry(e)
        except CakeError:
            continue
        try:
            if _rule_flags(rule, prog, t):
                corpus_fp.append(e.name)
        except Exception as ex:  # a crashing rule fails the gate
            corpus_fp.append(f"{e.name} (rule raised {type(ex).__name__})")
    archive_fp = 0
    for src in passing_sources:
        try:
            prog = sandbox.build(src, w, w.target_dims, t.name)
            if _rule_flags(rule, prog, t):
                archive_fp += 1
        except CakeError:
            continue
    ok = recall >= min_recall and not corpus_fp and archive_fp == 0
    return {"rule": rule.id, "recall": round(recall, 3), "true_positives": tp, "failing_programs": built,
            "corpus_false_positives": corpus_fp, "archive_false_positives": archive_fp,
            "corpus_size": len(corpus()), "passed": ok}


def _library_rule(rid: str) -> Optional[Rule]:
    if rid == "HW.TMEM_LANES_STRICT":
        return compile_rule(rid, "hardware_conformance", "gate", "TMEM loads are issued by exactly 4 warps",
                            _STRICT_LANES)
    return LIBRARY.get(rid)


def _llm_rule(llm, rid: str, sig: str, items: List[Dict[str, Any]]) -> Optional[Rule]:
    """Ask an LLM to synthesize rule code for a failure cluster."""
    examples = []
    for it in items[:2]:
        notes = "\n".join(f"  - {n['kind']} at line {n['loc'].get('line')}: {n['detail'].get('reason', '')}"
                          for n in (it.get("notes") or [])[:4])
        examples.append(f"```python\n{it['source'][:5000]}\n```\nSanitizer evidence:\n{notes}\n"
                        f"Failure: {it.get('failure')}")
    seed_src = "\n\n".join(textwrap.dedent(inspect.getsource(SEED[r].check)) for r in ("SAFE.VIEW_BOUNDS",))
    prompt = (f"The following Cake IR programs passed every verifier gate but failed at run time with the same "
              f"sanitizer signature `{sig}`.\n\n" + "\n\n".join(examples) +
              f"\n\nWrite a new verifier rule that rejects this class of programs before they run.\n{RULE_API}\n"
              f"Example of an existing rule:\n```python\n{seed_src}\n```\n"
              f"Reply with one ```python block defining `def check(ctx):` and one line `CATEGORY: <program_safety|"
              f"hardware_conformance|data_consistency|schedule_semantics>`.")
    try:
        resp = llm.generate("You maintain the static verifier of a GPU kernel compiler.", prompt, None)
    except Exception:
        return None
    m = re.search(r"```(?:python)?\s*\n(.*?)```", resp.text, re.S)
    if not m:
        return None
    cat = re.search(r"CATEGORY:\s*(\w+)", resp.text)
    try:
        return compile_rule(rid, cat.group(1) if cat else "program_safety", "gate",
                            f"evolved from runtime failures with signature {sig}", m.group(1))
    except CakeError:
        return None


def _approve(auto: bool, what: str) -> bool:
    if auto:
        return True
    try:
        return input(f"[merge gate] accept {what}? [y/N] ").strip().lower().startswith("y")
    except EOFError:
        return False


def harness_step(harness, archive, llm=None, auto_approve: bool = True, min_support: int = 2,
                 verbose: bool = True) -> Dict[str, Any]:
    """One outer-loop step of compiler-harness evolution."""
    state = harness.state
    report: Dict[str, Any] = {"rules": [], "calibration": None, "tactics": 0, "proposals": 0}
    ev = archive.evidence
    passing = [r.source for r in archive.ok()][-12:]
    # ---------------------------------------------------------------- verifier rules
    gaps = [e for e in ev if e["kind"] == "verifier_gap" and not e.get("_consumed")]
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in gaps:
        clusters[_signature(e)].append(e)
    active = {r.id for r in harness.rules()}
    for sig, items in clusters.items():
        if len(items) < min_support:
            continue
        cand_ids: List[str] = []
        for key, rids in SIGNATURE_RULES.items():
            if key in sig:
                cand_ids += [r for r in rids if r not in active]
        proposals: List[Rule] = []
        use_llm = llm is not None and getattr(llm, "name", "") != "scripted"
        if use_llm:
            rid = f"EVOLVED.{re.sub(r'[^A-Z0-9]+', '_', sig.upper())[:40]}_{state.version + 1}"
            r = _llm_rule(llm, rid, sig, items)
            if r is not None:
                proposals.append(r)
        proposals += [r for r in (_library_rule(i) for i in cand_ids) if r is not None]
        for rule in proposals:
            res = gate_rule(rule, [it["source"] for it in items], harness, passing)
            res["signature"] = sig
            res["support"] = len(items)
            approved = res["passed"] and _approve(auto_approve, f"verifier rule {rule.id} ({sig})")
            res["merged"] = approved
            report["rules"].append(res)
            if verbose:
                print(f"  [harness] rule {rule.id} for `{sig}` (support {len(items)}): recall {res['recall']}, "
                      f"corpus FP {res['corpus_false_positives'] or 0}, archive FP {res['archive_false_positives']}"
                      f" -> {'MERGED' if approved else 'rejected'}", flush=True)
            if approved:
                if rule.origin == "library":
                    state.library_enabled.append(rule.id)
                else:
                    state.evolved[rule.id] = {"category": rule.category, "disposition": rule.disposition,
                                              "description": rule.description, "source": rule.source}
                state.record("verifier_rule", rule=rule.id, signature=sig, gate=res,
                             evidence=[it["record"] for it in items])
                for it in items:
                    it["_consumed"] = True
                break
            state.record("verifier_rule_rejected", rule=rule.id, signature=sig, gate=res)
    # ---------------------------------------------------------------- cost-model calibration
    samples = [(e["features"], e["measured_us"]) for e in ev if e["kind"] == "calibration_sample"]
    cur = harness.calibration()
    if cur is not None and len(samples) >= 12:
        m = metrics(samples, cur.coef)
        if m["mae_log"] > 0.08 or abs(m["bias"]) > 0.06:
            new, rep, ok = calibrate(samples, cur)
            rep["trigger"] = {k: round(v, 3) for k, v in m.items()}
            approved = ok and _approve(auto_approve, f"cost-model calibration v{cur.version + 1}")
            rep["merged"] = approved
            report["calibration"] = rep
            if verbose:
                after = rep.get("after", {})
                print(f"  [harness] calibration on {len(samples)} samples: held-out |log err| "
                      f"{rep.get('before', {}).get('mae_log', float('nan')):.3f} -> {after.get('mae_log', float('nan')):.3f}"
                      f", spearman {rep.get('before', {}).get('spearman', float('nan')):.3f} -> "
                      f"{after.get('spearman', float('nan')):.3f} -> {'MERGED' if approved else 'kept'}", flush=True)
            if approved:
                state.calibrations[cur.target] = new
                state.record("calibration", target=cur.target, report={k: v for k, v in rep.items() if k != "coef"},
                             coef=new.coef)
    # ---------------------------------------------------------------- tactics
    known = {t["idea"] for t in state.tactics}
    for e in sorted((e for e in ev if e["kind"] == "tactic"), key=lambda e: -e["gain"]):
        idea = (e.get("idea") or "").strip()
        if not idea or idea in known or idea.startswith("fix:"):
            continue
        known.add(idea)
        state.tactics.append({"idea": idea, "gain": round(e["gain"], 3), "workload": harness.w.name,
                              "text": f"{idea} -- gave {e['gain']:.2f}x over its parent ({harness.w.name})"})
        report["tactics"] += 1
    state.tactics.sort(key=lambda t: -t["gain"])
    del state.tactics[12:]
    if report["tactics"]:
        state.record("tactics", added=report["tactics"])
    # ---------------------------------------------------------------- IR vocabulary proposals
    for e in ev:
        if e["kind"] == "ir_gap" and not e.get("_consumed"):
            e["_consumed"] = True
            reqs = e.get("requests") or [e.get("error") or "unknown capability"]
            for req in reqs:
                state.proposals.append({
                    "request": req, "evidence": e["record"], "status": "needs human review",
                    "principles": {p: "unchecked" for p in ("P1 ergonomic", "P2 performance-transparent",
                                                            "P3 canonical", "P4 statically typed",
                                                            "P5 analysis-friendly", "P6 test-gated",
                                                            "P7 analysis-consistent", "P8 hardware-grounded")}})
                report["proposals"] += 1
    if state.root:
        state.save()
    return report
