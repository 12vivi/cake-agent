"""Fixed harness vs co-evolving harness (paper Sec. 3.2 / Figure 4).

Runs the Cake arm twice per seed: once with the seed harness frozen, once with
harness evolution every ``evolve_every`` rounds.  Reports device-failure rates
(early vs late half of the run), the verifier rules adopted and rejected (with
their gate results), calibration updates and the best speedup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..agent.evolve import EvolveConfig, evolve
from ..evolution.state import HarnessState


def _rate(recs) -> float:
    dev = [r for r in recs if r.stage in ("runtime", "correctness", "timed")]
    return (sum(1 for r in dev if r.stage != "timed") / len(dev)) if dev else 0.0


def run(runs: int = 2, budget_tokens: int = 1_500_000, out_dir: str = "runs/coevolution", evolve_every: int = 6,
        llm: str = "scripted", workload: str = "kmeans_assign", verbose: bool = False) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for cond in ("fixed_harness", "co_evolving"):
        for i in range(runs):
            state = HarnessState(root=str(out / f"{cond}_run{i}" / "harness_state"))
            res = evolve(EvolveConfig(workload=workload, arm="cake", llm=llm, budget_tokens=budget_tokens,
                                      seed=200 + i, out_dir=str(out / f"{cond}_run{i}"), state=state,
                                      co_evolve=(cond == "co_evolving"), evolve_every=evolve_every,
                                      verbose=verbose))
            recs = res.archive.records[1:]
            mid = max(1, res.summary["rounds"] // 2)
            early = [r for r in recs if r.meta.get("round", 0) <= mid]
            late = [r for r in recs if r.meta.get("round", 0) > mid]
            adopted = list(state.library_enabled) + list(state.evolved)
            rejected = [{"rule": h["rule"], "corpus_false_positives": h["gate"]["corpus_false_positives"],
                         "recall": h["gate"]["recall"]} for h in state.history if h["kind"] == "verifier_rule_rejected"]
            row = {"condition": cond, "run": i, "best_speedup": res.summary["best_speedup"],
                   "device_failure_rate_early": round(_rate(early), 3),
                   "device_failure_rate_late": round(_rate(late), 3),
                   "device_failures": res.summary["device_failures"], "device_evals": res.summary["device_evals"],
                   "rules_adopted": adopted, "rules_rejected": rejected,
                   "calibration_version": state.calibrations["sm_100a"].version,
                   "tactics": len(state.tactics), "harness_version": state.version}
            rows.append(row)
            if state.root:
                state.save()
            print(f"[{cond} run {i}] best {row['best_speedup']:.3f}x, device failure rate "
                  f"{row['device_failure_rate_early']:.0%} -> {row['device_failure_rate_late']:.0%}, "
                  f"rules adopted {adopted}, rejected {[r['rule'] for r in rejected]}", flush=True)
    report = {"workload": workload, "llm": llm, "budget_tokens": budget_tokens, "runs": rows}
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = ["| Condition | Run | Best speedup | Device-failure rate (early -> late) | Rules adopted | Rules rejected "
          "(corpus false positives) | Calibration version |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rej = "; ".join(f"{x['rule']} ({', '.join(x['corpus_false_positives']) or 'recall ' + str(x['recall'])})"
                        for x in r["rules_rejected"]) or "-"
        md.append(f"| {r['condition']} | {r['run']} | {r['best_speedup']:.3f}x | {r['device_failure_rate_early']:.0%} "
                  f"-> {r['device_failure_rate_late']:.0%} | {', '.join(r['rules_adopted']) or '-'} | {rej} | "
                  f"v{r['calibration_version']} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
