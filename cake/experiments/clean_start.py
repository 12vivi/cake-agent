"""Matched clean-start comparison: Cake IR vs direct CUDA/PTX-level code (paper Table 2 / Figure 5).

Both arms share the workload contract, oracle, benchmark harness, target shape,
token budget, run seeds and agent backend; only the authored representation and
the feedback it affords differ.  Reported per arm: plateau by budget (count),
active evolve time and best-at-budget (median [min, max]), plus the device
failure rate; the trajectory is the best validated speedup so far at fixed
token checkpoints (mean with min-max band).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..agent.evolve import EvolveConfig, evolve
from .plots import line_band_svg

ARM_NAMES = {"cake": "Cake IR", "raw": "Direct CUDA/PTX"}


def _best_at(checkpoints: List[Dict[str, Any]], tokens: float) -> float:
    best = 1.0
    for c in checkpoints:
        if c["tokens"] <= tokens:
            best = max(best, c["best_speedup"])
    return best


def _mmm(xs: Sequence[float]) -> str:
    return f"{statistics.median(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]"


def run(workload: str = "kmeans_assign", arms: Sequence[str] = ("cake", "raw"), runs: int = 3,
        budget_tokens: int = 2_000_000, llm: str = "scripted", out_dir: str = "runs/clean_start",
        candidates_per_round: int = 4, device_evals_per_round: int = 2, verbose: bool = False) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    for arm in arms:
        for i in range(runs):
            res = evolve(EvolveConfig(workload=workload, arm=arm, llm=llm, budget_tokens=budget_tokens, seed=100 + i,
                                      out_dir=str(out / f"{arm}_run{i}"), candidates_per_round=candidates_per_round,
                                      device_evals_per_round=device_evals_per_round, verbose=verbose))
            s = res.summary
            print(f"[{arm} run {i}] best {s['best_speedup']:.3f}x, {s['device_failures']}/{s['device_evals']} device "
                  f"failures, plateau at {s['plateau_tokens']}, {s['active_evolve_s']} s", flush=True)
            results[arm].append({"summary": s, "checkpoints": res.archive.checkpoints})
    grid = [budget_tokens * k / 16 for k in range(2, 17)]
    table, traj = {}, {}
    for arm, rs in results.items():
        best = [r["summary"]["best_speedup"] for r in rs]
        act = [r["summary"]["active_evolve_s"] / 3600.0 for r in rs]
        plateau = sum(1 for r in rs if r["summary"]["plateau_tokens"] is not None)
        dev = sum(r["summary"]["device_evals"] for r in rs)
        fail = sum(r["summary"]["device_failures"] for r in rs)
        rej = sum(r["summary"]["verifier_rejects"] for r in rs)
        cands = sum(r["summary"]["candidates"] for r in rs)
        table[arm] = {"plateau": f"{plateau}/{len(rs)}", "active_evolve_h": _mmm(act), "best_at_budget": _mmm(best),
                      "device_failure_rate": round(fail / dev, 3) if dev else None,
                      "verifier_rejects": rej, "candidates": cands}
        curves = [[_best_at(r["checkpoints"], g) for g in grid] for r in rs]
        traj[arm] = {"tokens": grid, "mean": [statistics.mean(c[j] for c in curves) for j in range(len(grid))],
                     "min": [min(c[j] for c in curves) for j in range(len(grid))],
                     "max": [max(c[j] for c in curves) for j in range(len(grid))]}
    report = {"workload": workload, "llm": llm, "budget_tokens": budget_tokens, "runs": runs, "table": table,
              "trajectory": traj, "runs_detail": {a: [r["summary"] for r in rs] for a, rs in results.items()}}
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = ["| Representation | Plateau by budget | Active evolve (h) | Best at budget | Device-failure rate |",
          "|---|---|---|---|---|"]
    for arm in arms:
        t = table[arm]
        md.append(f"| {ARM_NAMES.get(arm, arm)} | {t['plateau']} | {t['active_evolve_h']} | {t['best_at_budget']} | "
                  f"{t['device_failure_rate']:.0%} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    render_figure(report, str(out / "figure.svg"))
    return report


def render_figure(report: Dict[str, Any], path: str) -> None:
    """Figure-5-style trajectory: mean best validated speedup per arm with a min-max band."""
    traj = report["trajectory"]
    series = [{"name": ARM_NAMES.get(a, a), "x": [g / 1e6 for g in traj[a]["tokens"]], "y": traj[a]["mean"],
               "lo": traj[a]["min"], "hi": traj[a]["max"]} for a in traj]
    line_band_svg(series, path,
                  title=f"Clean-start attainment ({report['workload']}, simulated B200)",
                  subtitle=f"{report['runs']} runs per representation, {report['llm']} agent; "
                           f"line: mean best speedup so far, band: min-max",
                  xlabel="Cumulative tokens (millions)", ylabel="Speedup over baseline (x)",
                  reference=1.0, reference_label="baseline", xfmt=lambda x: f"{x:g}M",
                  yfmt=lambda y: f"{y:.2f}x")
