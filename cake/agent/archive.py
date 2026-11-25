"""Run archive: every candidate, its evidence and the best-so-far trajectory.

"Retained results make decisions auditable and recurring findings reusable"
(paper Sec. 4): all records and routed evidence are appended to JSONL files.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..harness.evaluate import EvalRecord


def _signature(rec: EvalRecord) -> str:
    """Structural signature used to keep parents diverse."""
    sc = rec.meta.get("scripted")
    if sc:
        p = sc["params"]
        return "|".join(f"{k}={p[k]}" for k in ("SPLIT", "ACC", "PERSISTENT", "CN", "BN") if k in p)
    f = rec.features or {}
    roles = sorted((f.get("roles") or {}).keys())
    return f"{len(roles)}|{f.get('ctas_per_sm')}|{f.get('num_ctas')}"


class Archive:
    def __init__(self, out_dir: Optional[str] = None):
        self.records: List[EvalRecord] = []
        self.evidence: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.out = Path(out_dir) if out_dir else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            for name in ("records.jsonl", "evidence.jsonl"):
                (self.out / name).write_text("", encoding="utf-8")

    def add(self, rec: EvalRecord) -> None:
        self.records.append(rec)
        if self.out:
            with open(self.out / "records.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")

    def add_evidence(self, item: Dict[str, Any]) -> None:
        self.evidence.append(item)
        if self.out:
            with open(self.out / "evidence.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(item, default=str) + "\n")

    def ok(self) -> List[EvalRecord]:
        return [r for r in self.records if r.ok and r.speedup is not None]

    def best(self, k: int = 1) -> List[EvalRecord]:
        return sorted(self.ok(), key=lambda r: -r.speedup)[:k]

    def best_speedup(self) -> float:
        b = self.best(1)
        return b[0].speedup if b else 0.0

    def select_parents(self, k: int, rng: random.Random) -> List[EvalRecord]:
        """The best program plus the best of other structural families."""
        ranked = sorted(self.ok(), key=lambda r: -r.speedup)
        out, seen = [], set()
        for r in ranked:
            s = _signature(r)
            if s not in seen:
                out.append(r)
                seen.add(s)
            if len(out) >= k:
                break
        return out

    def recent_failures(self, k: int) -> List[EvalRecord]:
        fails = [r for r in self.records if not r.ok and r.stage != "filtered"]
        return fails[-k:][::-1]

    def checkpoint(self, tokens: int, wall_s: float, round_: int) -> None:
        cp = {"tokens": tokens, "best_speedup": self.best_speedup(), "wall_s": round(wall_s, 2), "round": round_,
              "candidates": len(self.records),
              "device_evals": sum(1 for r in self.records if r.stage in ("runtime", "correctness", "timed")),
              "device_failures": sum(1 for r in self.records if r.stage in ("runtime", "correctness"))}
        self.checkpoints.append(cp)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        if not self.out:
            return
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (self.out / "checkpoints.json").write_text(json.dumps(self.checkpoints, indent=2), encoding="utf-8")
        best = self.best(1)
        if best:
            (self.out / "best.py").write_text(best[0].source, encoding="utf-8")
