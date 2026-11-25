"""Findings and reports returned by the harness (paper Table 1).

A finding names the violated contract (``rule``), its category, its
disposition (blocking gate / performance report / non-blocking hint), the
program region (source line) and the affected resource, role or stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import Loc

CATEGORIES = {
    "program_safety": ("gate", "Identify synchronization, ordering, and memory-use hazards"),
    "hardware_conformance": ("gate", "Enforce supported resource, instruction, and architecture contracts"),
    "data_consistency": ("gate", "Check data flow and producer-consumer representation compatibility"),
    "schedule_semantics": ("gate", "Check structural invariants of the declared schedule"),
    "numerical_validation": ("gate", "Compare compiled outputs with an authoritative external reference"),
    "performance_analysis": ("report", "Estimate cost and identify broad bottleneck classes"),
    "optimization_guidance": ("hint", "Suggest promising revisions without blocking compilation"),
}


@dataclass
class Finding:
    rule: str
    category: str
    disposition: str          # "gate" | "report" | "hint"
    message: str
    loc: Optional[Loc] = None
    subject: Dict[str, Any] = field(default_factory=dict)
    hint: Optional[str] = None

    @property
    def blocking(self) -> bool:
        return self.disposition == "gate"

    def format(self) -> str:
        tag = {"gate": "ERROR", "report": "REPORT", "hint": "HINT"}[self.disposition]
        s = f"{tag} [{self.rule}] ({self.category}) {self.message}"
        if self.loc is not None and self.loc.lineno:
            s += f"\n    at {self.loc}"
        if self.subject:
            s += "\n    subject: " + ", ".join(f"{k}={v}" for k, v in self.subject.items())
        if self.hint:
            s += f"\n    hint: {self.hint}"
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "category": self.category, "disposition": self.disposition,
                "message": self.message, "loc": self.loc.to_dict() if self.loc else None,
                "subject": self.subject, "hint": self.hint}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Finding":
        loc = d.get("loc")
        return Finding(d["rule"], d["category"], d["disposition"], d["message"],
                       Loc(loc["line"], loc["text"]) if loc else None, d.get("subject") or {}, d.get("hint"))


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    coverage: List[str] = field(default_factory=list)   # explicit analysis-coverage limitations
    rules_run: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.blocking for f in self.findings)

    @property
    def gates(self) -> List[Finding]:
        return [f for f in self.findings if f.disposition == "gate"]

    def by_disposition(self, d: str) -> List[Finding]:
        return [f for f in self.findings if f.disposition == d]

    def format(self, max_items: int = 12) -> str:
        if not self.findings and not self.coverage:
            return "verifier: no findings"
        order = {"gate": 0, "report": 1, "hint": 2}
        fs = sorted(self.findings, key=lambda f: order[f.disposition])
        lines = [f.format() for f in fs[:max_items]]
        if len(fs) > max_items:
            lines.append(f"... {len(fs) - max_items} more findings")
        for c in self.coverage:
            lines.append(f"COVERAGE: {c}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"findings": [f.to_dict() for f in self.findings], "coverage": self.coverage,
                "blocked": self.blocked, "rules_run": self.rules_run}
