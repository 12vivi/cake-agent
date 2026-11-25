"""Workload contracts.

"The external workload contract is the stable authority ... [it] fixes the
shapes, oracle, tolerances, hardware, and permitted references" (paper Sec. 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..dtypes import DType


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dims: Tuple[str, ...]
    dtype: DType
    role: str  # "input" | "output"


@dataclass(frozen=True)
class Case:
    """One validation case: a shape, an input distribution and a CTA coverage policy."""
    name: str
    dims: Tuple[Tuple[str, int], ...]
    dist: str = "gaussian"
    seed: int = 0
    sample_ctas: int = 0  # 0 = run every CTA (full coverage); n = run n sampled CTAs

    @property
    def dim_dict(self) -> Dict[str, int]:
        return dict(self.dims)


@dataclass
class CheckResult:
    ok: bool
    summary: str
    rows_checked: int = 0
    rows_bad: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "summary": self.summary, "rows_checked": self.rows_checked,
                "rows_bad": self.rows_bad, "examples": self.examples[:5]}


class Workload:
    """Base class; subclasses define tensors, oracle, inputs and checks."""

    name: str = ""
    title: str = ""
    description: str = ""
    tensors: List[TensorSpec] = []
    target_dims: Dict[str, int] = {}
    target: str = "sm_100a"
    cases: List[Case] = []
    tolerance: str = ""
    permitted_references: str = ""
    baseline_name: str = "baseline"
    baseline_source: str = ""

    def tensor_specs(self, dims: Dict[str, int]) -> Dict[str, Tuple[Tuple[int, ...], DType]]:
        return {t.name: (tuple(int(dims[d]) for d in t.dims), t.dtype) for t in self.tensors}

    def make_inputs(self, dims: Dict[str, int], dist: str, seed: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def make_outputs(self, dims: Dict[str, int]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def oracle(self, inputs: Dict[str, np.ndarray], dims: Dict[str, int]) -> Dict[str, Any]:
        raise NotImplementedError

    def check(self, outputs: Dict[str, np.ndarray], expected: Dict[str, Any], inputs: Dict[str, np.ndarray],
              dims: Dict[str, int], full_coverage: bool) -> CheckResult:
        raise NotImplementedError

    def contract_text(self) -> str:
        lines = [f"Workload `{self.name}`: {self.title}", "", self.description.strip(), "", "Tensors:"]
        for t in self.tensors:
            lines.append(f"  - {t.name}: {t.role}, dtype {t.dtype.name}, shape ({', '.join(t.dims)})")
        lines.append("")
        lines.append("Target shape (timed): " + ", ".join(f"{k}={v}" for k, v in self.target_dims.items()))
        lines.append(f"Target GPU: {self.target}")
        lines.append("Validation: " + "; ".join(
            f"{c.name} [{', '.join(f'{k}={v}' for k, v in c.dims)}; {c.dist}"
            f"{'; sampled CTAs' if c.sample_ctas else ''}]" for c in self.cases))
        lines.append(f"Tolerance: {self.tolerance}")
        lines.append(f"Permitted references: {self.permitted_references}")
        return "\n".join(lines)


_REGISTRY: Dict[str, Callable[[], Workload]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_workload(name: str) -> Workload:
    from . import kmeans_assign, gemm  # noqa: F401  (registration side effects)
    if name not in _REGISTRY:
        raise KeyError(f"unknown workload `{name}`; available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]()


def list_workloads() -> List[str]:
    from . import kmeans_assign, gemm  # noqa: F401
    return sorted(_REGISTRY)
