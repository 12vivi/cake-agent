"""The Cake verifier: pre-compile gates, performance reports and hints (paper Table 1)."""
from __future__ import annotations

from typing import Iterable, List, Optional

from ..ir.core import Program
from ..targets import get_target
from .facts import Facts, analyze, representative_ctas  # noqa: F401
from .findings import CATEGORIES, Finding, Report  # noqa: F401
from .rules import LIBRARY, SEED, Ctx, Rule, seed_rules  # noqa: F401


def verify(program: Program, target=None, rules: Optional[Iterable[Rule]] = None,
           extra_rules: Iterable[Rule] = (), max_ops: int = 400_000) -> Report:
    """Run the verifier.  ``rules`` defaults to the seed rules; ``extra_rules`` adds evolved ones."""
    tgt = get_target(target or program.target)
    ctx = Ctx(program, tgt, max_ops)
    active: List[Rule] = list(rules) if rules is not None else seed_rules()
    active += [r for r in extra_rules if r.id not in {a.id for a in active}]
    report = Report()
    seen = set()
    for r in active:
        ctx._rule = r
        try:
            found = r.run(ctx)
        except Exception as e:  # a broken (e.g. freshly evolved) rule must not take the harness down
            report.coverage.append(f"rule {r.id} raised {type(e).__name__}: {e}")
            continue
        report.rules_run.append(r.id)
        for f in found:
            key = (f.rule, f.loc.lineno if f.loc else 0, f.message)
            if key not in seen:
                seen.add(key)
                report.findings.append(f)
    if "facts" in ctx.__dict__:
        ctas = [fs.block_idx for fs in ctx.facts]
        report.coverage.append(f"abstract execution covered CTA(s) {ctas} of grid {program.grid}; "
                               f"other CTAs are assumed to follow the same synchronization pattern")
        for fs in ctx.facts:
            if fs.truncated:
                report.coverage.append(f"analysis of CTA {fs.block_idx} stopped after {fs.ops_executed} "
                                       f"operations; later iterations were not checked")
    return report
