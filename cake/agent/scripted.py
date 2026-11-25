"""An offline, deterministic "kernel engineer" used when no LLM is available.

It is a *model of agent behaviour*, not a substitute for an LLM:

* it edits schedule knobs of a workload template, preferring edits suggested by
  the feedback it is shown (bottleneck attribution, hints, utilization);
* every edit carries a chance of introducing a realistic bug (missing release,
  desynchronized ring index, missing wait, lane-quadrant violation, ...); the
  direct-code arm has more hand-managed decisions per edit (addresses,
  parities, tx bytes, descriptors), so more chances to get one wrong;
* it repairs failures using only the feedback *text* -- localized Cake
  diagnostics usually identify the bug class, while coarse runtime errors leave
  it guessing among the classes consistent with the symptom.

Results produced with this backend demonstrate the pipeline mechanics; they are
not evidence about LLM behaviour.  Use an LLM backend for real experiments.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..harness import sandbox
from ..lir.core import LMbarExpectTx, LSet, LUmma, walk
from ..lir.printer import to_raw_source
from ..lower import lower
from .llm import LLM, LLMResponse
from .prompts import GenContext

CAKE_BUGS = ("no_release", "stage_mod", "missing_wait", "early_release", "no_commit_free", "smem_overlap",
             "lane_quadrant")
RAW_ONLY_BUGS = ("parity_init", "expect_tx", "swizzle_mismatch", "missing_init", "tmem_alloc", "addr_overlap")
BUG_TEXT = {
    "no_release": "declare the free-slot barrier with released=True",
    "stage_mod": "use the same ring index on producer and consumer",
    "missing_wait": "wait on the full barrier before the MMA",
    "early_release": "release the accumulator only after reading it",
    "no_commit_free": "commit the free barrier after each MMA",
    "smem_overlap": "place the C ring after the X tile",
    "lane_quadrant": "use 4 consecutive epilogue warps",
    "tail_mask": "mask ragged columns in the reduction",
    "parity_init": "start free-slot parities at 1",
    "expect_tx": "expect the exact transaction bytes",
    "swizzle_mismatch": "match descriptor swizzle to the TMA swizzle",
    "missing_init": "initialize every mbarrier",
    "tmem_alloc": "allocate enough TMEM columns",
    "addr_overlap": "keep shared-memory tiles disjoint",
}


def _templates(name: str):
    if name == "kmeans_assign":
        from ..workloads import kmeans_assign as km
        return km.template, km.KNOB_SPACE, km.BASELINE_PARAMS, km.normalize
    if name == "gemm":
        from ..workloads import gemm as gm
        space = {"BN": [64, 128, 256], "BK": [64], "STAGES": [1, 2, 3, 4, 5], "SPLIT": [False, True],
                 "SWIZZLE": ["128B", "64B", "none"]}
        return gm.template, space, dict(gm.DEFAULTS, SPLIT=False, STAGES=2), (lambda p: dict(gm.DEFAULTS, **p))
    raise KeyError(f"the scripted agent has no template for workload `{name}`")


# feedback text -> suggested knob edits (Cake arm sees attribution and hints)
_CAKE_ADVICE = [
    (r"serialized MMA \+ epilogue|HINT\.SPECIALIZE", [("SPLIT", True)]),
    (r"HINT\.ACC_BUFFER|double-buffer", [("ACC", 2)]),
    (r"latency bound|HINT\.SYNC_LOAD", [("CN", "smem")]),
    (r"only one CTA fits per SM", [("STAGES", "-1"), ("BN", 64)]),
    (r"spill", [("CHUNK", 64)]),
    (r"epilogue \(CUDA-core\) bound", [("EPI_WARPS", 8), ("FMA", True), ("BN", 64)]),
    (r"tensor-core bound", [("SWIZZLE", "128B"), ("BN", 256)]),
    (r"HINT\.PIPELINE", [("STAGES", "+1")]),
    (r"HINT\.SWIZZLE", [("SWIZZLE", "128B")]),
    (r"wave quantization", [("PERSISTENT", True)]),
]


class ScriptedLLM(LLM):
    name = "scripted"

    def __init__(self, seed: int = 0, p_bug: float = 0.22, p_guided: float = 0.65, raw_decisions: int = 3,
                 reasoning_tokens: Tuple[int, int] = (2500, 7000)):
        self.rng = random.Random(seed)
        self.p_bug = p_bug
        self.p_guided = p_guided
        self.raw_decisions = raw_decisions
        self.reasoning_tokens = reasoning_tokens
        self.last_meta: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ interface
    def generate(self, system: str, prompt: str, context: Optional[GenContext] = None) -> LLMResponse:
        if context is None:
            raise ValueError("ScriptedLLM needs the structured generation context")
        rng = self.rng
        tmpl, space, base, normalize = _templates(context.workload.name)
        parts, metas = [], []
        reasoning = 0
        for i in range(context.n):
            parent = self._pick_parent(context, i)
            params, bugs, idea = self._propose(context, parent, space, base, normalize)
            try:
                src = self._render(context, tmpl, params, bugs)
            except Exception as e:  # a template combination that does not construct is still a candidate
                src = tmpl(params, [b for b in bugs if b in CAKE_BUGS])
                idea += f" ({type(e).__name__})"
            parts.append(f"### Candidate {i + 1}: {idea}\n```python\n{src}```")
            metas.append({"params": params, "bugs": sorted(bugs), "parent": parent.id if parent else None})
            reasoning += rng.randint(*self.reasoning_tokens)
        self.last_meta = metas
        text = "\n\n".join(parts)
        return LLMResponse(text=text, input_tokens=int((len(system) + len(prompt)) / 3.6),
                           output_tokens=int(len(text) / 3.6) + reasoning, model="scripted", stop_reason="end_turn")

    # ------------------------------------------------------------------ policy
    def _pick_parent(self, ctx: GenContext, i: int):
        repairable = [f for f in ctx.failures if (f.meta.get("scripted") or {}).get("bugs")]
        if repairable and self.rng.random() < 0.45:
            return self.rng.choice(repairable[:3])
        if ctx.parents:
            return ctx.parents[0] if (i == 0 or len(ctx.parents) == 1) else self.rng.choice(ctx.parents)
        return None

    def _propose(self, ctx: GenContext, parent, space, base, normalize):
        rng = self.rng
        meta = (parent.meta.get("scripted") if parent is not None else None) or {"params": dict(base), "bugs": []}
        params = dict(meta["params"])
        bugs: Set[str] = set(meta["bugs"])
        fb = ctx.feedback.get(parent.id, "") if parent is not None else ""
        if parent is not None and not parent.ok and bugs:
            guess = self._diagnose(ctx.arm, fb)
            bugs.discard(guess)
            idea = f"fix: {BUG_TEXT.get(guess, guess)}"
            decisions = 1
        else:
            knob, val = self._mutation(ctx.arm, fb, params, space)
            params[knob] = val
            idea = f"set {knob}={val}"
            if rng.random() < 0.35:
                k2 = rng.choice(list(space))
                params[k2] = rng.choice(space[k2])
                idea += f", {k2}={params[k2]}"
            decisions = 1 if ctx.arm == "cake" else self.raw_decisions
        pool = CAKE_BUGS if ctx.arm == "cake" else CAKE_BUGS + RAW_ONLY_BUGS
        for _ in range(decisions):
            if rng.random() < self.p_bug:
                bugs.add(rng.choice(pool))
        return normalize(params), bugs, idea

    def _mutation(self, arm: str, fb: str, params: Dict[str, Any], space) -> Tuple[str, Any]:
        rng = self.rng
        options: List[Tuple[str, Any]] = []
        if rng.random() < self.p_guided:
            if arm == "cake":
                for pat, edits in _CAKE_ADVICE:
                    if re.search(pat, fb):
                        options.extend(edits)
            else:
                m = re.search(r"unit utilization: (\w+)", fb)
                top = m.group(1) if m else ""
                options = {"alu": [("EPI_WARPS", 8), ("FMA", True), ("SPLIT", True)],
                           "tc": [("SWIZZLE", "128B"), ("BN", 256)],
                           "tma": [("STAGES", "+1"), ("BN", 64)],
                           "tmem": [("CHUNK", 64)]}.get(top, [])
        options = [(k, v) for k, v in options if k in space]
        if options:
            k, v = rng.choice(options)
            if v in ("+1", "-1"):
                vals = space[k]
                cur = params.get(k, vals[0])
                idx = vals.index(cur) if cur in vals else 0
                idx = max(0, min(len(vals) - 1, idx + (1 if v == "+1" else -1)))
                v = vals[idx]
            if params.get(k) != v:
                return k, v
        k = rng.choice(list(space))
        choices = [x for x in space[k] if x != params.get(k)] or space[k]
        return k, rng.choice(choices)

    def _diagnose(self, arm: str, fb: str) -> str:
        """Pick the bug class to fix, using only the feedback text."""
        rng = self.rng
        if arm == "cake":
            if "SAFE.DEADLOCK" in fb:
                if "released=True" in fb:
                    return "no_release"
                if "finished" in fb:
                    return "no_commit_free"
                return rng.choice(["stage_mod", "no_commit_free"])
            if "SEM.INDEX_RANGE" in fb:
                return "stage_mod"
            if "SAFE.VIEW_BOUNDS" in fb:
                return "smem_overlap"
            if "SAFE.RACE" in fb:
                if re.search(r"overwrites `acc", fb):
                    return "early_release"
                if re.search(r"reads `sc\[", fb):
                    return "missing_wait"
                return rng.choice(["missing_wait", "early_release", "stage_mod"])
            if "TMEM_LANE" in fb or "lane quadrant" in fb:
                return "lane_quadrant"
            if "DC.TAIL_MASK" in fb:
                return "tail_mask"
            return rng.choice(["missing_wait", "early_release", "lane_quadrant", "stage_mod"])
        if "timed out" in fb:
            return rng.choice(["no_release", "no_commit_free", "stage_mod", "parity_init", "expect_tx"])
        if "illegal memory access" in fb:
            return rng.choice(["smem_overlap", "tmem_alloc", "addr_overlap"])
        if "illegal instruction" in fb:
            return rng.choice(["missing_init", "parity_init"])
        if "misaligned" in fb:
            return rng.choice(["addr_overlap", "smem_overlap"])
        return rng.choice(["missing_wait", "early_release", "lane_quadrant", "swizzle_mismatch", "stage_mod",
                           "addr_overlap", "smem_overlap"])

    # ------------------------------------------------------------------ rendering
    def _render(self, ctx: GenContext, tmpl, params, bugs: Set[str]) -> str:
        cake_bugs = [b for b in bugs if b in CAKE_BUGS]
        src = tmpl(params, cake_bugs)
        if ctx.arm == "cake":
            return src
        w = ctx.workload
        prog = sandbox.build(src, w, w.target_dims, w.target)
        k = lower(prog)
        _inject_raw_bugs(k, bugs, self.rng)
        return to_raw_source(k, prog.name)


def _inject_raw_bugs(k, bugs: Set[str], rng: random.Random) -> None:
    """Hand-managed mistakes of the direct-code arm, applied to the lowered kernel."""
    if "missing_init" in bugs and len(k.prologue) > 1:
        del k.prologue[rng.randrange(len(k.prologue))]
    tx_done = False
    for b in k.blocks:
        for op in walk(b.body):
            if ("parity_init" in bugs and isinstance(op, LSet) and op.name.startswith("ph_")
                    and isinstance(op.value, int) and op.value != 0):
                op.value = 0
            if "expect_tx" in bugs and isinstance(op, LMbarExpectTx) and not tx_done:
                op.nbytes = op.nbytes + 1024
                tx_done = True
            if "swizzle_mismatch" in bugs and isinstance(op, LUmma) and op.b.swizzle == "128B":
                op.b.swizzle = "64B"
    if "tmem_alloc" in bugs and k.tmem_cols > 32:
        k.tmem_cols //= 2
    if "addr_overlap" in bugs:
        for b in k.blocks:
            for op in walk(b.body):
                if op.kind == "tma_load" and op.tensor == "C":
                    op.dst = op.dst - 1024
