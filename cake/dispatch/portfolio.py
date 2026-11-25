"""From a tuned shape to a library: the generalization and dispatch stage (paper Sec. 6).

* Separate objective: seeds come from single-shape evolution; this stage scores
  dispatcher-inclusive performance over a fixed shape domain.
* The valid shape domain is declared (and hashed) before tuning; a deterministic
  split keeps held-out shards that tuning never sees, so routes cannot be
  tuned on the rows used to claim generalization.
* Seeds are bucketed by shape features; a bucket gets its own route only when
  it beats the default route by a measured margin ("routing complexity must be
  justified by measured workload gain"); guards are ordered in front of an
  explicit fallback.
* Validation covers held-out shapes, boundary/tail shapes, guard overlap and
  coverage, and the fallback path; incorrect seeds are excluded (they go back to
  the inner loop instead of being hidden behind routing).
* Gspan is the unweighted geometric mean of per-shape speedups over the
  baseline, with a per-call dispatcher overhead included.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import CakeError
from ..harness import sandbox
from ..lower import lower
from ..sim import device
from ..targets import get_target
from ..verify import verify
from ..workloads import get_workload

Shape = Dict[str, int]


def kmeans_domain() -> List[Shape]:
    """Declared shape domain for the KMeans-assign family (includes ragged tails)."""
    out = []
    for D in (64, 128, 256):
        for K in (256, 1000, 1024, 4096):
            for N, B in ((4000, 4), (16384, 8), (65536, 8)):
                out.append({"B": B, "N": N, "K": K, "D": D})
    return out


def domain_hash(shapes: Sequence[Shape]) -> str:
    return hashlib.sha1(json.dumps(sorted(json.dumps(s, sort_keys=True) for s in shapes)).encode()).hexdigest()[:12]


def split(shapes: Sequence[Shape], holdout: float = 0.3) -> Tuple[List[Shape], List[Shape]]:
    """Deterministic split by shape hash: held-out shards are fixed before any tuning."""
    tune, held = [], []
    for s in shapes:
        h = zlib.crc32(json.dumps(s, sort_keys=True).encode()) % 1000 / 1000.0
        (held if h < holdout else tune).append(s)
    return tune, held


def bucket(s: Shape) -> str:
    kcls = "K<=512" if s["K"] <= 512 else ("K<=2048" if s["K"] <= 2048 else "K>2048")
    aligned = "aligned" if s["K"] % 128 == 0 else "ragged"
    return f"D{s['D']}|{kcls}|{aligned}"


@dataclass
class Route:
    name: str
    source: str
    buckets: List[str] = field(default_factory=list)   # empty = default route (any applicable shape)

    def matches(self, s: Shape) -> bool:
        return not self.buckets or bucket(s) in self.buckets


@dataclass
class ShapeResult:
    shape: Shape
    route: str
    ok: bool
    us: Optional[float]
    baseline_us: Optional[float]
    speedup: Optional[float]
    note: str = ""


class Evaluator:
    """Applicability (construct + verify), correctness (reduced shape) and timing for (source, shape)."""

    def __init__(self, workload: str = "kmeans_assign", target: Optional[str] = None, samples: int = 3):
        self.w = get_workload(workload)
        self.t = get_target(target or self.w.target)
        self.samples = samples
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _reduced(self, s: Shape) -> Shape:
        r = dict(s)
        r["B"] = 1
        r["N"] = min(s["N"], 256 + s["N"] % 128) or s["N"]
        return r

    def run(self, source: str, s: Shape) -> Dict[str, Any]:
        key = (hashlib.sha1(source.encode()).hexdigest()[:12], json.dumps(s, sort_keys=True))
        if key in self._cache:
            return self._cache[key]
        res: Dict[str, Any] = {"applicable": False, "correct": False, "us": None, "why": ""}
        try:
            prog = sandbox.build(source, self.w, s, self.t.name)
            rep = verify(prog, self.t)
            if rep.blocked:
                res["why"] = "verifier: " + ", ".join(sorted({f.rule for f in rep.gates}))
                self._cache[key] = res
                return res
            res["applicable"] = True
            small = self._reduced(s)
            k_small = lower(sandbox.build(source, self.w, small, self.t.name))
            inp = self.w.make_inputs(small, "gaussian", 11)
            g = {**inp, **self.w.make_outputs(small)}
            device.run(k_small, self.t, g)
            outs = {x.name: g[x.name] for x in self.w.tensors if x.role == "output"}
            cr = self.w.check(outs, self.w.oracle(inp, small), inp, small, True)
            res["correct"] = cr.ok
            if not cr.ok:
                res["why"] = "incorrect: " + cr.summary
            else:
                res["us"] = device.measure(lower(prog), self.t, samples=self.samples).median_us
        except CakeError as e:
            res["why"] = f"{e.kind}: {e.message[:120]}"
        self._cache[key] = res
        return res


@dataclass
class Portfolio:
    routes: List[Route]
    fallback: Route
    domain: str
    dispatch_overhead_us: float = 1.0

    def select(self, s: Shape) -> Route:
        for r in self.routes:
            if r.matches(s):
                return r
        return self.fallback

    def table(self) -> List[Dict[str, Any]]:
        return [{"route": r.name, "guard": r.buckets or ["(default: any applicable shape)"]} for r in self.routes] + \
               [{"route": self.fallback.name, "guard": ["(fallback)"]}]


def gspan(speedups: Sequence[float]) -> float:
    xs = [x for x in speedups if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def build_portfolio(seeds: Sequence[Tuple[str, str]], shapes: Sequence[Shape], ev: Evaluator,
                    min_gain: float = 0.03, min_support: int = 2, overhead_us: float = 1.0,
                    verbose: bool = True) -> Tuple[Portfolio, Dict]:
    """Build routes from seeds on the tuning shapes only.

    A bucket earns its own route only if it has at least ``min_support`` tuning
    shapes and the route beats the default route there by ``min_gain``.
    """
    base_src = ev.w.baseline_source
    perf: Dict[str, Dict[str, float]] = {}       # seed -> shape key -> speedup
    excluded: Dict[str, List[str]] = {}
    for name, src in seeds:
        perf[name] = {}
        for s in shapes:
            r, b = ev.run(src, s), ev.run(base_src, s)
            key = json.dumps(s, sort_keys=True)
            if r["correct"] and r["us"] and b["us"]:
                perf[name][key] = b["us"] / (r["us"] + overhead_us)
            elif r["applicable"] and not r["correct"]:
                excluded.setdefault(name, []).append(f"{key}: {r['why']}")
        if verbose:
            print(f"  [dispatch] seed {name}: correct on {len(perf[name])}/{len(shapes)} tuning shapes, "
                  f"gspan {gspan(list(perf[name].values())):.3f}", flush=True)
    # default route: best geomean weighted by coverage
    def score(n):
        v = list(perf[n].values())
        return (len(v), gspan(v)) if v else (0, 0.0)
    default = max(perf, key=score)
    routes: List[Route] = []
    by_bucket: Dict[str, List[str]] = {}
    for s in shapes:
        by_bucket.setdefault(bucket(s), []).append(json.dumps(s, sort_keys=True))
    winners: Dict[str, str] = {}
    thin: List[str] = []
    for bk, keys in by_bucket.items():
        if len(keys) < min_support:
            thin.append(bk)   # too little evidence to justify a route: the default handles it
            continue
        best, best_g = None, 0.0
        dflt = [perf[default].get(k) for k in keys]
        dflt_g = gspan([x for x in dflt if x]) if all(dflt) else 0.0
        for n in perf:
            vals = [perf[n].get(k) for k in keys]
            if all(vals):
                g = gspan(vals)
                if g > best_g:
                    best, best_g = n, g
        if best is not None and best != default and (dflt_g == 0.0 or best_g > dflt_g * (1 + min_gain)):
            winners[bk] = best
    for name in sorted(set(winners.values())):
        routes.append(Route(name, dict(seeds)[name], sorted(b for b, n in winners.items() if n == name)))
    routes.append(Route(default, dict(seeds)[default], []))
    pf = Portfolio(routes, Route("baseline_fallback", base_src, []), "", overhead_us)
    return pf, {"default": default, "bucket_winners": winners, "excluded": excluded,
                "buckets_below_support": thin}


def validate(pf: Portfolio, shapes: Sequence[Shape], ev: Evaluator) -> Dict[str, Any]:
    """Evaluate the dispatcher on a shape set; incorrect outputs are counted, never hidden."""
    results: List[ShapeResult] = []
    for s in shapes:
        route = pf.select(s)
        r = ev.run(route.source, s)
        note = ""
        if not r["applicable"] or not r["correct"]:
            # the guard admitted a shape the route cannot handle: fall back explicitly and report it
            note = f"route {route.name} rejected ({r['why']}); fallback used"
            route = pf.fallback
            r = ev.run(route.source, s)
        b = ev.run(pf.fallback.source, s)
        us = (r["us"] + pf.dispatch_overhead_us) if r["us"] else None
        sp = (b["us"] / us) if (us and b["us"]) else None
        results.append(ShapeResult(s, route.name, bool(r["correct"]), us, b["us"], sp, note))
    overlaps = sum(1 for s in shapes if sum(1 for r in pf.routes[:-1] if r.buckets and r.matches(s)) > 1)
    return {
        "shapes": len(shapes),
        "incorrect": sum(1 for x in results if not x.ok),
        "fallback_used": sum(1 for x in results if x.route == pf.fallback.name),
        "guard_rejections": sum(1 for x in results if x.note),
        "guard_overlaps": overlaps,
        "gspan": gspan([x.speedup for x in results if x.speedup]),
        "per_route": _per_route(results),
        "results": [x.__dict__ for x in results],
    }


def _per_route(results: List[ShapeResult]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in results:
        d = out.setdefault(x.route, {"shapes": 0, "speedups": []})
        d["shapes"] += 1
        if x.speedup:
            d["speedups"].append(x.speedup)
    return {k: {"shapes": v["shapes"], "gspan": gspan(v["speedups"])} for k, v in out.items()}


def run_dispatch(seeds: Sequence[Tuple[str, str]], workload: str = "kmeans_assign",
                 domain: Optional[Sequence[Shape]] = None, holdout: float = 0.3, verbose: bool = True) -> Dict:
    t0 = time.time()
    domain = list(domain or kmeans_domain())
    manifest = {"domain_hash": domain_hash(domain), "shapes": len(domain), "holdout": holdout}
    tune, held = split(domain, holdout)
    ev = Evaluator(workload)
    pf, info = build_portfolio(seeds, tune, ev, verbose=verbose)
    pf.domain = manifest["domain_hash"]
    tuned = validate(pf, tune, ev)
    heldout = validate(pf, held, ev)
    fb = validate(Portfolio([], pf.fallback, pf.domain, pf.dispatch_overhead_us), held[: max(1, len(held) // 3)], ev)
    report = {"manifest": manifest, "routes": pf.table(), "selection": info,
              "tuning": {k: v for k, v in tuned.items() if k != "results"},
              "held_out": {k: v for k, v in heldout.items() if k != "results"},
              "fallback_check": {k: v for k, v in fb.items() if k in ("shapes", "incorrect", "gspan")},
              "held_out_results": heldout["results"], "wall_s": round(time.time() - t0, 1)}
    return report
