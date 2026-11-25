"""Grid-level execution on the simulated GPU: launch checks, occupancy, waves and timing.

``run`` executes CTAs functionally (with data) for correctness checks;
``measure`` simulates representative CTAs in timing-only mode and composes
kernel time from occupancy, waves, unit contention between co-resident CTAs
and an HBM bandwidth floor, then reports the median of noisy samples (the
paper's protocol uses CUPTI medians with the L2 flushed before each sample).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import SimError
from ..lir.core import LKernel
from ..targets import Target
from .engine import CtaResult, CtaSim, Rates


def regs_per_thread(block) -> int:
    threads = 32 * max(1, len(block.warps))
    regs = 24
    for r in block.regs.values():
        regs += math.ceil(int(np.prod(r.shape)) / threads)
    return regs


def registers_per_cta(kernel: LKernel, target: Target) -> int:
    used = set()
    total = 0
    for b in kernel.blocks:
        total += min(regs_per_thread(b), target.max_regs_per_thread) * 32 * len(b.warps)
        used.update(b.warps)
    total += (kernel.warps - len(used)) * 32 * 24
    return total


def check_launch(kernel: LKernel, target: Target) -> None:
    if not 1 <= kernel.warps <= target.max_warps_per_cta:
        raise SimError("launch", f"CTA with {kernel.warps} warps (max {target.max_warps_per_cta})")
    if any(int(g) < 1 for g in kernel.grid):
        raise SimError("launch", f"invalid grid {kernel.grid}")
    owner: Dict[int, str] = {}
    for b in kernel.blocks:
        for w in b.warps:
            if not 0 <= w < kernel.warps:
                raise SimError("launch", f"block `{b.name}` uses warp {w} but the CTA has {kernel.warps} warps")
            if w in owner:
                raise SimError("launch", f"warp {w} is assigned to both `{owner[w]}` and `{b.name}`")
            owner[w] = b.name
    if kernel.smem_bytes > target.smem_per_cta:
        raise SimError("launch", f"{kernel.smem_bytes} bytes of shared memory requested, the limit on "
                       f"{target.name} is {target.smem_per_cta}")
    if kernel.tmem_cols:
        if not target.tmem:
            raise SimError("unsupported", f"tensor memory is not available on {target.name}")
        if kernel.tmem_cols not in (32, 64, 128, 256, 512):
            raise SimError("launch", f"TMEM allocation of {kernel.tmem_cols} columns: must be a power of two "
                           f"in [32, 512]")
    regs = registers_per_cta(kernel, target)
    if regs > target.regs_per_sm:
        raise SimError("launch", f"{regs} registers per CTA exceed the {target.regs_per_sm}-register file")


def occupancy(kernel: LKernel, target: Target) -> Tuple[int, str]:
    regs = max(1, registers_per_cta(kernel, target))
    limits = {
        "smem": target.smem_per_sm // max(1, kernel.smem_bytes + 1024),
        "threads": target.max_threads_per_sm // (32 * kernel.warps),
        "tmem": (target.tmem_cols // kernel.tmem_cols) if kernel.tmem_cols else 10 ** 9,
        "registers": target.regs_per_sm // regs,
        "ctas": target.max_ctas_per_sm,
    }
    limiter = min(limits, key=limits.get)
    return int(limits[limiter]), limiter


def all_ctas(kernel: LKernel) -> List[Tuple[int, int, int]]:
    gx, gy, gz = (int(g) for g in kernel.grid)
    return [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)]


def sample_ctas(kernel: LKernel, n: int = 3) -> List[Tuple[int, int, int]]:
    ctas = all_ctas(kernel)
    if len(ctas) <= n:
        return ctas
    picks = {0, len(ctas) - 1, 1}
    step = max(1, len(ctas) // max(1, n - 2))
    i = step
    while len(picks) < n and i < len(ctas):
        picks.add(i)
        i += step
    return [ctas[i] for i in sorted(picks)]


def _seed(kernel: LKernel, extra: int = 0) -> int:
    h = zlib.crc32((kernel.name + "\n" + (kernel.source or repr(kernel.meta.get("fingerprint")))).encode())
    return (h + extra) & 0x7FFFFFFF


@dataclass
class RunResult:
    ctas_run: int
    notes: List[Dict[str, Any]]
    cycles: List[float]


def run(kernel: LKernel, target: Target, gmem: Dict[str, np.ndarray], *,
        ctas: Optional[Sequence[Tuple[int, int, int]]] = None, seed: int = 0,
        scalars: Optional[Dict[str, Any]] = None) -> RunResult:
    """Functionally execute CTAs (all by default); outputs are written into ``gmem``."""
    check_launch(kernel, target)
    if occupancy(kernel, target)[0] < 1:
        raise SimError("launch", "kernel does not fit on an SM")
    coords = list(ctas) if ctas is not None else all_ctas(kernel)
    notes: List[Dict[str, Any]] = []
    seen = set()
    cycles = []
    for i, c in enumerate(coords):
        res = CtaSim(kernel, target, gmem, c, compute=True, seed=_seed(kernel, seed * 7919 + i),
                     scalars=scalars).run()
        cycles.append(res.cycles)
        for n in res.notes:
            key = (n["kind"], n["loc"].get("line"), n["role"])
            if key not in seen:
                seen.add(key)
                notes.append(n)
    return RunResult(ctas_run=len(coords), notes=notes, cycles=cycles)


@dataclass
class Timing:
    median_us: float
    samples_us: List[float]
    model_us: float
    cta_cycles: float
    kernel_cycles: float
    waves: int
    ctas_per_sm: int
    limiter: str
    num_ctas: int
    profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"median_us": self.median_us, "samples_us": self.samples_us, "model_us": self.model_us,
                "cta_cycles": self.cta_cycles, "kernel_cycles": self.kernel_cycles, "waves": self.waves,
                "ctas_per_sm": self.ctas_per_sm, "limiter": self.limiter, "num_ctas": self.num_ctas,
                "profile": self.profile}


def measure(kernel: LKernel, target: Target, *, samples: int = 5, seed: int = 0, noise: float = 0.01,
            scalars: Optional[Dict[str, Any]] = None) -> Timing:
    """Time a kernel on the simulated device (median of ``samples`` noisy measurements)."""
    check_launch(kernel, target)
    c, limiter = occupancy(kernel, target)
    if c < 1:
        raise SimError("launch", f"kernel does not fit on an SM (limited by {limiter})")
    n = kernel.num_ctas
    c_eff = max(1, min(c, math.ceil(n / target.num_sms)))   # co-resident CTAs per SM
    active_sms = min(target.num_sms, math.ceil(n / c_eff))
    clock = target.clock_ghz * 1e9
    # bandwidth available to one SM; co-resident CTAs share it through the SM's TMA/LSU queues
    hbm_bpc = target.hbm_gbps * 1e9 / clock / active_sms
    l2_bpc = target.l2_gbps * 1e9 / clock / active_sms
    first = (0, 0, 0)
    gx, gy, gz = (int(g) for g in kernel.grid)
    last = (gx - 1, gy - 1, gz - 1)
    cold = Rates(default_bpc=hbm_bpc / c_eff, default_latency=target.hbm_latency)
    probe = CtaSim(kernel, target, {}, first, compute=False, rates=cold, scalars=scalars).run()
    rates = Rates(default_bpc=hbm_bpc, default_latency=target.hbm_latency)
    hbm_bytes = 0.0
    l2_frac: Dict[str, float] = {}
    for p in kernel.params:
        if p.kind == "scalar" or not p.shape:
            continue
        tb = float(np.prod(p.shape)) * p.dtype.nbytes
        loaded = probe.bytes_loaded.get(p.name, 0) * float(n)
        reuse = loaded / tb if tb else 0.0
        f = (1.0 - 1.0 / reuse) if (reuse >= 1.5 and tb <= 0.6 * target.l2_bytes) else 0.0
        rates.bytes_per_clk[p.name] = 1.0 / (f / l2_bpc + (1.0 - f) / hbm_bpc)
        rates.latency[p.name] = f * target.l2_latency + (1.0 - f) * target.hbm_latency
        hbm_bytes += min(loaded, tb) if f > 0 else loaded
        l2_frac[p.name] = round(f, 3)
    # Simulate the c_eff co-resident CTAs of one SM jointly: they overlap their
    # latencies but contend for the tensor core, CUDA cores, TMEM and memory queues.
    order = all_ctas(kernel) if n <= 65536 else None

    def group(start: int) -> List[Tuple[int, int, int]]:
        if order is not None:
            return [order[(start + i) % n] for i in range(c_eff)]
        return [first if start == 0 else last] * c_eff

    groups = [group(0)]
    if n > c_eff:
        groups.append(group(n - c_eff))
    results: List[CtaResult] = []
    for g in groups:
        results.append(CtaSim(kernel, target, {}, g, compute=False, rates=rates, scalars=scalars).run())
    slow = max(results, key=lambda r: r.cycles)
    T = slow.cycles
    waves = math.ceil(n / (target.num_sms * c_eff))
    t_round = T
    cycles = waves * t_round
    floor = hbm_bytes / (target.hbm_gbps * 1e9 / clock)
    bw_bound = floor > cycles
    cycles = max(cycles, floor)
    model_us = cycles / (target.clock_ghz * 1e3) + target.launch_us
    rng = np.random.default_rng(_seed(kernel, seed))
    samp = [float(model_us * math.exp(rng.normal(0.0, noise))) for _ in range(max(1, samples))]
    util = {u: round(b / T, 3) if T > 0 else 0.0 for u, b in slow.units.items()}
    profile = {
        "cta_cycles": round(T, 1),
        "waves": waves,
        "ctas_per_sm": c,
        "occupancy_limiter": limiter,
        "num_ctas": n,
        "co_resident_ctas": c_eff,
        "hbm_bound": bw_bound,
        "unit_utilization": util,
        "l2_hit_fraction": l2_frac,
        "roles": {k: {"busy": {a: round(b, 1) for a, b in v["busy"].items()},
                      "waited": {a: round(b, 1) for a, b in v["waited"].items()},
                      "finish": round(v["finish"], 1),
                      "regs_per_thread": v["regs_per_thread"],
                      "warps": v["warps"]} for k, v in slow.roles.items()},
    }
    return Timing(median_us=float(np.median(samp)), samples_us=samp, model_us=model_us, cta_cycles=T,
                  kernel_cycles=cycles, waves=waves, ctas_per_sm=c, limiter=limiter, num_ctas=n,
                  profile=profile)


def summarize_profile(profile: Dict[str, Any]) -> str:
    """Generic (Nsight-like) profile text: unit utilization and per-role stall breakdown."""
    lines = [f"SM time per wave {profile['cta_cycles']:.0f} cycles, {profile['num_ctas']} CTAs, "
             f"{profile['co_resident_ctas']} co-resident CTA/SM (occupancy limit {profile['ctas_per_sm']}, "
             f"limited by {profile['occupancy_limiter']}), {profile['waves']} waves"]
    util = profile.get("unit_utilization", {})
    if util:
        lines.append("unit utilization: " + ", ".join(f"{u} {v:.0%}" for u, v in
                                                     sorted(util.items(), key=lambda x: -x[1])))
    for name, r in profile.get("roles", {}).items():
        busy = sum(r["busy"].values())
        waited = sum(r["waited"].values())
        top_wait = max(r["waited"].items(), key=lambda x: x[1])[0] if r["waited"] else "-"
        lines.append(f"  role {name} (warps {r['warps']}, {r['regs_per_thread']} regs/thread): "
                     f"busy {busy:.0f} cyc, stalled on barriers {waited:.0f} cyc (mostly `{top_wait}`)")
    return "\n".join(lines)
