"""Agent-visible analytical cost model with bottleneck attribution (paper Sec. 3.1).

The model works on Cake IR *before* code generation: it counts the work every
role issues (tensor-core FLOPs, CUDA-core lane operations, TMEM and TMA bytes,
synchronous global loads, waits), composes a per-CTA critical path from the
declared roles, adds occupancy and wave effects, and multiplies each term by a
per-target calibration coefficient.  It is deliberately simpler than the
simulated device, so it mispredicts systematically until calibrated -- which is
what makes "a systematic misprediction becomes a calibration target" (Sec. 3.2)
meaningful.  It declines to predict on targets without calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import CakeError
from ..ir.core import Commit, Convert, Ewise, Fill, LoadG, Mma, Program, Reduce, TmaLoad, TmemLoad, TmemRef, Wait
from ..sim.engine import ALU_WORK, alu_efficiency
from ..sim.memory import MMA_SWIZZLE_FACTOR
from ..targets import Target, get_target
from ..verify.facts import _gen

TERMS = ("tc", "alu", "tmem", "tma", "latency", "sync", "fill", "launch")


@dataclass
class Calibration:
    target: str
    coef: Dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TERMS})
    version: int = 0
    samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "coef": self.coef, "version": self.version, "samples": self.samples,
                "metrics": self.metrics}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Calibration":
        return Calibration(d["target"], dict(d["coef"]), d.get("version", 0), d.get("samples", 0),
                           dict(d.get("metrics", {})))


def default_calibrations() -> Dict[str, Calibration]:
    """Seed calibrations: only targets with timing calibration get one (B200, H100)."""
    from ..targets import TARGETS
    return {name: Calibration(name) for name, t in TARGETS.items() if t.timing_calibrated}


@dataclass
class Estimate:
    us: Optional[float]
    covered: bool
    reason: str = ""
    raw: Dict[str, float] = field(default_factory=dict)       # per-CTA term cycles before calibration
    terms: Dict[str, float] = field(default_factory=dict)     # after calibration
    cta_cycles: float = 0.0
    waves: int = 0
    ctas_per_sm: int = 0
    limiter: str = ""
    bottleneck: str = ""
    attribution: str = ""
    hints: List[str] = field(default_factory=list)
    roles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    features: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"us": self.us, "covered": self.covered, "reason": self.reason, "raw": self.raw, "terms": self.terms,
                "cta_cycles": self.cta_cycles, "waves": self.waves, "ctas_per_sm": self.ctas_per_sm,
                "limiter": self.limiter, "bottleneck": self.bottleneck, "attribution": self.attribution,
                "hints": self.hints, "roles": self.roles}

    def format(self) -> str:
        if not self.covered:
            return f"cost model: no estimate ({self.reason})"
        s = (f"cost model: ~{self.us:.1f} us predicted; bottleneck: {self.bottleneck}. {self.attribution}\n"
             f"  per-CTA terms (cycles): " + ", ".join(f"{k} {v:.0f}" for k, v in self.terms.items() if v) +
             f"\n  occupancy: {self.ctas_per_sm} CTA/SM (limited by {self.limiter}), {self.waves} waves")
        for h in self.hints:
            s += f"\n  hint: {h}"
        return s


def _regs_per_thread(prog: Program, role) -> int:
    regs = 24
    threads = 32 * len(role.warps)
    for rt in prog.regs:
        if rt.role is role:
            n = 1
            for d in rt.shape:
                n *= d
            regs += -(-n // threads)
    return regs


def _occupancy(prog: Program, t: Target):
    smem = sum(p.nbytes for p in prog.pools) + 8 * sum(b.count for b in prog.barriers) + 128
    tmem_end = max((tm.col + tm.stages * tm.width for tm in prog.tmems), default=0)
    tmem = 0 if tmem_end == 0 else max(32, 1 << max(0, (tmem_end - 1).bit_length()))
    regs = sum(min(_regs_per_thread(prog, r), t.max_regs_per_thread) * 32 * len(r.warps) for r in prog.roles)
    limits = {
        "smem": t.smem_per_sm // max(1, smem + 1024),
        "threads": t.max_threads_per_sm // (32 * prog.warps),
        "tmem": (t.tmem_cols // tmem) if (tmem and t.tmem_cols) else 10 ** 9,
        "registers": t.regs_per_sm // max(1, regs),
        "ctas": t.max_ctas_per_sm,
    }
    lim = min(limits, key=limits.get)
    return max(0, int(limits[lim])), lim


def count_work(prog: Program, t: Target, block_idx=(0, 0, 0), max_ops: int = 200_000) -> Dict[str, Any]:
    """Expand one CTA and count the work each role issues."""
    gx, gy, gz = prog.grid
    env0 = {"blockIdx.x": block_idx[0], "blockIdx.y": block_idx[1], "blockIdx.z": block_idx[2],
            "gridDim.x": gx, "gridDim.y": gy, "gridDim.z": gz}
    env0.update(prog.meta.get("scalars", {}))
    roles: Dict[str, Dict[str, float]] = {}
    tma_bytes: Dict[str, float] = {}
    serial_mma = set()
    for r in prog.roles:
        env = dict(env0)
        acc = {"tc": 0.0, "alu_work": 0.0, "alu": 0.0, "tmem_bytes": 0.0, "gload": 0.0, "gload_bytes": 0.0,
               "waits": 0.0, "tma_issue": 0.0, "mma_sync": 0.0, "regs": float(_regs_per_thread(prog, r)),
               "warps": float(len(r.warps))}
        eff = alu_efficiency(len(r.warps))
        committed = set()
        waited = set()
        n = 0
        for op in _gen(prog.body[r], env):
            n += 1
            if n > max_ops:
                break
            if isinstance(op, Mma):
                m, nn, k = op.mnk
                factor = max(MMA_SWIZZLE_FACTOR[op.a.view.swizzle], MMA_SWIZZLE_FACTOR[op.b.view.swizzle])
                if isinstance(op.acc, TmemRef):
                    acc["tc"] += 2.0 * 128 * nn * k / t.tc_rate(op.a.dtype.name) * factor
                else:
                    legacy = 4.0 if t.tcgen05 else (1.25 if t.wgmma else 1.0)
                    c = 2.0 * m * nn * k / t.tc_rate(op.a.dtype.name) * factor * legacy
                    acc["tc"] += c
                    acc["mma_sync"] += c
            elif isinstance(op, TmaLoad):
                tma_bytes[op.src.name] = tma_bytes.get(op.src.name, 0.0) + op.nbytes
                acc["tma_issue"] += 1
            elif isinstance(op, (Ewise, Fill, Convert)):
                w = op.dst.shape[0] * (op.dst.shape[1] if len(op.dst.shape) > 1 else 1)
                kind = op.op if isinstance(op, Ewise) else ("fill" if isinstance(op, Fill) else "convert")
                acc["alu_work"] += w * ALU_WORK.get(kind, 1.0)
            elif isinstance(op, Reduce):
                w = op.src.shape[0] * op.src.shape[1]
                acc["alu_work"] += w * (2.0 if op.op in ("argmin", "argmax") else 1.0) + op.src.shape[0] * 3
            elif isinstance(op, TmemLoad):
                acc["tmem_bytes"] += op.src.shape[0] * op.src.shape[1] * 4
            elif isinstance(op, LoadG):
                acc["gload"] += 1
                w = 1
                for d in op.dst.shape:
                    w *= d
                acc["gload_bytes"] += w * op.src.dtype.nbytes
            elif isinstance(op, Wait):
                acc["waits"] += 1
                waited.add(id(op.barrier))
            elif isinstance(op, Commit):
                committed.add(id(op.barrier))
        acc["alu"] = acc["alu_work"] / (t.alu_lanes * eff)
        if committed & waited:
            serial_mma.add(r.name)
        roles[r.name] = acc
    return {"roles": roles, "tma_bytes": tma_bytes, "serial_mma": serial_mma}


def featurize(prog: Program, target=None) -> Dict[str, Any]:
    """JSON-serializable features of a program (work counts, occupancy, grid) for prediction/calibration."""
    t = get_target(target or prog.target)
    work = count_work(prog, t)
    c, lim = _occupancy(prog, t)
    tma_l2, tma_hbm = 0.0, 0.0
    n = prog.num_ctas
    for name, b in work["tma_bytes"].items():
        tensor = prog.tensor(name)
        size = float(tensor.dtype.nbytes)
        for d in tensor.shape:
            size *= d
        if (b * n / size if size else 1.0) >= 1.5:
            tma_l2 += b
        else:
            tma_hbm += b
    return {"target": t.name, "roles": work["roles"], "serial_mma": sorted(work["serial_mma"]),
            "tma_l2_bytes": tma_l2, "tma_hbm_bytes": tma_hbm, "ctas_per_sm": c, "limiter": lim, "num_ctas": n,
            "regs": {r.name: _regs_per_thread(prog, r) for r in prog.roles}}


def predict(feat: Dict[str, Any], coef: Dict[str, float]) -> Dict[str, Any]:
    """Fast prediction from features (used for ranking and calibration fitting)."""
    t = get_target(feat["target"])
    c = feat["ctas_per_sm"]
    n = feat["num_ctas"]
    c_eff = max(1, min(c, math.ceil(n / t.num_sms)))
    waves = math.ceil(n / (t.num_sms * c_eff))
    active = min(t.num_sms, math.ceil(n / c_eff))
    clock = t.clock_ghz * 1e9
    hbm_bpc = t.hbm_gbps * 1e9 / clock / active
    l2_bpc = t.l2_gbps * 1e9 / clock / active
    tma = (feat["tma_l2_bytes"] / l2_bpc + feat["tma_hbm_bytes"] / hbm_bpc) * c_eff
    roles = feat["roles"]
    tc = sum(r["tc"] for r in roles.values())
    alu_occ = sum(r["alu_work"] for r in roles.values()) / t.alu_lanes
    tmem = sum(r["tmem_bytes"] for r in roles.values()) / t.tmem_bytes_per_clk
    lat_by_role = {k: r["gload"] * t.l2_latency + r["tmem_bytes"] / t.tmem_bytes_per_clk +
                   (r["tmem_bytes"] > 0) * t.tmem_ld_latency * max(1.0, r["waits"] / 2)
                   for k, r in roles.items()}
    sync = max([r["waits"] for r in roles.values()] + [0.0]) * (t.mbar_cycles + t.wake_cycles)
    fill = t.hbm_latency + t.tma_overhead + 2 * t.umma_latency
    paths = {}
    for k, r in roles.items():
        p = coef["alu"] * r["alu"] + coef["latency"] * lat_by_role[k] + coef["tc"] * r["mma_sync"]
        if k in feat["serial_mma"]:
            p += coef["tc"] * r["tc"]
        paths[k] = p
    crit = max(paths, key=paths.get) if paths else ""
    units = {"tc": coef["tc"] * tc, "tma": coef["tma"] * tma, "tmem": coef["tmem"] * tmem}
    body = max([paths.get(crit, 0.0)] + list(units.values()))
    t_cta = coef["fill"] * fill + body + coef["sync"] * sync
    t_wave = max(t_cta, c_eff * max(coef["tc"] * tc, coef["alu"] * alu_occ, coef["tmem"] * tmem))
    us = waves * t_wave / (t.clock_ghz * 1e3) + coef["launch"] * t.launch_us
    raw = {"tc": tc, "alu": alu_occ, "tmem": tmem, "tma": tma, "latency": max(lat_by_role.values(), default=0.0),
           "sync": sync, "fill": fill, "launch": t.launch_us}
    return {"us": us, "raw": raw, "paths": paths, "crit": crit, "units": units, "t_cta": t_cta, "waves": waves,
            "c_eff": c_eff}


def estimate(prog: Program, target=None, calibration: Optional[Calibration] = None,
             calibrations: Optional[Dict[str, Calibration]] = None, force: bool = False) -> Estimate:
    t = get_target(target or prog.target)
    cal = calibration
    if cal is None:
        cal = (calibrations or default_calibrations()).get(t.name)
    if cal is None and not force:
        return Estimate(None, False, reason=f"no calibrated timing model for {t.name} (coverage limitation); "
                                            f"use on-device measurement")
    cal = cal or Calibration(t.name)
    try:
        feat = featurize(prog, t)
    except CakeError as e:
        return Estimate(None, False, reason=f"could not expand the schedule: {e.message}")
    if feat["ctas_per_sm"] < 1:
        return Estimate(None, False, reason="the CTA does not fit on an SM")
    pr = predict(feat, cal.coef)
    est = Estimate(us=pr["us"], covered=True, raw=pr["raw"],
                   terms={k: pr["raw"][k] * cal.coef[k] for k in TERMS}, cta_cycles=pr["t_cta"], waves=pr["waves"],
                   ctas_per_sm=pr["c_eff"], limiter=feat["limiter"],
                   roles={k: {"path_cycles": round(v, 1), "regs": feat["regs"][k]} for k, v in pr["paths"].items()})
    est.features = feat
    _attribute(est, prog, t, {"roles": feat["roles"], "serial_mma": set(feat["serial_mma"])}, pr["paths"],
               pr["crit"], pr["units"], feat["ctas_per_sm"], pr["c_eff"])
    return est


def _attribute(est: Estimate, prog: Program, t: Target, work, paths, crit_role, units, c, c_eff) -> None:
    roles = work["roles"]
    body = max(list(paths.values()) + list(units.values()) + [1.0])
    unit_max = max(units, key=units.get)
    if units[unit_max] >= 0.85 * body:
        names = {"tc": "tensor-core bound", "tma": "memory bound (TMA/L2 bandwidth)", "tmem": "TMEM-load bound"}
        est.bottleneck = names[unit_max]
        est.attribution = f"{unit_max} work ({units[unit_max]:.0f} cyc/CTA) dominates the critical path."
    else:
        r = roles.get(crit_role, {})
        lat = r.get("gload", 0) * t.l2_latency
        if lat > 0.3 * paths.get(crit_role, 1.0):
            est.bottleneck = "latency bound (synchronous global loads)"
            est.hints.append(f"role `{crit_role}` spends ~{lat:.0f} cyc/CTA waiting on lm.load latency; prefetch the "
                             f"operand into shared memory with TMA")
        elif crit_role in work["serial_mma"]:
            est.bottleneck = "serialized MMA + epilogue"
            est.hints.append(f"role `{crit_role}` waits for its own MMAs before the epilogue; split MMA issue into a "
                             f"separate warp role and double-buffer the TMEM accumulator")
        else:
            est.bottleneck = "epilogue (CUDA-core) bound"
            est.hints.append(f"the `{crit_role}` epilogue (~{paths[crit_role]:.0f} cyc/CTA) is the critical path; use "
                             f"fused ops (fma), more epilogue warps, or overlap via co-resident CTAs")
        est.attribution = f"critical role `{crit_role}` path ~{paths.get(crit_role, 0):.0f} cyc/CTA vs unit work " \
                          f"tc {units['tc']:.0f}, tma {units['tma']:.0f}."
    for k, r in roles.items():
        if r["regs"] > t.max_regs_per_thread:
            est.hints.append(f"role `{k}` needs ~{r['regs']:.0f} regs/thread and will spill; process the accumulator "
                             f"in column chunks")
    if c_eff == 1 and est.limiter in ("smem", "tmem") and prog.num_ctas > t.num_sms:
        est.hints.append(f"only one CTA fits per SM (limited by {est.limiter}); fewer stages / narrower tiles would "
                         f"allow co-resident CTAs to hide latency")
    if est.waves <= 4 and prog.num_ctas % (t.num_sms * max(1, c_eff)):
        frac = (prog.num_ctas % (t.num_sms * c_eff)) / (t.num_sms * c_eff)
        if frac < 0.5:
            est.hints.append(f"the last wave is only {frac:.0%} full (wave quantization); consider a persistent grid")
