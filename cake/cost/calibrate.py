"""Cost-model calibration: "a systematic misprediction becomes a calibration target" (Sec. 3.2).

Coefficients are fitted on measured samples (features + measured time) by
minimizing squared log error with Nelder-Mead in log space.  A new calibration
is accepted only if it improves held-out error without degrading held-out rank
correlation -- the cost model is used to *rank* candidates, so ranking quality
is part of the gate.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .model import TERMS, Calibration, predict

Sample = Tuple[Dict[str, Any], float]  # (features, measured_us)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics(samples: List[Sample], coef: Dict[str, float]) -> Dict[str, float]:
    if not samples:
        return {"n": 0, "mae_log": float("nan"), "spearman": float("nan"), "bias": float("nan")}
    pred = np.array([predict(f, coef)["us"] for f, _ in samples])
    meas = np.array([m for _, m in samples])
    err = np.log(pred / meas)
    return {"n": len(samples), "mae_log": float(np.mean(np.abs(err))), "spearman": spearman(pred, meas),
            "bias": float(np.mean(err))}


def nelder_mead(f, x0: np.ndarray, iters: int = 400, step: float = 0.25, tol: float = 1e-7) -> np.ndarray:
    n = len(x0)
    pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    return pts[int(np.argmin(vals))]


def fit(samples: List[Sample], init: Dict[str, float], reg: float = 0.01) -> Dict[str, float]:
    keys = list(TERMS)
    meas = np.log(np.array([m for _, m in samples]))

    def obj(x: np.ndarray) -> float:
        x = np.clip(x, -2.5, 2.5)
        coef = {k: float(math.exp(v)) for k, v in zip(keys, x)}
        pred = np.log(np.array([predict(f, coef)["us"] for f, _ in samples]))
        return float(np.mean((pred - meas) ** 2) + reg * np.mean(x ** 2))

    x0 = np.array([math.log(max(1e-3, init.get(k, 1.0))) for k in keys])
    best = np.clip(nelder_mead(obj, x0), -2.5, 2.5)
    return {k: round(float(math.exp(v)), 4) for k, v in zip(keys, best)}


def split(samples: List[Sample], holdout: float = 0.3) -> Tuple[List[Sample], List[Sample]]:
    """Deterministic train / held-out split keyed on the sample contents."""
    train, test = [], []
    for f, m in samples:
        key = repr(sorted((k, repr(v)) for k, v in f.items() if k != "regs")) + f"{m:.6f}"
        (test if (zlib.crc32(key.encode()) % 1000) / 1000.0 < holdout else train).append((f, m))
    return train, test


def calibrate(samples: List[Sample], current: Calibration, holdout: float = 0.3,
              min_samples: int = 8) -> Tuple[Calibration, Dict[str, Any], bool]:
    """Fit a candidate calibration and gate it on held-out samples."""
    train, test = split(samples, holdout)
    if len(train) < min_samples or len(test) < 3:
        return current, {"reason": f"not enough samples ({len(train)} train / {len(test)} held-out)",
                         "accepted": False}, False
    cand = fit(train, current.coef)
    before, after = metrics(test, current.coef), metrics(test, cand)
    accepted = (after["mae_log"] < before["mae_log"] * 0.95) and (after["spearman"] >= before["spearman"] - 0.02)
    report = {"train": len(train), "held_out": len(test), "before": before, "after": after, "coef": cand,
              "accepted": accepted}
    if not accepted:
        return current, report, False
    return Calibration(current.target, cand, current.version + 1, len(samples), after), report, True
