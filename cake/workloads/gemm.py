"""Dense BF16 GEMM ``OUT = A @ B^T`` (FP32 accumulation, BF16 output): a second kernel family."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from .. import dtypes as D
from .base import Case, CheckResult, TensorSpec, Workload, register

DEFAULTS: Dict[str, Any] = dict(BN=128, BK=64, STAGES=3, SPLIT=True, SWIZZLE="128B")


def template(params: Dict[str, Any], bugs: Iterable[str] = ()) -> str:
    p = dict(DEFAULTS)
    p.update(params)
    BN, BK, ST, SW = p["BN"], p["BK"], p["STAGES"], p["SWIZZLE"]
    rel = "" if "no_release" in set(bugs) else ", released=True"
    if p["SPLIT"]:
        roles = ("    load = lm.role(warps=[0])\n    mma = lm.role(warps=[1])\n"
                 "    epi = lm.role(warps=[2, 3, 4, 5])\n")
        mma, epi, warps = "mma", "epi", 6
    else:
        roles = "    load = lm.role(warps=[0])\n    comp = lm.role(warps=[1, 2, 3, 4])\n"
        mma, epi, warps = "comp", "comp", 5
    src = f'''@cake.schedule()
def gemm(lm, A: LM.tma2d, Bm: LM.tma2d, OUT: LM.gmem):
    M, N, K = lm.dims("M", "N", "K")
    BM, BN, BK, STAGES = 128, {BN}, {BK}, {ST}
    lm.launch(grid=(lm.cdiv(N, BN), lm.cdiv(M, BM)), warps={warps})
    m0 = lm.block_idx(1) * BM
    n0 = lm.block_idx(0) * BN
    A_BYTES = BM * BK * 2
    B_BYTES = BN * BK * 2
    pool = lm.smem(STAGES * (A_BYTES + B_BYTES))
    sa = pool.view(offset=0, shape=(BM, BK), dtype=lm.bf16, stage=STAGES, swizzle="{SW}")
    sb = pool.view(offset=STAGES * A_BYTES, shape=(BN, BK), dtype=lm.bf16, stage=STAGES, swizzle="{SW}")
    acc = lm.tmem(cols=0, width=BN, shape=(BM, BN), dtype=lm.f32)
{roles}    full = lm.barrier(count=STAGES, prod=[load], cons=[{mma}])
    free = lm.barrier(count=STAGES, prod=[{mma}], cons=[load]{rel})
    done = lm.barrier(prod=[{mma}], cons=[{epi}])
    nk = lm.cdiv(K, BK)
    with load:
        for k in lm.range(0, nk):
            lm.wait(free, stage=k % STAGES)
            sa.tma_load(A, coords=(m0, k * BK), stage=k % STAGES, barrier=full)
            sb.tma_load(Bm, coords=(n0, k * BK), stage=k % STAGES, barrier=full)
'''
    mma_body = '''        for k in lm.range(0, nk):
            lm.wait(full, stage=k % STAGES)
            lm.mma(acc, sa[k % STAGES], sb[k % STAGES], init=(k == 0))
            lm.commit(free, stage=k % STAGES)
        lm.commit(done)
'''
    epi_body = '''        lm.wait(done)
        lm.tmem_load(r, acc)
        lm.convert(o, r)
        lm.store(OUT, coords=(m0, n0), src=o, valid=M - m0)
'''
    regs = '''        r = lm.regs((BM, BN), lm.f32)
        o = lm.regs((BM, BN), lm.bf16)
'''
    if p["SPLIT"]:
        src += "    with mma:\n" + mma_body + "    with epi:\n" + regs + epi_body
    else:
        src += "    with comp:\n" + regs + mma_body + epi_body
    return src


BASELINE_SOURCE = template(dict(DEFAULTS, SPLIT=False, STAGES=2))


@register("gemm")
class Gemm(Workload):
    name = "gemm"
    title = "Dense BF16 GEMM: OUT = A @ B^T with FP32 accumulation and BF16 output"
    description = """
OUT[m, n] = bf16( sum_k A[m, k] * Bm[n, k] ), accumulated in FP32. NumPy reference:
    OUT = (A.astype(np.float32) @ Bm.astype(np.float32).T).astype(bfloat16)
M and N are multiples of the 128x128 output tile in every evaluated shape.
"""
    tensors = [
        TensorSpec("A", ("M", "K"), D.bf16, "input"),
        TensorSpec("Bm", ("N", "K"), D.bf16, "input"),
        TensorSpec("OUT", ("M", "N"), D.bf16, "output"),
    ]
    target_dims = {"M": 4096, "N": 4096, "K": 4096}
    target = "sm_100a"
    cases = [
        Case("small", (("M", 256), ("N", 256), ("K", 512)), "gaussian", 1),
        Case("deep_k", (("M", 128), ("N", 256), ("K", 2048)), "gaussian", 2),
    ]
    tolerance = "|out - ref| <= 2e-2 * max(1, |ref|) elementwise (BF16 output rounding)"
    permitted_references = "the specification above and the baseline Cake IR schedule"
    baseline_name = "serialized_baseline"
    baseline_source = BASELINE_SOURCE

    def make_inputs(self, dims, dist, seed):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((dims["M"], dims["K"]), dtype=np.float32)
        Bm = rng.standard_normal((dims["N"], dims["K"]), dtype=np.float32)
        return {"A": D.encode(A, D.bf16), "Bm": D.encode(Bm, D.bf16)}

    def make_outputs(self, dims):
        return {"OUT": np.full((dims["M"], dims["N"]), 0x7FC0, dtype=np.uint16)}  # NaN sentinel

    def oracle(self, inputs, dims):
        A = D.decode(inputs["A"], D.bf16).astype(np.float64)
        Bm = D.decode(inputs["Bm"], D.bf16).astype(np.float64)
        return {"OUT": A @ Bm.T}

    def check(self, outputs, expected, inputs, dims, full_coverage):
        out = D.decode(outputs["OUT"], D.bf16).astype(np.float64)
        ref = expected["OUT"]
        written = ~np.isnan(out)
        if full_coverage and not written.all():
            return CheckResult(False, f"{int((~written).sum())} output elements were never written",
                               int(out.size), int((~written).sum()))
        if not written.any():
            return CheckResult(False, "the kernel wrote no output", 0, 0)
        err = np.abs(out - ref)
        bad = written & (err > 2e-2 * np.maximum(1.0, np.abs(ref)))
        nbad = int(bad.sum())
        if nbad == 0:
            return CheckResult(True, f"{int(written.sum())} elements match", int(written.sum()), 0)
        idx = np.argwhere(bad)[:5]
        ex = [{"m": int(i), "n": int(j), "got": float(out[i, j]), "expected": float(ref[i, j])} for i, j in idx]
        return CheckResult(False, f"{nbad} of {int(written.sum())} elements differ (max abs err "
                                  f"{float(err[written].max()):.3g})", int(written.sum()), nbad, ex)
