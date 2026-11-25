"""Flash-KMeans ``assign`` (paper Sec. 5): nearest-centroid assignment.

For every token x_{b,n} in R^D return argmin_k ||x - c_{b,k}||^2.  Since ||x||^2
is constant per row, kernels minimise ``||c_k||^2 - 2 x.c_k`` with the centroid
norms precomputed (``CN``).  The kernel is a BF16 tensor-core GEMM fused with a
scalar argmin epilogue -- "assign is a compute-bound BF16 GEMM-and-reduction
kernel" that "exercises the tensor-core pipeline and scalar epilogue".
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from .. import dtypes as D
from .base import Case, CheckResult, TensorSpec, Workload, register

TARGET_DIMS = {"B": 32, "N": 65536, "K": 1024, "D": 128}


# ----------------------------------------------------------------------------- schedule template
DEFAULTS: Dict[str, Any] = dict(BN=128, STAGES=2, ACC=1, SPLIT=False, EPI_WARPS=4, CHUNK=0, CN="global",
                                PERSISTENT=False, FMA=True, SWIZZLE="128B")

# The "tuned Triton-style" baseline: pipelined TMA loads by a producer warp, but
# MMA issue and the argmin epilogue serialized in one 4-warp role.
BASELINE_PARAMS = dict(DEFAULTS)

KNOB_SPACE = {
    "BN": [64, 128, 256],
    "STAGES": [1, 2, 3, 4, 5, 6],
    "ACC": [1, 2],
    "SPLIT": [False, True],
    "EPI_WARPS": [4, 8],
    "CHUNK": [0, 64, 32],
    "CN": ["global", "smem"],
    "PERSISTENT": [False, True],
    "FMA": [True, False],
    "SWIZZLE": ["128B", "64B", "32B", "none"],
}

# Bug classes the scripted agent can introduce (see cake.agent.scripted).
BUGS = ("no_release", "stage_mod", "missing_wait", "lane_quadrant", "tail_mask", "smem_overlap",
        "no_commit_free", "early_release")


def normalize(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(DEFAULTS)
    p.update(params)
    if p["PERSISTENT"] or p["ACC"] == 2:
        p["SPLIT"] = True
    if not p["SPLIT"]:
        p["ACC"] = 1
    if p["CHUNK"] and p["CHUNK"] >= p["BN"]:
        p["CHUNK"] = 0
    return p


def template(params: Dict[str, Any], bugs: Iterable[str] = ()) -> str:
    """Render a Cake IR schedule for the assign kernel from schedule knobs."""
    p = normalize(params)
    bugs = set(bugs)
    BN, ST, ACC, SPLIT, EW = p["BN"], p["STAGES"], p["ACC"], p["SPLIT"], p["EPI_WARPS"]
    CH = p["CHUNK"] or BN
    PERS, SW = p["PERSISTENT"], p["SWIZZLE"]
    cn_smem = p["CN"] == "smem"
    L = []
    a = L.append
    cn_ann = "LM.tma2d" if cn_smem else "LM.gmem"
    a("@cake.schedule()")
    a(f"def kmeans_assign(lm, X: LM.tma3d, C: LM.tma3d, CN: {cn_ann}, OUT: LM.gmem):")
    a('    B, N, K, D = lm.dims("B", "N", "K", "D")')
    a(f"    BM, BN, STAGES, ACC = 128, {BN}, {ST}, {ACC}")
    if SPLIT:
        mma_w = [1]
        epi_w = list(range(2, 2 + EW))
    else:
        mma_w = []
        epi_w = list(range(1, 1 + EW))
    if "lane_quadrant" in bugs:
        removed = sum(1 for w in epi_w if w % 4 == 1)
        epi_w = [w for w in epi_w if w % 4 != 1]
        nxt = max(epi_w) + 1
        while removed:
            if nxt % 4 == 2:
                epi_w.append(nxt)
                removed -= 1
            nxt += 1
    warps = max(epi_w) + 1
    a("    TPB = lm.cdiv(N, BM)          # row tiles per batch")
    a("    nk = lm.cdiv(K, BN)           # centroid chunks")
    if PERS:
        a("    T = B * TPB")
        a("    G = lm.min(T, lm.num_sms())  # persistent grid")
        a(f"    lm.launch(grid=(G,), warps={warps})")
    else:
        a(f"    lm.launch(grid=(TPB, B), warps={warps})")
    a("    X_BYTES = BM * D * 2")
    a("    C_BYTES = BN * D * 2")
    pool_expr = "X_BYTES + STAGES * C_BYTES"
    if cn_smem:
        a(f"    CW = {CH}")
        a("    NCW = lm.cdiv(K, CW)")
        pool_expr += " + NCW * CW * 4"
    a(f"    pool = lm.smem({pool_expr})")
    a(f'    sx = pool.view(offset=0, shape=(BM, D), dtype=lm.bf16, swizzle="{SW}")')
    c_off = "X_BYTES - 1024" if "smem_overlap" in bugs else "X_BYTES"
    a(f'    sc = pool.view(offset={c_off}, shape=(BN, D), dtype=lm.bf16, stage=STAGES, swizzle="{SW}")')
    if cn_smem:
        a('    scn = pool.view(offset=X_BYTES + STAGES * C_BYTES, shape=(1, CW), dtype=lm.f32, stage=NCW, '
          'swizzle="none")')
    a("    acc = lm.tmem(cols=0, width=BN, shape=(BM, BN), dtype=lm.f32, stage=ACC)")
    a("")
    a("    load = lm.role(warps=[0])")
    if SPLIT:
        a(f"    mma = lm.role(warps={mma_w})")
        a(f"    epi = lm.role(warps={epi_w})")
        mma, epi = "mma", "epi"
    else:
        a(f"    comp = lm.role(warps={epi_w})")
        mma = epi = "comp"
    a(f"    x_full = lm.barrier(prod=[load], cons=[{mma}])")
    if PERS:
        a(f"    x_free = lm.barrier(prod=[{mma}], cons=[load], released=True)")
    a(f"    c_full = lm.barrier(count=STAGES, prod=[load], cons=[{mma}])")
    rel = "" if "no_release" in bugs else ", released=True"
    a(f"    c_free = lm.barrier(count=STAGES, prod=[{mma}], cons=[load]{rel})")
    a(f"    acc_full = lm.barrier(count=ACC, prod=[{mma}], cons=[{epi}])")
    if SPLIT:
        a(f"    acc_free = lm.barrier(count=ACC, prod=[{epi}], cons=[{mma}], released=True)")
    if cn_smem:
        a(f"    cn_full = lm.barrier(prod=[load], cons=[{epi}])")
        if PERS:
            a(f"    cn_free = lm.barrier(prod=[{epi}], cons=[load], released=True)")
    a("")

    smod = "STAGES"   # producer-side ring index
    # consumer-side ring index; the `stage_mod` bug desynchronizes it from the producer
    cmod = "(STAGES - 1)" if ("stage_mod" in bugs and ST > 1) else "STAGES"

    # Inside a role block the per-tile body is indented by I: a persistent kernel
    # wraps it in a tile loop, a one-tile-per-CTA kernel does not.
    I = "            " if PERS else "        "

    def tile_header() -> None:
        if PERS:
            a("        for t in lm.range(lm.block_idx(0), T, G):")
            a(f"{I}b = t // TPB")
            a(f"{I}m0 = (t % TPB) * BM")
            a(f"{I}ti = (t - lm.block_idx(0)) // G")
        else:
            a(f"{I}b = lm.block_idx(1)")
            a(f"{I}m0 = lm.block_idx(0) * BM")
            a(f"{I}ti = 0")

    # ---------------- producer
    a("    with load:")
    tile_header()
    if PERS:
        a(f"{I}lm.wait(x_free)")
    a(f"{I}sx.tma_load(X, coords=(b, m0, 0), barrier=x_full)")
    if cn_smem:
        if PERS:
            a(f"{I}lm.wait(cn_free)")
        a(f"{I}for q in range(NCW):")
        a(f"{I}    scn.tma_load(CN, coords=(b, q * CW), stage=q, barrier=cn_full)")
    a(f"{I}for j in lm.range(0, nk):")
    a(f"{I}    g = ti * nk + j")
    a(f"{I}    lm.wait(c_free, stage=g % {smod})")
    a(f"{I}    sc.tma_load(C, coords=(b, j * BN, 0), stage=g % {smod}, barrier=c_full)")
    a("")

    # ---------------- MMA issue (+ epilogue when not split)
    nparts = BN // CH

    def col_of(q: int) -> str:
        return f"j * BN + {q * CH}" if q else "j * BN"

    def cn_fetch(ind: str, q: int) -> None:
        if cn_smem:
            stage = f"j * {nparts} + {q}" if nparts > 1 else "j"
            a(f"{ind}lm.ld_smem(cn, scn[{stage}])")
        else:
            a(f"{ind}lm.load(cn, CN, coords=(b, {col_of(q)}))")

    def epilogue(ind: str, acc_ref: str) -> None:
        # the first cn fetch was issued before waiting on the accumulator
        for q in range(nparts):
            if q:
                cn_fetch(ind, q)
            part = f"{acc_ref}.part({q * CH}, {CH})" if nparts > 1 else acc_ref
            if SPLIT and q == 0 and "early_release" in bugs:
                a(f"{ind}lm.arrive(acc_free, stage=g % ACC)")
            a(f"{ind}lm.tmem_load(s, {part})")
            if SPLIT and q == nparts - 1 and "early_release" not in bugs:
                a(f"{ind}lm.arrive(acc_free, stage=g % ACC)")
            if p["FMA"]:
                a(f'{ind}lm.ewise(s, "fma", s, -2.0, cn.row())')
            else:
                a(f'{ind}lm.ewise(s, "mul", s, -2.0)')
                a(f'{ind}lm.ewise(s, "add", s, cn.row())')
            valid = "" if "tail_mask" in bugs else f", valid=K - ({col_of(q)})"
            a(f'{ind}lm.reduce(best, s, "argmin", idx=arg, index_offset={col_of(q)}{valid})')

    if SPLIT:
        a("    with mma:")
        tile_header()
        a(f"{I}lm.wait(x_full)")
        a(f"{I}for j in lm.range(0, nk):")
        a(f"{I}    g = ti * nk + j")
        if "missing_wait" not in bugs:
            a(f"{I}    lm.wait(c_full, stage=g % {cmod})")
        a(f"{I}    lm.wait(acc_free, stage=g % ACC)")
        a(f"{I}    lm.mma(acc[g % ACC], sx, sc[g % {cmod}], init=True)")
        if "no_commit_free" not in bugs:
            a(f"{I}    lm.commit(c_free, stage=g % {cmod})")
        a(f"{I}    lm.commit(acc_full, stage=g % ACC)")
        if PERS:
            a(f"{I}lm.commit(x_free)")
        a("")
        a("    with epi:")
    else:
        a("    with comp:")
    a("        best = lm.regs((BM,), lm.f32)")
    a("        arg = lm.regs((BM,), lm.i32)")
    a(f"        s = lm.regs((BM, {CH}), lm.f32)")
    a(f"        cn = lm.regs(({CH},), lm.f32)")
    tile_header()
    a(f"{I}lm.fill(best, float('inf'))")
    a(f"{I}lm.fill(arg, 0)")
    if not SPLIT:
        a(f"{I}lm.wait(x_full)")
    if cn_smem:
        a(f"{I}lm.wait(cn_full)")
    a(f"{I}for j in lm.range(0, nk):")
    a(f"{I}    g = ti * nk + j")
    J = I + "    "
    if not SPLIT:
        if "missing_wait" not in bugs:
            a(f"{J}lm.wait(c_full, stage=g % {cmod})")
        a(f"{J}lm.mma(acc, sx, sc[g % {cmod}], init=True)")
        if "no_commit_free" not in bugs:
            a(f"{J}lm.commit(c_free, stage=g % {cmod})")
        a(f"{J}lm.commit(acc_full)")
        cn_fetch(J, 0)
        a(f"{J}lm.wait(acc_full)")
        epilogue(J, "acc")
    else:
        cn_fetch(J, 0)
        a(f"{J}lm.wait(acc_full, stage=g % ACC)")
        epilogue(J, "acc[g % ACC]")
    if cn_smem and PERS:
        a(f"{I}lm.arrive(cn_free)")
    a(f"{I}lm.store(OUT, coords=(b, m0), src=arg, valid=N - m0)")
    return "\n".join(L) + "\n"


BASELINE_SOURCE = template(BASELINE_PARAMS)


# ----------------------------------------------------------------------------- contract
@register("kmeans_assign")
class KMeansAssign(Workload):
    name = "kmeans_assign"
    title = "Flash-KMeans assign: nearest-centroid argmin (BF16 GEMM + argmin epilogue)"
    description = """
For each batch b and token n: OUT[b, n] = argmin_k ( CN[b, k] - 2 * dot(X[b, n, :], C[b, k, :]) ),
which equals argmin_k ||X[b,n] - C[b,k]||^2 because ||X[b,n]||^2 is constant per row.
CN[b, k] = ||C[b, k]||^2 is precomputed in FP32 from the BF16 centroids.
Inputs are BF16; dot products must accumulate in FP32 (tensor cores). Ties may resolve to any
minimizing index. NumPy reference:
    d = CN[:, None, :] - 2 * np.einsum("bnd,bkd->bnk", X, C);  OUT = d.argmin(-1)
"""
    tensors = [
        TensorSpec("X", ("B", "N", "D"), D.bf16, "input"),
        TensorSpec("C", ("B", "K", "D"), D.bf16, "input"),
        TensorSpec("CN", ("B", "K"), D.f32, "input"),
        TensorSpec("OUT", ("B", "N"), D.i32, "output"),
    ]
    target_dims = dict(TARGET_DIMS)
    target = "sm_100a"
    cases = [
        Case("small", (("B", 2), ("N", 1024), ("K", 1024), ("D", 128)), "gaussian", 1),
        Case("clustered", (("B", 1), ("N", 2048), ("K", 1024), ("D", 128)), "clustered", 2),
        Case("multi_tile", (("B", 1), ("N", 128 * (2 * 148 + 3)), ("K", 1024), ("D", 128)), "gaussian", 3,
             sample_ctas=3),
    ]
    tolerance = ("an index is correct if its distance is within 2e-4*max(1,|d_min|)+1e-3 of the minimum "
                 "(fp32 accumulation-order differences and exact ties are accepted)")
    permitted_references = ("this mathematical specification, the NumPy reference above, the evaluation "
                            "contract and the baseline Cake IR schedule; no low-level (CUDA/PTX/SASS) "
                            "implementations of the kernel")
    baseline_name = "triton_style_baseline"
    baseline_source = BASELINE_SOURCE

    def make_inputs(self, dims, dist, seed):
        B, N, K, Dm = dims["B"], dims["N"], dims["K"], dims["D"]
        rng = np.random.default_rng(seed)
        if dist == "gaussian":
            X = rng.standard_normal((B, N, Dm), dtype=np.float32)
            C = rng.standard_normal((B, K, Dm), dtype=np.float32)
        elif dist == "clustered":
            C = rng.standard_normal((B, K, Dm), dtype=np.float32) * 2.0
            lab = rng.integers(0, K, size=(B, N))
            X = np.take_along_axis(C, lab[:, :, None], axis=1) + 0.25 * rng.standard_normal((B, N, Dm),
                                                                                              dtype=np.float32)
        elif dist == "ties":
            C = rng.standard_normal((B, K, Dm), dtype=np.float32)
            C[:, 1::2] = C[:, 0:K - K % 2:2][:, :C[:, 1::2].shape[1]]
            X = rng.standard_normal((B, N, Dm), dtype=np.float32)
        elif dist == "uniform":
            X = rng.uniform(-1, 1, (B, N, Dm)).astype(np.float32)
            C = rng.uniform(-1, 1, (B, K, Dm)).astype(np.float32)
        else:
            raise ValueError(f"unknown distribution {dist}")
        Xq = D.encode(X, D.bf16)
        Cq = D.encode(C, D.bf16)
        Cf = D.decode(Cq, D.bf16)
        CN = (Cf.astype(np.float64) ** 2).sum(-1).astype(np.float32)
        return {"X": Xq, "C": Cq, "CN": CN}

    def make_outputs(self, dims):
        return {"OUT": np.full((dims["B"], dims["N"]), -1, dtype=np.int32)}

    def oracle(self, inputs, dims):
        B, N = dims["B"], dims["N"]
        X = D.decode(inputs["X"], D.bf16)
        C = D.decode(inputs["C"], D.bf16)
        CN = inputs["CN"].astype(np.float64)
        idx = np.empty((B, N), dtype=np.int32)
        dmin = np.empty((B, N), dtype=np.float64)
        for b in range(B):
            Cb = C[b].astype(np.float64)
            for n0 in range(0, N, 4096):
                d = CN[b][None, :] - 2.0 * (X[b, n0:n0 + 4096].astype(np.float64) @ Cb.T)
                idx[b, n0:n0 + 4096] = d.argmin(axis=1)
                dmin[b, n0:n0 + 4096] = d.min(axis=1)
        return {"OUT": idx, "dmin": dmin}

    def check(self, outputs, expected, inputs, dims, full_coverage):
        out = outputs["OUT"]
        K = dims["K"]
        written = out != -1
        if full_coverage and not written.all():
            miss = np.argwhere(~written)
            return CheckResult(False, f"{int((~written).sum())} of {out.size} output rows were never written "
                                      f"(first: b={miss[0][0]}, n={miss[0][1]})", int(out.size),
                               int((~written).sum()),
                               [{"b": int(m[0]), "n": int(m[1]), "got": -1} for m in miss[:5]])
        if not written.any():
            return CheckResult(False, "the kernel wrote no output rows", 0, 0)
        bs, ns = np.nonzero(written)
        got = out[bs, ns]
        bad_range = (got < 0) | (got >= K)
        X = D.decode(inputs["X"], D.bf16)
        C = D.decode(inputs["C"], D.bf16)
        CN = inputs["CN"].astype(np.float64)
        gk = np.clip(got, 0, K - 1)
        chosen = CN[bs, gk] - 2.0 * np.einsum("nd,nd->n", X[bs, ns].astype(np.float64),
                                              C[bs, gk].astype(np.float64))
        dmin = expected["dmin"][bs, ns]
        tol = 2e-4 * np.maximum(1.0, np.abs(dmin)) + 1e-3
        bad = bad_range | (chosen - dmin > tol)
        nbad = int(bad.sum())
        if nbad == 0:
            return CheckResult(True, f"{len(got)} rows match the reference", len(got), 0)
        ex = []
        for i in np.nonzero(bad)[0][:5]:
            ex.append({"b": int(bs[i]), "n": int(ns[i]), "got": int(got[i]),
                       "expected": int(expected["OUT"][bs[i], ns[i]]),
                       "distance_gap": None if bad_range[i] else float(chosen[i] - dmin[i])})
        return CheckResult(False, f"{nbad} of {len(got)} rows assigned to a non-nearest centroid "
                                  f"({nbad / len(got):.1%})", len(got), nbad, ex)
