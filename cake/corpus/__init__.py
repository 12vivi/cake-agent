"""The validated kernel corpus.

Compiler changes are "test-gated across the kernel corpus" (paper Sec. 3.2):
a new verifier rule must not reject any corpus kernel, and IR changes must keep
every corpus kernel constructing, verifying and running correctly (P6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..workloads import kmeans_assign as km
from ..workloads import gemm as gm


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    workload: str
    source: str
    dims: Dict[str, int]
    run: bool = True          # also check numerics on the simulated device (False: static/compile case)
    target: str = "sm_100a"


FIG3_FMHA_FRAGMENT = '''@cake.schedule()
def fmha_fwd(lm, Q: LM.tma3d, O: LM.tma2d):
    # Figure 3 of the paper (coordinates in numpy order): declare resources and roles;
    # the producer-consumer hand-off is an explicit barrier; addresses, phases and
    # warp identity are derived by lowering.
    pool = lm.smem(98304)
    smem_q = pool.view(offset=0, shape=(128, 128), dtype=lm.bf16, stage=3)
    tmem_acc = lm.tmem(cols=0, width=128, shape=(128, 128), dtype=lm.f32)
    load = lm.role(warps=[0])
    mma = lm.role(warps=[1])
    pipe = lm.pipeline(stages=3)
    q_full = lm.barrier(count=3, prod=[load], cons=[mma], init_count=1, pipeline=pipe)
    with load:
        for stage in lm.range(0, 3):
            smem_q.tma_load(Q, coords=(stage, 0, 0), stage=stage, barrier=q_full)
    with mma:
        for stage in lm.range(0, 3):
            lm.wait(q_full, stage=stage)
            lm.fence_proxy()
            lm.mma(tmem_acc, smem_q[stage], smem_q[stage], init=(stage == 0))
'''

_KM_VARIANTS = {
    "kmeans_baseline": km.BASELINE_PARAMS,
    "kmeans_stages1": dict(STAGES=1),
    "kmeans_specialized": dict(SPLIT=True, ACC=2, STAGES=2),
    "kmeans_persistent_smem_8warps": dict(SPLIT=True, ACC=2, STAGES=4, PERSISTENT=True, CN="smem", EPI_WARPS=8),
    "kmeans_bn256_chunked": dict(SPLIT=True, ACC=2, BN=256, STAGES=2, CHUNK=64, EPI_WARPS=8),
    "kmeans_bn64_unfused": dict(BN=64, STAGES=3, FMA=False, SWIZZLE="64B"),
}

SMALL_KM = {"B": 2, "N": 1024, "K": 1024, "D": 128}
TAIL_KM = {"B": 1, "N": 1000, "K": 1000, "D": 128}


def corpus() -> List[CorpusEntry]:
    out: List[CorpusEntry] = []
    for name, params in _KM_VARIANTS.items():
        out.append(CorpusEntry(name, "kmeans_assign", km.template(params), dict(SMALL_KM)))
    out.append(CorpusEntry("kmeans_baseline_tail", "kmeans_assign", km.template(km.BASELINE_PARAMS), dict(TAIL_KM)))
    out.append(CorpusEntry("kmeans_specialized_tail", "kmeans_assign",
                           km.template(dict(SPLIT=True, ACC=2, CN="smem")), dict(TAIL_KM)))
    out.append(CorpusEntry("gemm_baseline", "gemm", gm.BASELINE_SOURCE, {"M": 256, "N": 256, "K": 512}))
    out.append(CorpusEntry("gemm_specialized", "gemm", gm.template({"STAGES": 4}), {"M": 256, "N": 256, "K": 512}))
    out.append(CorpusEntry("fig3_fmha_fragment", "fig3", FIG3_FMHA_FRAGMENT, {}, run=False))
    return out


def build_entry(e: CorpusEntry, target: Optional[str] = None):
    """Construct a corpus entry's Cake IR program."""
    from .. import dtypes as D
    from ..harness import sandbox
    from ..workloads import get_workload
    if e.workload == "fig3":
        from ..harness.sandbox import load
        sched, _ = load(e.source)
        return sched.build({}, {"Q": ((3, 128, 128), D.bf16), "O": ((128, 128), D.bf16)}, target=target or e.target)
    w = get_workload(e.workload)
    return sandbox.build(e.source, w, e.dims, target or e.target)
