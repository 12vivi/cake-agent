"""Architecture matrix (paper Table 3) and simulated-device parameters.

Capability flags drive the *hardware conformance* checks; the numeric fields
parameterize the simulated GPU that stands in for real hardware in this
reproduction.  Numbers are approximations of public specifications, not
measurements.  ``timing_calibrated`` mirrors the paper: the cost model is
calibrated only for B200 and H100 and declines to predict elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .errors import TargetError


@dataclass(frozen=True)
class Target:
    name: str
    sku: str
    family: str
    notes: str
    # instruction / feature surface
    tma: bool
    tmem: bool
    tcgen05: bool
    wgmma: bool
    mma_sync: bool
    cp_async: bool
    clusters: bool
    mma_dtypes: Tuple[str, ...]
    # resources
    num_sms: int
    clock_ghz: float
    smem_per_sm: int
    smem_per_cta: int
    tmem_cols: int
    max_threads_per_sm: int
    max_ctas_per_sm: int
    # performance parameters of the simulated device
    tc_flops_per_clk: float        # dense BF16 FLOP / clk / SM
    hbm_gbps: float
    l2_gbps: float
    l2_bytes: int
    timing_calibrated: bool
    regs_per_sm: int = 65536
    max_regs_per_thread: int = 255
    max_warps_per_cta: int = 32
    alu_lanes: int = 128
    smem_bytes_per_clk: int = 128
    tmem_bytes_per_clk: int = 512
    hbm_latency: int = 700
    l2_latency: int = 300
    tma_overhead: int = 120
    mbar_cycles: int = 24
    wake_cycles: int = 30
    issue_cycles: int = 4
    commit_cycles: int = 20
    tmem_ld_latency: int = 40
    umma_latency: int = 64
    launch_us: float = 3.0
    extra: Dict[str, float] = field(default_factory=dict)

    def tc_rate(self, dtype_name: str) -> float:
        """Tensor-core FLOP/clk/SM for an input dtype."""
        if dtype_name == "fp8":
            return self.tc_flops_per_clk * 2.0
        return self.tc_flops_per_clk

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // 32

    def summary(self) -> str:
        feats = [n for n, on in (("TMA", self.tma), ("TMEM", self.tmem), ("tcgen05.mma", self.tcgen05),
                                 ("WGMMA", self.wgmma), ("mma.sync", self.mma_sync),
                                 ("cp.async", self.cp_async), ("clusters", self.clusters)) if on]
        return (f"{self.name} ({self.sku}): {self.num_sms} SMs @ {self.clock_ghz} GHz, "
                f"{self.smem_per_cta // 1024} KB smem/CTA, TMEM {self.tmem_cols} cols, "
                f"tensor-core dtypes {list(self.mma_dtypes)}, features: {', '.join(feats)}")


TARGETS: Dict[str, Target] = {
    "sm_80": Target(
        name="sm_80", sku="A100", family="ampere",
        notes="mma.sync, ldmatrix, cp.async; no TMA, clusters, or TMEM",
        tma=False, tmem=False, tcgen05=False, wgmma=False, mma_sync=True, cp_async=True, clusters=False,
        mma_dtypes=("bf16", "f16"),
        num_sms=108, clock_ghz=1.41, smem_per_sm=167936, smem_per_cta=166912, tmem_cols=0,
        max_threads_per_sm=2048, max_ctas_per_sm=32,
        tc_flops_per_clk=2048, hbm_gbps=2039, l2_gbps=5000, l2_bytes=40 << 20,
        timing_calibrated=False, launch_us=4.0, hbm_latency=650, l2_latency=280),
    "sm_89": Target(
        name="sm_89", sku="L40S / RTX 6000 Ada", family="ada",
        notes="mma.sync, ldmatrix, cp.async; FP8 tensor cores, no TMA or TMEM",
        tma=False, tmem=False, tcgen05=False, wgmma=False, mma_sync=True, cp_async=True, clusters=False,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=142, clock_ghz=2.52, smem_per_sm=102400, smem_per_cta=101376, tmem_cols=0,
        max_threads_per_sm=1536, max_ctas_per_sm=24,
        tc_flops_per_clk=1024, hbm_gbps=864, l2_gbps=3000, l2_bytes=96 << 20,
        timing_calibrated=False, launch_us=3.5),
    "sm_90a": Target(
        name="sm_90a", sku="H100 / H200", family="hopper",
        notes="WGMMA + mma.sync; TMA, thread-block clusters, async barriers, DSM",
        tma=True, tmem=False, tcgen05=False, wgmma=True, mma_sync=True, cp_async=True, clusters=True,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=132, clock_ghz=1.83, smem_per_sm=233472, smem_per_cta=232448, tmem_cols=0,
        max_threads_per_sm=2048, max_ctas_per_sm=32,
        tc_flops_per_clk=4096, hbm_gbps=3350, l2_gbps=20000, l2_bytes=50 << 20,
        timing_calibrated=True, launch_us=3.0, hbm_latency=650, l2_latency=280),
    "sm_100a": Target(
        name="sm_100a", sku="B200", family="blackwell",
        notes="tcgen05.mma + TMEM; 2-CTA MMA (cta_group::2), tcgen05.{ld,cp,shift}",
        tma=True, tmem=True, tcgen05=True, wgmma=False, mma_sync=True, cp_async=True, clusters=True,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=148, clock_ghz=1.86, smem_per_sm=233472, smem_per_cta=232448, tmem_cols=512,
        max_threads_per_sm=2048, max_ctas_per_sm=32,
        tc_flops_per_clk=8192, hbm_gbps=8000, l2_gbps=40000, l2_bytes=126 << 20,
        timing_calibrated=True, launch_us=3.0),
    "sm_103a": Target(
        name="sm_103a", sku="B300", family="blackwell",
        notes="adds tcgen05.ld.red, K=96 block-scaled MMA",
        tma=True, tmem=True, tcgen05=True, wgmma=False, mma_sync=True, cp_async=True, clusters=True,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=160, clock_ghz=1.86, smem_per_sm=233472, smem_per_cta=232448, tmem_cols=512,
        max_threads_per_sm=2048, max_ctas_per_sm=32,
        tc_flops_per_clk=8192, hbm_gbps=8000, l2_gbps=40000, l2_bytes=126 << 20,
        timing_calibrated=False, launch_us=3.0),
    "sm_120a": Target(
        name="sm_120a", sku="RTX 5090 / RTX PRO 6000", family="blackwell-consumer",
        notes="mma.sync + ldmatrix, TMA, clusters, DSM; no tcgen05/TMEM",
        tma=True, tmem=False, tcgen05=False, wgmma=False, mma_sync=True, cp_async=True, clusters=True,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=170, clock_ghz=2.41, smem_per_sm=102400, smem_per_cta=101376, tmem_cols=0,
        max_threads_per_sm=1536, max_ctas_per_sm=24,
        tc_flops_per_clk=512, hbm_gbps=1792, l2_gbps=5000, l2_bytes=96 << 20,
        timing_calibrated=False, launch_us=3.0),
    "sm_121a": Target(
        name="sm_121a", sku="DGX Spark (GB10)", family="blackwell-consumer",
        notes="same instruction surface as sm_120a, separate cubin, different SFU rates",
        tma=True, tmem=False, tcgen05=False, wgmma=False, mma_sync=True, cp_async=True, clusters=True,
        mma_dtypes=("bf16", "f16", "fp8"),
        num_sms=48, clock_ghz=2.5, smem_per_sm=102400, smem_per_cta=101376, tmem_cols=0,
        max_threads_per_sm=1536, max_ctas_per_sm=24,
        tc_flops_per_clk=512, hbm_gbps=273, l2_gbps=2000, l2_bytes=24 << 20,
        timing_calibrated=False, launch_us=3.0),
}

DEFAULT_TARGET = "sm_100a"


def get_target(name) -> Target:
    """Exact target lookup.

    Cake "maps the attached GPU exactly, reports unsupported targets rather than
    silently substituting another architecture" (Sec. 2.3).
    """
    if isinstance(name, Target):
        return name
    t = TARGETS.get(str(name))
    if t is None:
        raise TargetError(f"unsupported target `{name}`",
                          hint=f"supported targets: {', '.join(TARGETS)}")
    return t


def table3() -> str:
    """Render the architecture coverage table."""
    rows = ["| Target | SKU | Tensor-core path and notable features |", "|---|---|---|"]
    for t in TARGETS.values():
        rows.append(f"| {t.name} | {t.sku} | {t.notes} |")
    return "\n".join(rows)
