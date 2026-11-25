"""Prompts for the kernel-evolution agent and parsing of its candidates."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

CAKE_REFERENCE = r"""
# Cake IR reference

A Cake program is one Python function decorated with `@cake.schedule()`. Its first parameter `lm` is
the builder; the remaining parameters are the contract's tensors annotated `LM.tma2d` / `LM.tma3d`
(a TMA descriptor exists) or `LM.gmem` (plain global pointer). `cake` and `LM` are in scope; imports,
`while`, `try`, classes and names starting with `_` are not allowed.

A program DECLARES resources and warp roles, then writes each role's instruction stream inside
`with role:`. Lowering derives shared-memory addresses, mbarrier phase/parity bookkeeping, TMA
transaction byte counts, TMEM column numbers, MMA descriptors and warp predicates for you.

## Static configuration
- `B, N, K, D = lm.dims("B", "N", "K", "D")`: concrete problem sizes (ints).
- `lm.launch(grid=(gx, gy[, gz]), warps=W)`: grid and warps per CTA (W <= 32).
- `lm.block_idx(axis)` (symbolic), `lm.num_sms()`, `lm.cdiv(a, b)`, `lm.min`, `lm.max`, `lm.select(c, a, b)`.

## Resources (declare outside role blocks)
- `pool = lm.smem(nbytes)`; `v = pool.view(offset, shape=(rows, cols), dtype=lm.bf16, stage=S, swizzle="128B")`:
  an S-deep ring of 2-D tiles; stage i starts at `offset + i * rows*cols*elem_bytes`. You choose byte
  offsets: keep views disjoint and inside the pool, aligned to the swizzle atom (1024 B for "128B",
  512 B "64B", 256 B "32B", 128 B "none"). Rows must be a multiple of the swizzle span in bytes.
  Address stages with `v[stage]` (stage may be symbolic).
- `acc = lm.tmem(cols=c0, width=W, shape=(128, W), dtype=lm.f32, stage=S)`: tensor-memory accumulator
  stages occupying columns [c0 + i*W, c0 + (i+1)*W); 512 columns exist on sm_100a.
  `acc[s].part(start, width)` selects a column sub-range.
- `r = lm.role(warps=[...])`: a warp role; its warps execute the role's block as ONE instruction stream.
- `p = lm.pipeline(stages=S)`.
- `bar = lm.barrier(count=S, prod=[roles], cons=[roles], init_count=1, pipeline=p, released=False)`:
  S mbarrier slots. A phase of a slot completes after `init_count` arrivals and once all TMA bytes of
  its transaction group landed. `released=True` starts every slot released (the first wait on each
  slot passes immediately): use it for "buffer slot is free" barriers that producers wait on.

## Roles and control flow
- `with r:` makes the following operations belong to role r (no nesting).
- `for j in lm.range(start, stop[, step]):` is a traced runtime loop (j is symbolic). Plain Python
  `for i in range(n)` loops are unrolled at construction.
- `t = lm.regs(shape, dtype, init=None)`: a register tile private to the current role (declare it
  inside the role; `init=` emits a fill).

## Synchronization
- `lm.wait(bar, stage=s)`: wait for the next phase of slot s (per-role phase tracking is derived).
- `lm.arrive(bar, stage=s)`: one arrival on slot s.
- `lm.commit(bar, stage=s)`: tcgen05.commit - arrive once every MMA this role issued so far has completed.
- `lm.fence_proxy()`: order this role's generic shared-memory stores before async-proxy (MMA/TMA) reads.

## Data movement
- `v.tma_load(T, coords=(...), stage=s, barrier=bar)`: asynchronous TMA copy of one tile of tensor T
  (coordinates in numpy index order; the tile spans the LAST two dims) into `v[s]`; completion is
  signalled on slot s of `bar` (`slot=` overrides). Consecutive tma_loads that signal the same slot form
  one transaction group (one arrival). Out-of-range elements are zero-filled.
  `v.async_load(...)` is the cp.async variant (sm_80/sm_89).
- `lm.load(t, T, coords)`: synchronous global -> register load (out-of-range elements read 0); its
  latency is exposed on the role's critical path.
- `lm.store(T, coords, t, valid=n)`: store the first `valid` rows of register tile t (must stay in bounds).
- `lm.tmem_load(t, acc[s])` (or a `.part(...)`): tcgen05.ld TMEM -> registers. Warp w can only reach TMEM
  lanes 32*(w % 4) .. 32*(w % 4)+31, so a role reading 128 lanes needs warps covering every w % 4.
- `lm.st_smem(v[s], t)` / `lm.ld_smem(t, v[s])`: generic shared-memory store / load ((n,) tiles match (1, n) views).

## Compute
- `lm.mma(acc[s], a, b, init=True, b_major="K")`: tensor-core MMA, acc = (init ? 0 : acc) + A @ B^T with
  A (M, K) and B (N, K) shared-memory tiles (b_major="N": B is (K, N)). A TMEM accumulator issues
  tcgen05.mma (ASYNCHRONOUS: signal completion with lm.commit); a register accumulator issues a
  synchronous warp-level MMA. tcgen05 shapes: M in {64, 128}, N a multiple of 16 in [16, 256].
- `lm.fill(t, value)`; `lm.ewise(dst, op, a, b=None, c=None)`, op in copy/neg/abs/add/sub/mul/min/max/fma
  (fma = a*b + c); operands: tiles of dst's shape, scalars, or broadcasts `v.row()` (one value per column)
  / `v.col()` (one value per row).
- `lm.reduce(dst, src, op, idx=None, index_offset=0, valid=None)`: row reduction of (M, N) `src`
  ACCUMULATED into (M,) `dst` (initialize it first); op in sum/min/max/argmin/argmax; argmin/argmax also
  update `idx` (i32) with column + index_offset (strictly better wins); `valid` masks columns >= valid.
- `lm.convert(dst, src)`: type conversion (rounds to dst precision).
"""

RAW_REFERENCE = r"""
# Direct kernel API reference (CUDA/PTX level)

A kernel is one Python function decorated with `@cake.raw_kernel()`; its first parameter `k` is the
builder, the others are tensors annotated `LM.tma2d` / `LM.tma3d` / `LM.gmem`. You manage shared-memory
byte addresses, mbarrier objects and their phase parities, transaction byte counts, TMEM columns, MMA
descriptors and warp predicates yourself. Imports, `while`, `try` and `_names` are not allowed.

## Configuration
- `B, N, K, D = k.dims("B", "N", "K", "D")`; `k.launch(grid=(...), warps=W, smem_bytes=S, tmem_cols=T)`
  (T: power of two in [32, 512] when TMEM is used, else 0).
- `k.block_idx(axis)` (symbolic), `k.num_sms()`, `k.cdiv`, `k.min`, `k.max`, `k.align(x, a)`, `k.select`.
- `with k.prologue():` runs once (warp 0) before a CTA-wide barrier: initialize mbarriers here.
- `with k.warps([ids], "name"):` code executed by exactly these warps (one instruction stream).
- `t = k.regs(shape, k.f32, name="t")`: register tile of the current warp block.
- `k.set("name", expr)` / `k.reg("name")`: scalar registers (e.g. phase bits). `for i in k.range(a, b[, s]):`.

## mbarriers (8-byte objects in shared memory, 8-byte aligned, must not overlap data)
- `k.mbar_init(addr, count)`: a phase completes after `count` arrivals and zero pending tx bytes.
- `k.mbar_arrive(addr)`; `k.mbar_expect_tx(addr, nbytes)`: one arrival + nbytes expected (TMA completions
  subtract their bytes).
- `k.mbar_wait(addr, parity)`: returns once the phase with that parity completed. A fresh barrier is in
  phase 0: wait(addr, 0) blocks until the first completion, wait(addr, 1) passes immediately. Keep one
  parity bit per slot and flip it after every wait.
- `k.fence_proxy_async()`: order generic smem stores before async-proxy reads.

## Memory
- `k.tma_load(dst_addr, T, coords, box=(rows, cols), swizzle, mbar_addr)`: async bulk tensor copy of a
  box from the last two dims of T at `coords`; completes `rows*cols*elem` tx bytes on the mbarrier.
  dst must be aligned to the swizzle atom (1024 B for "128B", 128 B for "none"); box dims <= 256,
  inner dim a multiple of 16 bytes. Out-of-range elements are zero-filled. `k.cp_async(...)` on sm_80/89.
- `k.ld_global(t, T, coords)` (out-of-range reads 0), `k.st_global(T, coords, t, valid=None)` (stores
  must be in bounds), `k.ld_shared(t, addr, dtype, swizzle)`, `k.st_shared(addr, t, dtype, swizzle)`.
- `k.tmem_load(t, col, lanes, ncols)`: tcgen05.ld; warp w accesses TMEM lanes 32*(w%4)..+31 only.

## Tensor cores
- `d = k.smem_desc(addr, rows, cols, dtype, swizzle)`: the swizzle MUST equal the one the tile was written with.
- `k.umma(tmem_col, a_desc, b_desc, M, N, K, accumulate, b_major="K")`: tcgen05.mma (asynchronous),
  TMEM[:, col:col+N] (+)= A @ B^T; M in {64,128}, N multiple of 16 <= 256, K multiple of 16.
- `k.umma_commit(mbar_addr)`: arrive on the mbarrier once all previously issued UMMAs of this block complete.
- `k.mma_sync(acc_regs, a_desc, b_desc, M, N, K, accumulate)`: synchronous warp-level MMA.

## CUDA-core operations
- `k.fill(t, v)`, `k.ewise(dst, op, a, b, c)` (copy/neg/abs/add/sub/mul/min/max/fma; `t.row()` / `t.col()`
  broadcast a vector), `k.reduce(dst, src, op, idx=None, index_offset=0, valid=None)` (row reduction
  accumulated into dst; argmin/argmax update idx), `k.convert(dst, src)`.
"""

ROLE = {
    "cake": ("You are an expert GPU kernel engineer optimizing a kernel for a simulated NVIDIA B200-class GPU. "
             "You write kernels in Cake IR, a typed, hardware-explicit schedule language. The compiler harness "
             "verifies each schedule before it runs, estimates its cost, and returns localized diagnostics."),
    "raw": ("You are an expert GPU kernel engineer optimizing a kernel for a simulated NVIDIA B200-class GPU. "
            "You write kernels directly at the CUDA/PTX level through a thin Python API (shared-memory "
            "addresses, mbarriers, TMA, tcgen05/TMEM). The environment compiles, runs and times them."),
}

PROTOCOL = {
    "cake": ("Each candidate is checked in this order: IR construction (typing), verifier gates (deadlock, "
             "races, hardware contracts, data-flow consistency), cost-model ranking, then the best-ranked "
             "candidates run on the device: numerical validation on every validation case, then timing at "
             "the target shape. You see verifier findings, cost-model bottleneck attribution, sanitizer "
             "evidence for runtime failures, timings and a profile."),
    "raw": ("Each candidate is compiled, run on every validation case and, if correct, timed at the target "
            "shape. You see compiler errors, runtime errors, correctness outcomes, timings and unit "
            "utilization."),
}

OUTPUT_FORMAT = """
## Output format
Reply with {n} candidate(s). For each candidate write a heading line
`### Candidate <i>: <one-line description of the schedule idea>`
followed by ONE fenced ```python block containing the complete program (decorator included).
Candidates should differ structurally (roles, pipelining depth, buffering, tile shapes, persistence),
not only in constants. Keep explanations short.
"""


@dataclass
class GenContext:
    """Everything the generator sees for one round (real LLMs get it as text; the scripted agent uses fields)."""
    arm: str
    workload: Any
    n: int
    round: int
    parents: List[Any] = field(default_factory=list)       # EvalRecords
    failures: List[Any] = field(default_factory=list)      # EvalRecords
    feedback: Dict[str, str] = field(default_factory=dict)  # record id -> feedback text
    tactics: List[Dict[str, Any]] = field(default_factory=list)
    baseline_us: float = 0.0
    rng: Any = None
    harness: Any = None


def system_prompt(workload, target_summary: str, arm: str, n: int, baseline_source: str) -> str:
    ref = CAKE_REFERENCE if arm == "cake" else RAW_REFERENCE
    return "\n".join([
        ROLE[arm], "", ref.strip(), "", "# Target", target_summary, "", "# Task contract", workload.contract_text(),
        "", "# Evaluation protocol", PROTOCOL[arm], "", "# Baseline implementation", "```python",
        baseline_source.strip(), "```", OUTPUT_FORMAT.format(n=n)])


def _trim(src: str, limit: int = 9000) -> str:
    return src if len(src) <= limit else src[:limit] + "\n# ... (truncated)"


def user_prompt(ctx: GenContext) -> str:
    L: List[str] = [f"# Round {ctx.round}"]
    if ctx.parents:
        L.append("## Best programs so far")
        for p in ctx.parents:
            sp = f"{p.speedup:.3f}x baseline" if p.speedup else "baseline"
            L.append(f"### {p.id} ({sp}): {p.meta.get('idea', '')}")
            L.append("```python")
            L.append(_trim(p.source.strip()))
            L.append("```")
            fb = ctx.feedback.get(p.id)
            if fb:
                L.append("Feedback:")
                L.append(fb)
    if ctx.failures:
        L.append("## Recent failed candidates")
        for f in ctx.failures:
            L.append(f"- {f.id} ({f.meta.get('idea', '')}):")
            L.append("  " + ctx.feedback.get(f.id, "failed").replace("\n", "\n  "))
    if ctx.tactics:
        L.append("## Tactics that worked before")
        for t in ctx.tactics[:8]:
            L.append(f"- {t['text']}")
    L.append("## Task")
    L.append(f"Propose {ctx.n} new candidate(s) that are faster than the best program while staying correct. "
             f"Fix the reported problems first when a promising idea failed.")
    return "\n".join(L)


_HEAD = re.compile(r"^#{2,4}\s*Candidate\s*\d*\s*[:\-.]?\s*(.*)$", re.M)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse_candidates(text: str, limit: int) -> List[Tuple[str, str]]:
    """Extract (idea, source) pairs from a model response."""
    out: List[Tuple[str, str]] = []
    for m in _BLOCK.finditer(text):
        src = m.group(1)
        if "@cake." not in src:
            continue
        head = None
        for h in _HEAD.finditer(text, 0, m.start()):
            head = h
        idea = head.group(1).strip() if head else ""
        out.append((idea[:160], src))
        if len(out) >= limit:
            break
    return out
