import pytest

from cake.corpus import FIG3_FMHA_FRAGMENT, build_entry, corpus
from cake.errors import CakeTypeError, ConstructionError, SandboxError
from cake.harness import sandbox
from cake.workloads.kmeans_assign import BASELINE_SOURCE

from conftest import SMALL

HEADER = '''@cake.schedule()
def k(lm, X: LM.tma3d, C: LM.tma3d, CN: LM.gmem, OUT: LM.gmem):
    B, N, K, D = lm.dims("B", "N", "K", "D")
    lm.launch(grid=(1, 1), warps=6)
    pool = lm.smem(65536)
    sx = pool.view(offset=0, shape=(128, 128), dtype=lm.bf16)
    load = lm.role(warps=[0])
    epi = lm.role(warps=[2, 3, 4, 5])
    bar = lm.barrier(prod=[load], cons=[epi])
'''


def test_baseline_constructs(km):
    prog = sandbox.build(BASELINE_SOURCE, km, SMALL, "sm_100a")
    assert [r.name for r in prog.roles] == ["load", "comp"]
    assert prog.grid == (4, 2, 1)


def test_figure3_fragment_constructs():
    e = next(x for x in corpus() if x.name == "fig3_fmha_fragment")
    prog = build_entry(e)
    assert {b.name for b in prog.barriers} == {"q_full"}
    assert "lm.fence_proxy()" in FIG3_FMHA_FRAGMENT


def test_type_error_points_at_the_line(km):
    src = HEADER + '''    with load:
        r = lm.regs((128, 128), lm.f32)
        cn = lm.regs((128,), lm.f32)
        lm.ewise(r, "add", r, cn)
'''
    with pytest.raises(CakeTypeError) as e:
        sandbox.build(src, km, SMALL, "sm_100a")
    assert "ambiguous broadcast" in e.value.message
    assert e.value.loc.lineno == 13 and "lm.ewise" in e.value.loc.text


def test_registers_are_private_to_a_role(km):
    src = HEADER + '''    with load:
        r = lm.regs((128,), lm.f32, init=0.0)
    with epi:
        lm.fill(r, 1.0)
'''
    with pytest.raises(CakeTypeError, match="belongs to role `load`"):
        sandbox.build(src, km, SMALL, "sm_100a")


def test_tma_cannot_convert_types(km):
    src = HEADER + '''    sc = pool.view(offset=32768, shape=(128, 128), dtype=lm.f16)
    with load:
        sc.tma_load(C, coords=(0, 0, 0), barrier=bar)
'''
    with pytest.raises(CakeTypeError, match="cannot convert"):
        sandbox.build(src, km, SMALL, "sm_100a")


def test_symbolic_python_if_is_rejected(km):
    src = HEADER + '''    with load:
        for j in lm.range(0, 4):
            if j == 0:
                lm.arrive(bar)
'''
    with pytest.raises(ConstructionError, match="Python bool"):
        sandbox.build(src, km, SMALL, "sm_100a")


@pytest.mark.parametrize("snippet,msg", [
    ("import os\n", "Import"),
    ("x = (1).__class__\n", "private attribute"),
    ("while True:\n    pass\n", "While"),
    ("f = lambda: 1\n", "Lambda"),
])
def test_sandbox_rejects_unsafe_python(km, snippet, msg):
    with pytest.raises(SandboxError, match=msg):
        sandbox.build(snippet + BASELINE_SOURCE, km, SMALL, "sm_100a")
