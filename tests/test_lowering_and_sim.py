import pytest

from cake.errors import SimError
from cake.harness import sandbox
from cake.lir.core import LMbarExpectTx, LSet, LUmma, walk
from cake.lir.printer import to_raw_source
from cake.lower import lower
from cake.sim import device
from cake.workloads.kmeans_assign import template

from conftest import SMALL


def _run(km, b200, kernel, dims=SMALL, seed=1):
    inp = km.make_inputs(dims, "gaussian", seed)
    g = {**inp, **km.make_outputs(dims)}
    res = device.run(kernel, b200, g)
    return km.check({"OUT": g["OUT"]}, km.oracle(inp, dims), inp, dims, True), res


def _kernel(km, params, bugs=(), dims=SMALL):
    return lower(sandbox.build(template(params, bugs), km, dims, "sm_100a"))


def test_lowering_derives_metadata(km):
    k = _kernel(km, dict(SPLIT=True, ACC=2, STAGES=3))
    load = next(b for b in k.blocks if b.name == "load")
    inits = {op.name: op.value for op in load.body if isinstance(op, LSet)}
    assert inits["ph_c_free"] == 0b111          # released 3-slot barrier: every parity starts at 1
    tx = [op for op in walk(load.body) if isinstance(op, LMbarExpectTx)]
    assert {int(op.nbytes) for op in tx} == {128 * 128 * 2}   # one transaction group per TMA load
    assert k.tmem_cols == 256 and k.smem_bytes <= 232448


@pytest.mark.parametrize("params", [dict(), dict(SPLIT=True, ACC=2, STAGES=3),
                                    dict(SPLIT=True, ACC=2, STAGES=4, PERSISTENT=True, CN="smem", EPI_WARPS=8),
                                    dict(SPLIT=True, ACC=2, BN=256, CHUNK=64, EPI_WARPS=8)])
def test_schedules_are_numerically_correct(km, b200, params):
    cr, _ = _run(km, b200, _kernel(km, params))
    assert cr.ok, cr.summary


def test_tail_shapes_are_correct(km, b200):
    dims = {"B": 1, "N": 300, "K": 300, "D": 128}
    cr, _ = _run(km, b200, _kernel(km, dict(SPLIT=True, ACC=2), dims=dims), dims=dims)
    assert cr.ok, cr.summary


def test_raw_round_trip_is_shape_generic(km, b200):
    raw = to_raw_source(_kernel(km, dict(SPLIT=True, ACC=2), dims=km.target_dims), "kmeans_assign")
    assert "k.cdiv(N, 128)" in raw
    k = sandbox.build(raw, km, SMALL, "sm_100a", kind="raw")
    cr, _ = _run(km, b200, k)
    assert cr.ok


def test_missing_release_hangs(km, b200):
    with pytest.raises(SimError) as e:
        _run(km, b200, _kernel(km, dict(SPLIT=True), ("no_release",)))
    assert e.value.fault == "hang" and e.value.detail["blocked"]


def test_missing_wait_reads_stale_data(km, b200):
    cr, res = _run(km, b200, _kernel(km, dict(SPLIT=True), ("missing_wait",)))
    assert not cr.ok
    assert any(n["kind"] in ("STALE_SMEM_READ", "UNINIT_SMEM_READ") for n in res.notes)


def test_lane_quadrant_violation_corrupts_rows(km, b200):
    cr, res = _run(km, b200, _kernel(km, dict(SPLIT=True), ("lane_quadrant",)))
    assert not cr.ok
    assert any(n["kind"] == "TMEM_LANE_QUADRANT" for n in res.notes)


def test_descriptor_swizzle_mismatch_gives_wrong_results(km, b200):
    k = _kernel(km, dict(SPLIT=True))
    for b in k.blocks:
        for op in walk(b.body):
            if isinstance(op, LUmma):
                op.b.swizzle = "64B"
    cr, _ = _run(km, b200, k)
    assert not cr.ok


def test_misaligned_tma_destination_faults(km, b200):
    k = _kernel(km, dict(SPLIT=True))
    for b in k.blocks:
        for op in walk(b.body):
            if op.kind == "tma_load" and op.tensor == "C":
                op.dst = op.dst + 16
    with pytest.raises(SimError) as e:
        _run(km, b200, k)
    assert e.value.fault == "misaligned"


def test_timing_rewards_specialization_and_occupancy(km, b200):
    base = device.measure(_kernel(km, {}, dims=km.target_dims), b200).median_us
    spec = device.measure(_kernel(km, dict(SPLIT=True), dims=km.target_dims), b200).median_us
    deep = device.measure(_kernel(km, dict(SPLIT=True, ACC=2, STAGES=4), dims=km.target_dims), b200)
    assert spec < base                         # MMA / epilogue overlap helps
    assert deep.ctas_per_sm == 1 and deep.median_us > spec   # deep rings cost co-residency
