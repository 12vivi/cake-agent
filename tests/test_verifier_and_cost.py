import numpy as np
import pytest

from cake.corpus import build_entry, corpus
from cake.cost import estimate
from cake.cost.calibrate import calibrate, metrics
from cake.cost.model import Calibration, featurize
from cake.harness import sandbox
from cake.verify import LIBRARY, SEED, verify
from cake.workloads.kmeans_assign import template

from conftest import TAIL


def _gates(km, params, bugs=(), dims=None, extra=()):
    prog = sandbox.build(template(params, bugs), km, dims or km.target_dims, "sm_100a")
    rep = verify(prog, extra_rules=list(extra))
    return {f.rule for f in rep.gates}, rep


def test_baseline_is_clean_with_guidance(km):
    gates, rep = _gates(km, {})
    assert not gates
    assert {"HINT.SPECIALIZE", "HINT.SYNC_LOAD"} <= {f.rule for f in rep.findings}
    assert rep.coverage   # analysis coverage is stated explicitly


@pytest.mark.parametrize("bug,rule", [
    ("no_release", "SAFE.DEADLOCK"), ("no_commit_free", "SAFE.DEADLOCK"), ("stage_mod", "SAFE.DEADLOCK"),
    ("missing_wait", "SAFE.RACE"), ("early_release", "SAFE.RACE"), ("smem_overlap", "SAFE.VIEW_BOUNDS"),
])
def test_bugs_are_localized_by_the_seed_verifier(km, bug, rule):
    gates, rep = _gates(km, dict(SPLIT=True, ACC=2, STAGES=3), (bug,))
    assert rule in gates
    f = next(f for f in rep.gates if f.rule == rule)
    assert f.loc is not None and f.loc.lineno > 0


def test_deadlock_hint_names_the_fix(km):
    _, rep = _gates(km, dict(SPLIT=True), ("no_release",))
    f = next(f for f in rep.gates if f.rule == "SAFE.DEADLOCK")
    assert "released=True" in f.hint and "c_free" in f.message


def test_latent_rules_are_not_seed_rules(km):
    assert {"HW.TMEM_LANES", "SAFE.PROXY_FENCE", "DC.TAIL_MASK"} <= set(LIBRARY)
    assert not ({"HW.TMEM_LANES", "SAFE.PROXY_FENCE", "DC.TAIL_MASK"} & set(SEED))
    gates, _ = _gates(km, dict(SPLIT=True), ("lane_quadrant",))
    assert not gates                               # the seed verifier misses it ...
    gates, _ = _gates(km, dict(SPLIT=True), ("lane_quadrant",), extra=[LIBRARY["HW.TMEM_LANES"]])
    assert "HW.TMEM_LANES" in gates                # ... the library rule catches it


def test_tail_mask_rule_needs_a_ragged_shape(km):
    lib = [LIBRARY["DC.TAIL_MASK"]]
    assert "DC.TAIL_MASK" not in _gates(km, dict(SPLIT=True), ("tail_mask",), extra=lib)[0]
    assert "DC.TAIL_MASK" in _gates(km, dict(SPLIT=True), ("tail_mask",), dims=TAIL, extra=lib)[0]
    assert "DC.TAIL_MASK" not in _gates(km, dict(SPLIT=True), (), dims=TAIL, extra=lib)[0]


def test_corpus_is_clean_under_seed_and_library_rules():
    for e in corpus():
        rep = verify(build_entry(e), extra_rules=list(LIBRARY.values()))
        assert not rep.blocked, (e.name, [f.format() for f in rep.gates])


def test_hardware_conformance_on_other_targets(km):
    prog = sandbox.build(template({}), km, km.target_dims, "sm_120a")
    rules = {f.rule for f in verify(prog).gates}
    assert "HW.TMEM" in rules and "HW.MMA" in rules


def test_cost_model_attribution_and_coverage(km):
    est = estimate(sandbox.build(template({}), km, km.target_dims, "sm_100a"))
    assert est.covered and est.us > 0 and est.bottleneck == "serialized MMA + epilogue"
    est2 = estimate(sandbox.build(template({}), km, km.target_dims, "sm_120a"))
    assert not est2.covered and "coverage" in est2.reason


def test_calibration_recovers_scaled_terms(km):
    rng = np.random.default_rng(0)
    truth = dict(Calibration("sm_100a").coef, tc=1.6, alu=1.4)
    from cake.cost.model import predict
    samples = []
    for params in [{}, dict(STAGES=1), dict(SPLIT=True), dict(BN=64), dict(SPLIT=True, ACC=2),
                   dict(EPI_WARPS=8), dict(FMA=False), dict(SPLIT=True, CN="smem"), dict(BN=256, CHUNK=64),
                   dict(STAGES=3), dict(SWIZZLE="64B"), dict(SPLIT=True, ACC=2, STAGES=1), dict(BN=64, STAGES=4),
                   dict(SPLIT=True, EPI_WARPS=8), dict(CN="smem"), dict(SPLIT=True, BN=64)]:
        f = featurize(sandbox.build(template(params), km, km.target_dims, "sm_100a"))
        samples.append((f, predict(f, truth)["us"] * float(np.exp(rng.normal(0, 0.01)))))
    new, rep, ok = calibrate(samples, Calibration("sm_100a"))
    assert ok and rep["after"]["mae_log"] < rep["before"]["mae_log"]
    assert metrics(samples, new.coef)["mae_log"] < 0.05
