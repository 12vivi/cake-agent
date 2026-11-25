import pytest

from cake.agent.evolve import EvolveConfig, evolve, route
from cake.agent.prompts import parse_candidates
from cake.dispatch.portfolio import Evaluator, build_portfolio, split, validate
from cake.errors import SandboxError
from cake.evolution.loop import _library_rule, gate_rule
from cake.evolution.state import HarnessState, compile_rule
from cake.harness.evaluate import Harness
from cake.verify.rules import LIBRARY
from cake.workloads.kmeans_assign import template


def test_parse_candidates():
    text = ("### Candidate 1: split roles\n```python\n@cake.schedule()\ndef a(lm): pass\n```\n"
            "noise\n### Candidate 2: deeper ring\n```python\n@cake.schedule()\ndef b(lm): pass\n```\n"
            "```python\nprint('no decorator')\n```")
    c = parse_candidates(text, 5)
    assert [i for i, _ in c] == ["split roles", "deeper ring"]


@pytest.mark.parametrize("arm", ["cake", "raw"])
def test_scripted_evolution_runs_end_to_end(tmp_path, arm):
    res = evolve(EvolveConfig(arm=arm, budget_tokens=120_000, seed=5, verbose=False, out_dir=str(tmp_path)))
    s = res.summary
    # the direct arm's printed baseline is the same schedule; it differs only by measurement noise
    assert s["candidates"] > 0 and s["best_speedup"] >= 0.95 and s["tokens_used"] >= 120_000
    assert (tmp_path / "records.jsonl").exists() and (tmp_path / "summary.json").exists()
    assert res.archive.checkpoints[0]["tokens"] == 0
    if arm == "cake":
        assert s["verifier_rejects"] + s["device_evals"] + s["ranked_out"] + s["construct_errors"] == s["candidates"]


def test_cake_feedback_is_localized_raw_feedback_is_coarse(km):
    src = template(dict(SPLIT=True), ("no_release",))
    cake = Harness(km, arm="cake")
    rec = cake.evaluate(src)
    assert rec.stage == "verify" and "SAFE.DEADLOCK" in cake.feedback(rec) and "line" in cake.feedback(rec)
    from cake.harness import sandbox
    from cake.lir.printer import to_raw_source
    from cake.lower import lower
    raw_src = to_raw_source(lower(sandbox.build(src, km, km.target_dims, "sm_100a")), "kmeans_assign")
    raw = Harness(km, arm="raw")
    rr = raw.evaluate(raw_src)
    assert rr.stage == "runtime" and raw.feedback(rr).startswith("kernel timed out")


def test_routing_sends_verifier_gaps_and_calibration_samples(km):
    h = Harness(km, arm="cake")
    bad = h.evaluate(template(dict(SPLIT=True), ("lane_quadrant",)))
    good = h.evaluate(template(dict(SPLIT=True)))
    kinds = {e["kind"] for e in route(bad, 1.0)} | {e["kind"] for e in route(good, 1.0)}
    assert {"verifier_gap", "calibration_sample", "tactic"} <= kinds


def test_rule_gating_rejects_overbroad_and_accepts_precise(km):
    h = Harness(km, arm="cake")
    failing = [template(dict(SPLIT=True), ("lane_quadrant",)), template(dict(STAGES=1), ("lane_quadrant",))]
    passing = [template(dict(SPLIT=True))]
    strict = gate_rule(_library_rule("HW.TMEM_LANES_STRICT"), failing, h, passing)
    assert not strict["passed"] and strict["corpus_false_positives"]
    precise = gate_rule(LIBRARY["HW.TMEM_LANES"], failing, h, passing)
    assert precise["passed"] and precise["recall"] == 1.0


def test_evolved_rule_code_is_sandboxed():
    with pytest.raises(SandboxError):
        compile_rule("X", "program_safety", "gate", "", "import os\ndef check(ctx):\n    yield 1\n")
    r = compile_rule("X", "program_safety", "gate", "", "def check(ctx):\n    return []\n")
    assert r.origin == "evolved"


def test_harness_state_round_trip(tmp_path):
    st = HarnessState(root=str(tmp_path))
    st.library_enabled.append("HW.TMEM_LANES")
    st.evolved["EVOLVED.T"] = {"category": "program_safety", "disposition": "gate", "description": "t",
                               "source": "def check(ctx):\n    return []\n"}
    st.record("verifier_rule", rule="HW.TMEM_LANES")
    st.save()
    st2 = HarnessState.load(str(tmp_path))
    ids = st2.rule_ids()
    assert "HW.TMEM_LANES" in ids and "EVOLVED.T" in ids and st2.version == 1


def test_dispatch_split_is_deterministic_and_validated():
    dom = [{"B": 1, "N": n, "K": k, "D": 128} for n in (1000, 2048) for k in (256, 1000)]
    t1, h1 = split(dom, 0.5)
    t2, h2 = split(dom, 0.5)
    assert t1 == t2 and h1 == h2 and not ({str(x) for x in t1} & {str(x) for x in h1})
    ev = Evaluator(samples=1)
    seeds = [("stages1", template(dict(STAGES=1))), ("specialized", template(dict(SPLIT=True)))]
    pf, info = build_portfolio(seeds, dom, ev, verbose=False)
    rep = validate(pf, dom, ev)
    assert rep["incorrect"] == 0 and rep["gspan"] > 0
