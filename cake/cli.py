"""Command-line interface: ``python -m cake <command>`` (or the ``cake`` script)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _dims(spec: Optional[str], default: Dict[str, int]) -> Dict[str, int]:
    d = dict(default)
    if spec:
        for part in spec.split(","):
            k, v = part.split("=")
            d[k.strip()] = int(v)
    return d


def _src(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_targets(a) -> int:
    from .targets import table3
    print(table3())
    return 0


def cmd_verify(a) -> int:
    from .harness import sandbox
    from .verify import verify
    from .workloads import get_workload
    from .evolution.state import HarnessState
    w = get_workload(a.workload)
    prog = sandbox.build(_src(a.file), w, _dims(a.dims, w.target_dims), a.target or w.target)
    state = HarnessState.load(a.state) if a.state else HarnessState()
    rep = verify(prog, a.target or w.target, rules=state.rules())
    print(rep.format(max_items=50))
    return 1 if rep.blocked else 0


def cmd_lower(a) -> int:
    from .harness import sandbox
    from .lir.printer import to_raw_source
    from .lower import lower
    from .workloads import get_workload
    w = get_workload(a.workload)
    prog = sandbox.build(_src(a.file), w, _dims(a.dims, w.target_dims), a.target or w.target)
    print(to_raw_source(lower(prog), prog.name))
    return 0


def cmd_run(a) -> int:
    from .harness.evaluate import Harness
    from .workloads import get_workload
    from .evolution.state import HarnessState
    w = get_workload(a.workload)
    state = HarnessState.load(a.state) if a.state else HarnessState()
    h = Harness(w, a.target, state, arm=a.arm)
    rec = h.evaluate(_src(a.file))
    print(f"stage: {rec.stage}  ok: {rec.ok}")
    print(h.feedback(rec))
    return 0 if rec.ok else 1


def cmd_evolve(a) -> int:
    from .agent.evolve import EvolveConfig, evolve
    from .evolution.state import HarnessState
    state = HarnessState.load(a.state) if a.state else None
    res = evolve(EvolveConfig(workload=a.workload, arm=a.arm, llm=a.llm, target=a.target,
                              budget_tokens=int(float(a.budget)), candidates_per_round=a.candidates,
                              device_evals_per_round=a.device_evals, seed=a.seed, out_dir=a.out, state=state,
                              co_evolve=a.co_evolve, evolve_every=a.evolve_every, auto_approve=not a.ask))
    if state is not None and state.root:
        state.save()
    print(json.dumps(res.summary, indent=2, default=str))
    return 0


def cmd_harness(a) -> int:
    """Run one harness-evolution step on the evidence of a finished run."""
    from .agent.archive import Archive
    from .agent.llm import make_llm
    from .evolution.loop import harness_step
    from .evolution.state import HarnessState
    from .harness.evaluate import EvalRecord, Harness
    from .workloads import get_workload
    run = Path(a.run)
    arch = Archive(None)
    for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        arch.records.append(EvalRecord.from_dict(json.loads(line)))
    for line in (run / "evidence.jsonl").read_text(encoding="utf-8").splitlines():
        arch.evidence.append(json.loads(line))
    state = HarnessState.load(a.state)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    h = Harness(get_workload(summary["workload"]), summary.get("target"), state, arm="cake")
    rep = harness_step(h, arch, llm=make_llm(a.llm), auto_approve=not a.ask)
    state.save()
    print(json.dumps(rep, indent=2, default=str))
    return 0


def cmd_corpus(a) -> int:
    from .corpus import build_entry, corpus
    from .lower import lower
    from .sim import device
    from .targets import get_target
    from .verify import verify
    from .workloads import get_workload
    bad = 0
    for e in corpus():
        prog = build_entry(e)
        t = get_target(e.target)
        rep = verify(prog, t)
        status = "verified" if not rep.blocked else "REJECTED " + ",".join(f.rule for f in rep.gates)
        if e.run and not rep.blocked:
            w = get_workload(e.workload)
            inp = w.make_inputs(e.dims, "gaussian", 7)
            g = {**inp, **w.make_outputs(e.dims)}
            device.run(lower(prog), t, g)
            cr = w.check({x.name: g[x.name] for x in w.tensors if x.role == "output"}, w.oracle(inp, e.dims), inp,
                         e.dims, True)
            status += "; numerics OK" if cr.ok else f"; numerics FAIL ({cr.summary})"
            bad += 0 if cr.ok else 1
        bad += 1 if rep.blocked else 0
        print(f"{e.name:32s} {status}")
    return 1 if bad else 0


def cmd_dispatch(a) -> int:
    from .dispatch.portfolio import kmeans_domain, run_dispatch
    from .workloads import kmeans_assign as km
    seeds = [("stages1", km.template(dict(STAGES=1))), ("specialized", km.template(dict(SPLIT=True))),
             ("bn64_8warps", km.template(dict(BN=64, EPI_WARPS=8, SWIZZLE="64B"))),
             ("persistent_smem", km.template(dict(SPLIT=True, ACC=2, STAGES=4, PERSISTENT=True, CN="smem",
                                                  EPI_WARPS=8)))]
    for path in a.seed or []:
        seeds.append((Path(path).stem, _src(path)))
    dom = kmeans_domain()
    if a.quick:
        dom = [s for s in dom if s["N"] != 65536]
    rep = run_dispatch(seeds, domain=dom)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "dispatch.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("manifest", "routes", "tuning", "held_out", "fallback_check", "wall_s")},
                     indent=2, default=str))
    return 0


def cmd_experiment(a) -> int:
    if a.which == "clean-start":
        from .experiments.clean_start import run
        rep = run(workload=a.workload, runs=a.runs, budget_tokens=int(float(a.budget)), llm=a.llm, out_dir=a.out)
        print(json.dumps(rep["table"], indent=2))
    else:
        from .experiments.coevolution import run
        rep = run(runs=a.runs, budget_tokens=int(float(a.budget)), llm=a.llm, out_dir=a.out, workload=a.workload)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="cake", description="Cake: compiler-agent co-design (unofficial reproduction)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("targets", help="print the architecture coverage table")
    s.set_defaults(fn=cmd_targets)

    for name, fn, hlp in (("verify", cmd_verify, "run the verifier on a Cake IR file"),
                          ("lower", cmd_lower, "print the lowered (direct-code) form of a Cake IR file"),
                          ("run", cmd_run, "evaluate a program through the full harness")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("file")
        s.add_argument("--workload", default="kmeans_assign")
        s.add_argument("--dims", help="e.g. B=2,N=1024,K=1024,D=128 (default: target shape)")
        s.add_argument("--target")
        s.add_argument("--state", help="harness-state directory (evolved rules, calibration)")
        if name == "run":
            s.add_argument("--arm", choices=["cake", "raw"], default="cake")
        s.set_defaults(fn=fn)

    s = sub.add_parser("evolve", help="run the kernel-evolution agent")
    s.add_argument("--workload", default="kmeans_assign")
    s.add_argument("--arm", choices=["cake", "raw"], default="cake")
    s.add_argument("--llm", default="scripted", help="scripted | anthropic[:model] | openai:<model>")
    s.add_argument("--target")
    s.add_argument("--budget", default="2e6", help="token budget")
    s.add_argument("--candidates", type=int, default=4)
    s.add_argument("--device-evals", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="runs/evolve")
    s.add_argument("--state", help="harness-state directory to load/save")
    s.add_argument("--co-evolve", action="store_true", help="evolve the harness during the run")
    s.add_argument("--evolve-every", type=int, default=8)
    s.add_argument("--ask", action="store_true", help="human merge gate for harness changes")
    s.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("harness-evolve", help="one harness-evolution step from a run's evidence")
    s.add_argument("--run", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--llm", default="scripted")
    s.add_argument("--ask", action="store_true")
    s.set_defaults(fn=cmd_harness)

    s = sub.add_parser("corpus", help="validate the kernel corpus (the P6 test gate)")
    s.set_defaults(fn=cmd_corpus)

    s = sub.add_parser("dispatch", help="build and validate a dispatcher-backed portfolio (Sec. 6)")
    s.add_argument("--seed", action="append", help="extra seed program (Cake IR file)")
    s.add_argument("--quick", action="store_true", help="skip the largest shapes")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_dispatch)

    s = sub.add_parser("experiment", help="reproduce the paper's experiments (scaled)")
    s.add_argument("which", choices=["clean-start", "coevolution"])
    s.add_argument("--workload", default="kmeans_assign")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--budget", default="2e6")
    s.add_argument("--llm", default="scripted")
    s.add_argument("--out", default="runs/experiment")
    s.set_defaults(fn=cmd_experiment)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
