"""Versioned, persistent state of the compiler harness (the outer loop's object of evolution).

It records which verifier rules are active (seed + adopted library rules +
evolved rule code), per-target cost-model calibrations, the tactic library,
IR-vocabulary proposals and an auditable change history.
"""
from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cost.model import Calibration, default_calibrations
from ..errors import SandboxError
from ..verify.rules import LIBRARY, SEED, Rule

_RULE_ALLOWED = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If,
    ast.Return, ast.Pass, ast.Continue, ast.Break, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Attribute,
    ast.Call, ast.keyword, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Subscript, ast.Slice, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.JoinedStr, ast.FormattedValue, ast.Yield, ast.Starred,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
)
if getattr(ast, "Index", None) is not None:  # Python < 3.9
    _RULE_ALLOWED = _RULE_ALLOWED + (ast.Index,)


def compile_rule(rule_id: str, category: str, disposition: str, description: str, source: str) -> Rule:
    """Compile evolved rule code (a ``def check(ctx): ... yield ctx.finding(...)`` function) in a sandbox."""
    from ..ir import core as ir
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"rule `{rule_id}` has a syntax error: {e.msg} (line {e.lineno})") from None
    for node in ast.walk(tree):
        if not isinstance(node, _RULE_ALLOWED):
            raise SandboxError(f"rule `{rule_id}` uses `{type(node).__name__}`, which is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxError(f"rule `{rule_id}` accesses private attribute `{node.attr}`")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"rule `{rule_id}` uses name `{node.id}`")
    builtins = {"len": len, "range": range, "min": min, "max": max, "sorted": sorted, "set": set, "list": list,
                "tuple": tuple, "dict": dict, "any": any, "all": all, "isinstance": isinstance, "enumerate": enumerate,
                "zip": zip, "sum": sum, "abs": abs, "str": str, "int": int, "True": True, "False": False,
                "None": None}
    ns: Dict[str, Any] = {"__builtins__": builtins, "walk": ir.walk}
    for name in ("Mma", "TmaLoad", "TmemLoad", "TmemRef", "SmemRef", "Wait", "Arrive", "Commit", "Reduce", "Ewise",
                 "LoadG", "StoreG", "StSmem", "LdSmem", "Fill", "Convert", "FenceProxy", "Loop", "RegTile",
                 "SmemView", "TmemView", "Barrier", "Role"):
        ns[name] = getattr(ir, name)
    exec(compile(tree, f"<rule {rule_id}>", "exec", dont_inherit=True), ns)  # noqa: S102 - AST-whitelisted
    fn = ns.get("check")
    if not callable(fn):
        raise SandboxError(f"rule `{rule_id}` must define `def check(ctx):`")
    return Rule(rule_id, category, disposition, description, fn, "evolved", source)


@dataclass
class HarnessState:
    root: Optional[str] = None
    version: int = 0
    library_enabled: List[str] = field(default_factory=list)
    evolved: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    calibrations: Dict[str, Calibration] = field(default_factory=default_calibrations)
    tactics: List[Dict[str, Any]] = field(default_factory=list)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ rules
    def rules(self) -> List[Rule]:
        out = list(SEED.values())
        out += [LIBRARY[i] for i in self.library_enabled if i in LIBRARY]
        for rid, spec in self.evolved.items():
            try:
                out.append(compile_rule(rid, spec["category"], spec["disposition"], spec["description"],
                                        spec["source"]))
            except SandboxError:
                continue
        return out

    def rule_ids(self) -> List[str]:
        return [r.id for r in self.rules()]

    def record(self, kind: str, **info) -> None:
        self.version += 1
        self.history.append({"version": self.version, "kind": kind, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             **info})

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "library_enabled": self.library_enabled, "evolved": self.evolved,
                "calibrations": {k: v.to_dict() for k, v in self.calibrations.items()},
                "tactics": self.tactics, "proposals": self.proposals, "history": self.history}

    def save(self, root: Optional[str] = None) -> Path:
        root = root or self.root
        if root is None:
            raise ValueError("no harness-state directory")
        p = Path(root)
        p.mkdir(parents=True, exist_ok=True)
        (p / "manifest.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        rules_dir = p / "rules"
        rules_dir.mkdir(exist_ok=True)
        for rid, spec in self.evolved.items():
            (rules_dir / f"{rid.replace('.', '_')}.py").write_text(spec["source"], encoding="utf-8")
        self.root = str(p)
        return p

    @staticmethod
    def load(root: str) -> "HarnessState":
        p = Path(root) / "manifest.json"
        if not p.exists():
            return HarnessState(root=root)
        d = json.loads(p.read_text(encoding="utf-8"))
        st = HarnessState(root=root, version=d.get("version", 0), library_enabled=d.get("library_enabled", []),
                          evolved=d.get("evolved", {}), tactics=d.get("tactics", []),
                          proposals=d.get("proposals", []), history=d.get("history", []))
        cals = default_calibrations()
        for k, v in d.get("calibrations", {}).items():
            cals[k] = Calibration.from_dict(v)
        st.calibrations = cals
        return st
