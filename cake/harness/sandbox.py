"""Construct agent-authored programs inside a restricted Python subset.

Agent code is untrusted.  Before execution the source is parsed and checked
against an AST whitelist (no imports, no ``while``, no attribute names starting
with ``_``, no ``global``/``try``/``class``/``lambda``...), then executed with a
minimal builtins table and only the Cake API in scope.  This is a guard against
accidents, not a security boundary: run untrusted models in a container.
"""
from __future__ import annotations

import ast
import hashlib
import linecache
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from ..errors import CakeError, ConstructionError, Loc, SandboxError

_ALLOWED = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.With, ast.withitem, ast.If, ast.Pass, ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Attribute, ast.Call, ast.keyword, ast.Tuple, ast.List, ast.Dict, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.Subscript, ast.Slice, ast.ListComp, ast.comprehension, ast.Starred,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
    ast.BitXor, ast.BitAnd, ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.And, ast.Or, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.JoinedStr, ast.FormattedValue,
)
_INDEX = getattr(ast, "Index", None)
if _INDEX is not None:  # Python < 3.9
    _ALLOWED = _ALLOWED + (_INDEX,)

_FORBIDDEN_HINT = {
    "Import": "imports are not allowed; `cake` and `LM` are already in scope",
    "ImportFrom": "imports are not allowed; `cake` and `LM` are already in scope",
    "While": "use `for ... in lm.range(...)` (traced) or a bounded Python `for ... in range(...)`",
    "Lambda": "define helpers as plain functions",
    "Try": "exceptions cannot be caught in schedules",
    "ClassDef": "classes are not allowed",
    "Global": "global/nonlocal are not allowed",
    "Nonlocal": "global/nonlocal are not allowed",
}

MAX_RANGE = 4096


def _safe_range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise ConstructionError(f"Python range of length {len(r)} exceeds {MAX_RANGE}",
                                hint="use lm.range(...) for runtime loops; Python loops are unrolled")
    return r


SAFE_BUILTINS = {
    "range": _safe_range, "len": len, "min": min, "max": max, "abs": abs, "int": int, "float": float,
    "bool": bool, "list": list, "tuple": tuple, "enumerate": enumerate, "zip": zip, "sum": sum,
    "True": True, "False": False, "None": None, "isinstance": isinstance, "round": round,
}


def check_source(source: str, filename: str = "<candidate>") -> ast.Module:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e.msg}", loc=Loc(e.lineno or 0, e.text or "", filename)) from None
    lines = source.splitlines()

    def loc(node) -> Loc:
        ln = getattr(node, "lineno", 0)
        return Loc(ln, lines[ln - 1] if 0 < ln <= len(lines) else "", filename)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            name = type(node).__name__
            raise SandboxError(f"`{name}` is not allowed in Cake programs", loc=loc(node),
                               hint=_FORBIDDEN_HINT.get(name))
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxError(f"access to private attribute `{node.attr}` is not allowed", loc=loc(node))
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"name `{node.id}` is not allowed", loc=loc(node))
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "cake" and target.attr in ("schedule", "raw_kernel")):
                    raise SandboxError("only @cake.schedule() / @cake.raw_kernel() decorators are allowed",
                                       loc=loc(dec))
    return tree


def _namespace() -> Dict[str, Any]:
    from ..ir.builder import LM, schedule
    from ..lir.builder import raw_kernel
    cake_ns = SimpleNamespace(schedule=schedule, raw_kernel=raw_kernel, LM=LM)
    return {"__builtins__": dict(SAFE_BUILTINS), "cake": cake_ns, "LM": LM, "__name__": "candidate"}


def load(source: str, kind: str = "schedule") -> Tuple[Any, str]:
    """Execute ``source`` and return its (single) schedule or raw-kernel object and the filename."""
    digest = hashlib.sha1(source.encode()).hexdigest()[:10]
    filename = f"<candidate-{digest}>"
    tree = check_source(source, filename)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    ns = _namespace()
    try:
        exec(compile(tree, filename, "exec", dont_inherit=True), ns)  # noqa: S102 - AST-whitelisted
    except CakeError:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise ConstructionError(f"{type(e).__name__}: {e}", loc=_user_loc(e, filename)) from None
    from ..ir.builder import Schedule
    from ..lir.builder import RawKernelFn
    want = Schedule if kind == "schedule" else RawKernelFn
    objs = [v for k, v in ns.items() if isinstance(v, want)]
    if not objs:
        other = "@cake.raw_kernel()" if kind == "schedule" else "@cake.schedule()"
        deco = "@cake.schedule()" if kind == "schedule" else "@cake.raw_kernel()"
        raise ConstructionError(f"no {deco} function found in the program",
                                hint=f"this arm expects {deco}; {other} programs are not accepted here")
    if len(objs) > 1:
        raise ConstructionError(f"found {len(objs)} kernels; submit exactly one per candidate")
    obj = objs[0]
    obj.source = source
    return obj, filename


def _user_loc(e: BaseException, filename: str) -> Optional[Loc]:
    for fr in reversed(traceback.extract_tb(e.__traceback__)):
        if fr.filename == filename:
            return Loc(fr.lineno or 0, fr.line or linecache.getline(filename, fr.lineno or 0), filename)
    return None


def build(source: str, workload, dims: Dict[str, int], target: str, kind: str = "schedule"):
    """Construct a Cake IR program (``kind="schedule"``) or LIR kernel (``"raw"``) at ``dims``."""
    obj, filename = load(source, kind)
    tensors = workload.tensor_specs(dims)
    try:
        return obj.build(dims, tensors, target=target, source=source)
    except CakeError:
        raise
    except RecursionError:
        raise ConstructionError("recursion limit exceeded while constructing the program") from None
    except Exception as e:
        raise ConstructionError(f"{type(e).__name__}: {e}", loc=_user_loc(e, filename)) from None
