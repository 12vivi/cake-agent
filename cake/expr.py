"""Symbolic scalar expressions.

Schedules are traced: ``for j in lm.range(0, nk)`` binds ``j`` to a :class:`Var`,
and arithmetic on it (``j % STAGES``, ``j * BN``) builds expression trees that are
evaluated per iteration by the verifier, the lowering and the simulator.
Constant sub-expressions fold eagerly, so purely static values stay Python ints.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, Iterable, Set

import numpy as np

from .errors import ConstructionError, capture_loc, register_api_file

register_api_file(__file__)

_BINOPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
}


def _cdiv(a, b):
    return -(-a // b)


_CALLS = {
    "min": min,
    "max": max,
    "cdiv": _cdiv,
    "align": lambda x, a: (x + a - 1) // a * a,
    "select": lambda c, a, b: a if c else b,
    "not": lambda a: not a,
}


class Expr:
    __slots__ = ()

    # --- arithmetic -------------------------------------------------------
    def __add__(self, o): return binop("+", self, o)
    def __radd__(self, o): return binop("+", o, self)
    def __sub__(self, o): return binop("-", self, o)
    def __rsub__(self, o): return binop("-", o, self)
    def __mul__(self, o): return binop("*", self, o)
    def __rmul__(self, o): return binop("*", o, self)
    def __floordiv__(self, o): return binop("//", self, o)
    def __rfloordiv__(self, o): return binop("//", o, self)
    def __mod__(self, o): return binop("%", self, o)
    def __rmod__(self, o): return binop("%", o, self)
    def __and__(self, o): return binop("&", self, o)
    def __rand__(self, o): return binop("&", o, self)
    def __or__(self, o): return binop("|", self, o)
    def __ror__(self, o): return binop("|", o, self)
    def __xor__(self, o): return binop("^", self, o)
    def __rxor__(self, o): return binop("^", o, self)
    def __lshift__(self, o): return binop("<<", self, o)
    def __rlshift__(self, o): return binop("<<", o, self)
    def __rshift__(self, o): return binop(">>", self, o)
    def __rrshift__(self, o): return binop(">>", o, self)
    def __neg__(self): return binop("-", 0, self)
    def __pos__(self): return self

    # --- comparisons build expressions -----------------------------------
    def __eq__(self, o): return binop("==", self, o)  # type: ignore[override]
    def __ne__(self, o): return binop("!=", self, o)  # type: ignore[override]
    def __lt__(self, o): return binop("<", self, o)
    def __le__(self, o): return binop("<=", self, o)
    def __gt__(self, o): return binop(">", self, o)
    def __ge__(self, o): return binop(">=", self, o)

    def __hash__(self):
        return id(self)

    def __bool__(self):
        raise ConstructionError(
            f"symbolic value `{self}` was used where a Python bool is needed",
            loc=capture_loc(),
            hint="schedule loops are traced, so Python `if`/`while` cannot branch on a "
                 "loop variable; pass the expression as an operand instead "
                 "(e.g. `init=(j == 0)`), or use lm.select(cond, a, b)")

    def __index__(self):
        raise ConstructionError(
            f"symbolic value `{self}` was used as a Python integer (size, index or range bound)",
            loc=capture_loc(),
            hint="shapes, offsets and Python-level loop bounds must be static; only "
                 "operands of Cake ops may be symbolic")

    __int__ = __index__

    def __float__(self):
        return float(self.__index__())

    def __iter__(self):
        raise ConstructionError(f"cannot iterate over symbolic value `{self}`", loc=capture_loc())

    # --- interface ------------------------------------------------------------
    def eval(self, env: Dict[str, Any]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def free_vars(self) -> Set[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def c(self) -> str:
        """Render as a C expression (used by the CUDA listing)."""
        return str(self)


class Const(Expr):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def eval(self, env):
        return self.value

    def free_vars(self):
        return set()

    def __bool__(self):
        return bool(self.value)

    def __index__(self):
        return operator.index(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return repr(self.value)

    __str__ = __repr__


class Var(Expr):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def eval(self, env):
        try:
            return env[self.name]
        except KeyError:
            raise ConstructionError(f"unbound variable `{self.name}`") from None

    def free_vars(self):
        return {self.name}

    def __repr__(self):
        return self.name

    __str__ = __repr__


class BinOp(Expr):
    __slots__ = ("op", "a", "b")

    def __init__(self, op: str, a: Expr, b: Expr):
        self.op, self.a, self.b = op, a, b

    def eval(self, env):
        return _BINOPS[self.op](self.a.eval(env), self.b.eval(env))

    def free_vars(self):
        return self.a.free_vars() | self.b.free_vars()

    def __repr__(self):
        return f"({self.a} {self.op} {self.b})"

    __str__ = __repr__

    def c(self):
        op = {"and": "&&", "or": "||", "//": "/"}.get(self.op, self.op)
        return f"({_c(self.a)} {op} {_c(self.b)})"


class Call(Expr):
    __slots__ = ("fn", "args")

    def __init__(self, fn: str, args):
        self.fn, self.args = fn, tuple(args)

    def eval(self, env):
        return _CALLS[self.fn](*[a.eval(env) for a in self.args])

    def free_vars(self):
        s: Set[str] = set()
        for a in self.args:
            s |= a.free_vars()
        return s

    def __repr__(self):
        return f"{self.fn}({', '.join(map(str, self.args))})"

    __str__ = __repr__

    def c(self):
        a = [_c(x) for x in self.args]
        if self.fn == "select":
            return f"({a[0]} ? {a[1]} : {a[2]})"
        if self.fn == "cdiv":
            return f"(({a[0]} + {a[1]} - 1) / {a[1]})"
        if self.fn == "not":
            return f"(!{a[0]})"
        return f"{self.fn}({', '.join(a)})"


class DimInt(int):
    """An ``int`` that remembers how it was computed from problem dimensions.

    ``lm.dims("N")`` returns ``DimInt(65536, "N")``; arithmetic keeps a symbolic
    rendering (``k.cdiv(N, 128)``) next to the value, so lowered kernels can be
    printed as shape-generic source instead of shape-specific constants.
    """

    def __new__(cls, value: int, sym: str):
        obj = int.__new__(cls, int(value))
        obj.sym = sym
        return obj

    def _mk(self, other, op: str, value, rev: bool = False):
        if isinstance(other, bool) or not isinstance(other, int):
            return value
        a, b = (symbol(other), self.sym) if rev else (self.sym, symbol(other))
        return DimInt(value, f"({a} {op} {b})")

    def __add__(self, o):
        if isinstance(o, Expr):
            return NotImplemented
        return self._mk(o, "+", int(self) + o)

    def __radd__(self, o):
        return self._mk(o, "+", o + int(self), rev=True)

    def __sub__(self, o):
        if isinstance(o, Expr):
            return NotImplemented
        return self._mk(o, "-", int(self) - o)

    def __rsub__(self, o):
        return self._mk(o, "-", o - int(self), rev=True)

    def __mul__(self, o):
        if isinstance(o, Expr):
            return NotImplemented
        return self._mk(o, "*", int(self) * o)

    def __rmul__(self, o):
        return self._mk(o, "*", o * int(self), rev=True)

    def __floordiv__(self, o):
        if isinstance(o, Expr):
            return NotImplemented
        return self._mk(o, "//", int(self) // o)

    def __rfloordiv__(self, o):
        return self._mk(o, "//", o // int(self), rev=True)

    def __mod__(self, o):
        if isinstance(o, Expr):
            return NotImplemented
        return self._mk(o, "%", int(self) % o)

    def __rmod__(self, o):
        return self._mk(o, "%", o % int(self), rev=True)

    def __neg__(self):
        return DimInt(-int(self), f"(-{self.sym})")

    def __reduce__(self):  # pickles / deep-copies as a plain int
        return (int, (int(self),))


def symbol(x) -> str:
    """Source rendering of a static value (DimInts render symbolically)."""
    if isinstance(x, DimInt):
        return x.sym
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return f"float({str(x)!r})"
    return repr(x)


def static_call(fn: str, args) -> int:
    """Evaluate min/max/cdiv on static ints, keeping a symbolic rendering if any argument is a DimInt."""
    vals = [int(a) for a in args]
    v = _CALLS[fn](*vals)
    if any(isinstance(a, DimInt) for a in args):
        return DimInt(v, f"k.{fn}({', '.join(symbol(a) for a in args)})")
    return v


Scalar = (bool, int, float, np.integer, np.floating, np.bool_)


def wrap(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, Scalar):
        if isinstance(x, (np.integer,)):
            x = int(x)
        elif isinstance(x, np.floating):
            x = float(x)
        elif isinstance(x, np.bool_):
            x = bool(x)
        return Const(x)
    raise ConstructionError(f"cannot use a value of type {type(x).__name__} in a scalar expression",
                            loc=capture_loc())


def binop(op: str, a, b):
    a, b = wrap(a), wrap(b)
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(_BINOPS[op](a.value, b.value))
    # light algebraic simplification keeps listings readable
    if op == "+" and isinstance(b, Const) and b.value == 0:
        return a
    if op == "+" and isinstance(a, Const) and a.value == 0:
        return b
    if op == "*" and isinstance(b, Const) and b.value == 1:
        return a
    if op == "*" and isinstance(a, Const) and a.value == 1:
        return b
    return BinOp(op, a, b)


def call(fn: str, *args):
    args = tuple(wrap(a) for a in args)
    if all(isinstance(a, Const) for a in args):
        return Const(_CALLS[fn](*[a.value for a in args]))
    return Call(fn, args)


def ev(x, env: Dict[str, Any]):
    """Evaluate ``x`` (expression or plain scalar) under ``env``."""
    if isinstance(x, Expr):
        v = x.eval(env)
        if isinstance(v, np.integer):
            return int(v)
        return v
    return x


def static(x):
    """Return the Python value of ``x`` if it is static, else raise."""
    if isinstance(x, Const):
        return x.value
    if isinstance(x, Expr):
        raise ConstructionError(f"expected a static value, got symbolic `{x}`", loc=capture_loc())
    return x


def is_static(x) -> bool:
    return not isinstance(x, Expr) or isinstance(x, Const)


def free_vars(x) -> Set[str]:
    return x.free_vars() if isinstance(x, Expr) else set()


def _c(x) -> str:
    if isinstance(x, Expr):
        return x.c()
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def to_c(x) -> str:
    return _c(x)


def to_py(x) -> str:
    """Render ``x`` as Python source (used to print raw kernels)."""
    if isinstance(x, Expr):
        return str(x)
    return repr(x)


def same(a, b) -> bool:
    """Structural equality of two scalar expressions (by rendering)."""
    return to_py(a) == to_py(b)


def exprs_str(xs: Iterable) -> str:
    return ", ".join(to_py(x) for x in xs)
