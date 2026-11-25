"""Error types and source locations.

Every diagnostic Cake produces is *localized*: it carries the source line of the
agent-authored program that caused it (paper Sec. 3.1: "A finding identifies the
affected program region and the class of violated contract").
"""
from __future__ import annotations

import linecache
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Loc:
    lineno: int = 0
    text: str = ""
    filename: str = "<cake>"

    def __str__(self) -> str:
        if not self.lineno:
            return "<unknown location>"
        return f"line {self.lineno}: {self.text.strip()}"

    def to_dict(self) -> dict:
        return {"line": self.lineno, "text": self.text.strip()}


# Files that implement the agent-facing API. Frames from these files are skipped
# when we look for the user's source line.
_API_FILES: set = set()


def register_api_file(path: str) -> None:
    _API_FILES.add(os.path.normcase(os.path.abspath(path)))


register_api_file(__file__)


def capture_loc() -> Loc:
    """Return the location of the innermost frame that belongs to user code."""
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if fn.startswith("<") or os.path.normcase(os.path.abspath(fn)) not in _API_FILES:
            if fn.startswith("<frozen") or fn.endswith("contextlib.py"):
                f = f.f_back
                continue
            line = linecache.getline(fn, f.f_lineno)
            return Loc(f.f_lineno, line.rstrip("\n"), fn)
        f = f.f_back
    return Loc()


class CakeError(Exception):
    """Base class for all Cake errors."""

    kind = "error"

    def __init__(self, message: str, loc: Optional[Loc] = None, hint: Optional[str] = None,
                 code: Optional[str] = None):
        self.message = message
        self.loc = loc
        self.hint = hint
        self.code = code
        super().__init__(self.format())

    def format(self) -> str:
        parts = []
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(self.message)
        s = " ".join(parts)
        if self.loc is not None and self.loc.lineno:
            s += f"\n  at {self.loc}"
        if self.hint:
            s += f"\n  hint: {self.hint}"
        return s

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "code": self.code,
            "message": self.message,
            "loc": self.loc.to_dict() if self.loc else None,
            "hint": self.hint,
        }


class ConstructionError(CakeError):
    """IR construction check failed (bad use of the schedule API)."""

    kind = "construction"


class CakeTypeError(ConstructionError):
    """A typing rule of Cake IR was violated (design principle P4)."""

    kind = "type"


class SandboxError(CakeError):
    """Agent-authored source used a construct outside the allowed Python subset."""

    kind = "sandbox"


class LoweringError(CakeError):
    kind = "lowering"


class TargetError(CakeError):
    kind = "target"


class SimError(CakeError):
    """Runtime failure on the simulated device.

    ``fault`` is a coarse class (what a CUDA runtime would report), while
    ``detail`` holds sanitizer-style evidence used by the harness.
    """

    kind = "runtime"

    def __init__(self, fault: str, message: str, loc: Optional[Loc] = None,
                 detail: Optional[dict] = None, hint: Optional[str] = None):
        self.fault = fault
        self.detail = detail or {}
        super().__init__(message, loc=loc, hint=hint, code=fault)

    def coarse(self) -> str:
        return {
            "hang": "kernel timed out (watchdog): the kernel did not terminate",
            "illegal_address": "CUDA error: an illegal memory access was encountered",
            "misaligned": "CUDA error: misaligned address",
            "illegal_instruction": "CUDA error: an illegal instruction was encountered",
            "launch": "CUDA error: too many resources requested for launch",
            "unsupported": "CUDA error: unsupported operation on this device",
        }.get(self.fault, f"CUDA error: {self.fault}")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["fault"] = self.fault
        d["detail"] = self.detail
        return d
