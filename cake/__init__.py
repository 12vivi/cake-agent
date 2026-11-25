"""Cake: compiler-agent co-design for kernel evolution (unofficial reproduction).

Agents author Cake IR (``@cake.schedule()``); the harness verifies it, ranks it with a
calibrated cost model, lowers it to LIR and evaluates it on a simulated GPU.
"""
from .ir.builder import LM, schedule  # noqa: F401
from .lir.builder import raw_kernel  # noqa: F401
from .lower import lower  # noqa: F401
from .targets import TARGETS, get_target  # noqa: F401

__version__ = "0.1.0"
