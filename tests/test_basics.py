import numpy as np
import pytest

from cake import dtypes as D
from cake.errors import ConstructionError
from cake.expr import BinOp, Const, DimInt, Var, ev, symbol, wrap
from cake.targets import get_target, table3
from cake.errors import TargetError


def test_expr_folding_and_eval():
    j = Var("j")
    e = (j + 3) * 2 % 5
    assert isinstance(e, BinOp)
    assert ev(e, {"j": 4}) == ((4 + 3) * 2) % 5
    assert isinstance(wrap(2) + 3, Const) and (wrap(2) + 3).value == 5
    assert (j + 0) is j


def test_symbolic_bool_is_rejected():
    with pytest.raises(ConstructionError, match="Python bool"):
        if Var("j") == 0:
            pass


def test_diminit_keeps_symbolic_rendering():
    n = DimInt(1000, "N")
    x = (n + 127) // 128 * 4
    assert int(x) == (1000 + 127) // 128 * 4
    assert symbol(x) == "(((N + 127) // 128) * 4)"
    assert symbol(float("inf")) == "float('inf')"


def test_bf16_roundtrip_and_rounding():
    x = np.array([1.0, -2.5, 3.140625, 1e-3, 65504.0], dtype=np.float32)
    q = D.quantize(x, D.bf16)
    assert np.allclose(q, x, rtol=8e-3)
    assert D.quantize(np.float32([1.00390625]), D.bf16)[0] == 1.0  # ties to even


def test_fp8_encoding():
    x = np.array([0.0, 1.0, -3.5, 448.0, 1000.0], dtype=np.float32)
    q = D.quantize(x, D.fp8)
    assert q[1] == 1.0 and q[2] == -3.5 and q[3] == 448.0 and q[4] == 448.0


def test_targets_are_exact():
    assert get_target("sm_100a").tmem and not get_target("sm_120a").tmem
    with pytest.raises(TargetError):
        get_target("sm_99")
    assert "sm_121a" in table3()
