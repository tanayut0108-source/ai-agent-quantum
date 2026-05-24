"""Tests for the qubit parameter module."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_agent.core.qubit_parameter import ParameterRegister, QubitParameter
from quantum_agent.core.ternary_logic import NO, UNCERTAIN, YES


class TestQubitParameter:
    def test_from_float_zero(self) -> None:
        p = QubitParameter.from_float(0.0)
        assert p.to_float() == pytest.approx(0.0, abs=1e-6)
        assert p.state.probabilities[UNCERTAIN] == pytest.approx(1.0, abs=1e-6)

    def test_from_float_positive_one(self) -> None:
        p = QubitParameter.from_float(1.0)
        assert p.to_float() == pytest.approx(1.0, abs=1e-6)

    def test_from_float_negative_one(self) -> None:
        p = QubitParameter.from_float(-1.0)
        assert p.to_float() == pytest.approx(-1.0, abs=1e-6)

    def test_from_float_mid_positive(self) -> None:
        p = QubitParameter.from_float(0.5)
        assert p.to_float() > 0.0

    def test_from_float_mid_negative(self) -> None:
        p = QubitParameter.from_float(-0.5)
        assert p.to_float() < 0.0

    def test_from_float_clamps(self) -> None:
        p_high = QubitParameter.from_float(5.0)
        p_low = QubitParameter.from_float(-5.0)
        assert p_high.to_float() == pytest.approx(1.0, abs=1e-6)
        assert p_low.to_float() == pytest.approx(-1.0, abs=1e-6)

    def test_from_bool_true(self) -> None:
        p = QubitParameter.from_bool(True)
        assert p.to_float() == pytest.approx(1.0, abs=1e-6)

    def test_from_bool_false(self) -> None:
        p = QubitParameter.from_bool(False)
        assert p.to_float() == pytest.approx(-1.0, abs=1e-6)

    def test_uncertain_factory(self) -> None:
        p = QubitParameter.uncertain()
        assert p.to_float() == pytest.approx(0.0, abs=1e-6)

    def test_superposed_factory(self) -> None:
        p = QubitParameter.superposed()
        probs = p.state.probabilities
        for v in probs.values():
            assert v == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_measure_returns_basis(self) -> None:
        p = QubitParameter.from_bool(True)
        result = p.measure()
        assert result == YES

    def test_rotate(self) -> None:
        p = QubitParameter.superposed()
        p.rotate(1.0)
        total = sum(p.state.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_amplify(self) -> None:
        p = QubitParameter.superposed()
        p.amplify(YES, 3.0)
        assert p.state.probabilities[YES] > p.state.probabilities[NO]

    def test_dampen(self) -> None:
        p = QubitParameter.superposed()
        p.dampen(YES, 0.1)
        assert p.state.probabilities[YES] < p.state.probabilities[NO]

    def test_entropy_pure(self) -> None:
        p = QubitParameter.from_bool(True)
        assert p.entropy == pytest.approx(0.0, abs=1e-6)

    def test_clone(self) -> None:
        p = QubitParameter.from_float(0.7, name="test")
        c = p.clone()
        assert c.name == p.name
        assert c.to_float() == pytest.approx(p.to_float(), abs=1e-6)
        c.amplify(YES, 5.0)
        assert c.to_float() != pytest.approx(p.to_float(), abs=0.01)

    def test_repr(self) -> None:
        p = QubitParameter.from_float(0.5, name="my_param")
        r = repr(p)
        assert "QubitParameter" in r
        assert "my_param" in r


class TestParameterRegister:
    def test_add_and_get(self) -> None:
        reg = ParameterRegister()
        p = QubitParameter.from_float(0.5, name="p1")
        reg.add(p)
        assert reg.get("p1") is p
        assert reg.size == 1

    def test_remove(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.uncertain(name="p1"))
        assert reg.remove("p1")
        assert not reg.remove("p1")
        assert reg.size == 0

    def test_measure_all(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.from_bool(True, name="a"))
        reg.add(QubitParameter.from_bool(False, name="b"))
        results = reg.measure_all()
        assert results["a"] == YES
        assert results["b"] == NO

    def test_expected_values(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.from_float(1.0, name="a"))
        reg.add(QubitParameter.from_float(-1.0, name="b"))
        vals = reg.expected_values()
        assert vals["a"] == pytest.approx(1.0, abs=1e-6)
        assert vals["b"] == pytest.approx(-1.0, abs=1e-6)

    def test_bulk_rotate(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.superposed(name="a"))
        reg.add(QubitParameter.superposed(name="b"))
        reg.bulk_rotate(0.5)
        for _, p in reg.items():
            total = sum(p.state.probabilities.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_entangle_pair(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.from_bool(True, name="a"))
        reg.add(QubitParameter.from_bool(False, name="b"))
        reg.entangle_pair("a", "b", strength=0.5)
        pa = reg.get("a")
        pb = reg.get("b")
        assert pa is not None
        assert pb is not None
        total_a = sum(pa.state.probabilities.values())
        total_b = sum(pb.state.probabilities.values())
        assert total_a == pytest.approx(1.0, abs=1e-6)
        assert total_b == pytest.approx(1.0, abs=1e-6)

    def test_from_float_array(self) -> None:
        reg = ParameterRegister()
        values = np.array([0.1, -0.5, 0.9], dtype=np.float64)
        names = reg.from_float_array(values, prefix="x")
        assert len(names) == 3
        assert reg.size == 3
        assert reg.get("x_0") is not None

    def test_to_float_array(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.from_float(1.0, name="a"))
        reg.add(QubitParameter.from_float(-1.0, name="b"))
        arr = reg.to_float_array(["a", "b", "missing"])
        assert arr[0] == pytest.approx(1.0, abs=1e-6)
        assert arr[1] == pytest.approx(-1.0, abs=1e-6)
        assert arr[2] == pytest.approx(0.0, abs=1e-6)

    def test_summary(self) -> None:
        reg = ParameterRegister()
        reg.add(QubitParameter.from_float(0.5, name="a"))
        s = reg.summary()
        assert s["count"] == 1
        assert "mean_value" in s

    def test_repr(self) -> None:
        reg = ParameterRegister(capacity=100)
        assert "ParameterRegister" in repr(reg)
