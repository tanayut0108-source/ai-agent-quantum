"""Tests for the ternary logic module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantum_agent.core.ternary_logic import (
    NO,
    UNCERTAIN,
    YES,
    TernaryBasis,
    TernaryState,
    ternary_and,
    ternary_consensus,
    ternary_not,
    ternary_or,
)


class TestTernaryBasis:
    def test_values(self) -> None:
        assert NO == -1
        assert UNCERTAIN == 0
        assert YES == 1

    def test_index_mapping(self) -> None:
        assert TernaryBasis.index(NO) == 0
        assert TernaryBasis.index(UNCERTAIN) == 1
        assert TernaryBasis.index(YES) == 2


class TestTernaryState:
    def test_default_is_uncertain(self) -> None:
        s = TernaryState()
        p = s.probabilities
        assert p[UNCERTAIN] == pytest.approx(1.0, abs=1e-9)
        assert p[NO] == pytest.approx(0.0, abs=1e-9)
        assert p[YES] == pytest.approx(0.0, abs=1e-9)

    def test_set_pure_positive(self) -> None:
        s = TernaryState()
        s.set_pure(YES)
        assert s.probabilities[YES] == pytest.approx(1.0, abs=1e-9)
        assert s.dominant == YES

    def test_set_pure_negative(self) -> None:
        s = TernaryState()
        s.set_pure(NO)
        assert s.probabilities[NO] == pytest.approx(1.0, abs=1e-9)
        assert s.dominant == NO

    def test_superpose_uniform(self) -> None:
        s = TernaryState()
        s.superpose_uniform()
        for v in s.probabilities.values():
            assert v == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_probabilities_sum_to_one(self) -> None:
        s = TernaryState(np.array([0.5, 0.3, 0.7], dtype=np.complex128))
        total = sum(s.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_measure_collapses(self) -> None:
        s = TernaryState()
        s.set_pure(YES)
        result = s.measure()
        assert result == YES
        assert s.probabilities[YES] == pytest.approx(1.0, abs=1e-9)

    def test_entropy_pure_state(self) -> None:
        s = TernaryState()
        s.set_pure(NO)
        assert s.entropy() == pytest.approx(0.0, abs=1e-9)

    def test_entropy_uniform(self) -> None:
        s = TernaryState()
        s.superpose_uniform()
        assert s.entropy() == pytest.approx(math.log2(3), abs=1e-6)

    def test_rotate(self) -> None:
        s = TernaryState()
        s.superpose_uniform()
        s.rotate(0.5)
        total = sum(s.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_clone(self) -> None:
        s = TernaryState(np.array([0.5, 0.3, 0.7], dtype=np.complex128))
        c = s.clone()
        assert np.allclose(s.amplitudes, c.amplitudes)
        c.set_pure(NO)
        assert not np.allclose(s.amplitudes, c.amplitudes)

    def test_repr(self) -> None:
        s = TernaryState()
        r = repr(s)
        assert "TernaryState" in r


class TestTernaryGates:
    def test_not_swaps_no_yes(self) -> None:
        s = TernaryState()
        s.set_pure(YES)
        result = ternary_not(s)
        assert result.probabilities[NO] == pytest.approx(1.0, abs=1e-9)

    def test_not_preserves_uncertain(self) -> None:
        s = TernaryState()
        result = ternary_not(s)
        assert result.probabilities[UNCERTAIN] == pytest.approx(1.0, abs=1e-9)

    def test_and_yes_yes(self) -> None:
        a = TernaryState()
        a.set_pure(YES)
        b = TernaryState()
        b.set_pure(YES)
        result = ternary_and(a, b)
        assert result.dominant == YES

    def test_and_no_anything(self) -> None:
        a = TernaryState()
        a.set_pure(NO)
        b = TernaryState()
        b.set_pure(YES)
        result = ternary_and(a, b)
        assert result.dominant == NO

    def test_or_yes_anything(self) -> None:
        a = TernaryState()
        a.set_pure(YES)
        b = TernaryState()
        b.set_pure(NO)
        result = ternary_or(a, b)
        assert result.dominant == YES

    def test_or_no_no(self) -> None:
        a = TernaryState()
        a.set_pure(NO)
        b = TernaryState()
        b.set_pure(NO)
        result = ternary_or(a, b)
        assert result.dominant == NO

    def test_consensus_empty(self) -> None:
        result = ternary_consensus([])
        assert result.probabilities[UNCERTAIN] == pytest.approx(1.0, abs=1e-9)

    def test_consensus_multiple(self) -> None:
        states = []
        for _ in range(5):
            s = TernaryState()
            s.set_pure(YES)
            states.append(s)
        result = ternary_consensus(states)
        assert result.dominant == YES
