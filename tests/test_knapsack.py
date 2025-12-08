# tests/test_knapsack.py

"""
Unit tests for unbounded knapsack solver.

Run with:
    pytest tests/test_knapsack.py -v

Or:
    python -m pytest tests/test_knapsack.py -v
"""

import numpy as np
import pytest

from msmeqe.utils.knapsack import (
    solve_unbounded_knapsack,
    solve_unbounded_knapsack_with_stats,
    validate_solution,
)


class TestUnboundedKnapsack:
    """Test suite for unbounded knapsack solver."""

    def test_simple_case(self):
        """Test simple case with integer weights."""
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 20.0

        counts = solve_unbounded_knapsack(values, weights, budget)

        # Check feasibility
        total_weight = np.sum(counts * weights)
        assert total_weight <= budget + 0.01, "Solution exceeds budget"

        # Check at least one item selected
        assert np.sum(counts) > 0, "No items selected"

        # Validate solution
        assert validate_solution(counts, values, weights, budget)

    def test_fractional_weights(self):
        """Test with small fractional weights."""
        values = np.array([3.14, 2.71, 1.41, 1.73])
        weights = np.array([0.25, 0.15, 0.10, 0.20])
        budget = 1.0

        counts = solve_unbounded_knapsack(values, weights, budget, precision=0.01)

        total_weight = np.sum(counts * weights)
        assert total_weight <= budget + 0.01
        assert validate_solution(counts, values, weights, budget, tolerance=0.01)

    def test_single_item_optimal(self):
        """Test case where one item dominates."""
        values = np.array([100.0, 10.0, 5.0])
        weights = np.array([10.0, 10.0, 10.0])
        budget = 50.0

        counts = solve_unbounded_knapsack(values, weights, budget)

        # Item 0 should dominate (best value/weight ratio)
        assert counts[0] >= counts[1], "Should prefer item with best value"
        assert counts[0] >= counts[2], "Should prefer item with best value"

    def test_many_items(self):
        """Test with many items (stress test)."""
        np.random.seed(42)
        m = 100
        values = np.random.uniform(1.0, 10.0, m)
        weights = np.random.uniform(0.5, 5.0, m)
        budget = 100.0

        counts = solve_unbounded_knapsack(values, weights, budget, precision=0.05)

        total_weight = np.sum(counts * weights)
        assert total_weight <= budget + 0.1, "Solution exceeds budget"
        assert validate_solution(counts, values, weights, budget, tolerance=0.1)

    def test_edge_case_empty(self):
        """Test with no items."""
        values = np.array([])
        weights = np.array([])
        budget = 10.0

        counts = solve_unbounded_knapsack(values, weights, budget)

        assert len(counts) == 0, "Should return empty array"

    def test_edge_case_zero_budget(self):
        """Test with zero budget."""
        values = np.array([10.0, 20.0])
        weights = np.array([5.0, 10.0])
        budget = 0.0

        counts = solve_unbounded_knapsack(values, weights, budget)

        assert np.all(counts == 0), "Should select no items with zero budget"

    def test_edge_case_negative_budget(self):
        """Test with negative budget."""
        values = np.array([10.0, 20.0])
        weights = np.array([5.0, 10.0])
        budget = -5.0

        counts = solve_unbounded_knapsack(values, weights, budget)

        assert np.all(counts == 0), "Should select no items with negative budget"

    def test_high_precision(self):
        """Test with high precision (small discretization)."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.01, 0.02, 0.03])
        budget = 0.1

        counts = solve_unbounded_knapsack(values, weights, budget, precision=0.001)

        total_weight = np.sum(counts * weights)
        assert total_weight <= budget + 0.001

    def test_low_precision(self):
        """Test with low precision (large discretization)."""
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 20.0

        counts = solve_unbounded_knapsack(values, weights, budget, precision=1.0)

        total_weight = np.sum(counts * weights)
        assert total_weight <= budget + 1.0

    def test_invalid_input_negative_values(self):
        """Test that negative values raise error."""
        values = np.array([10.0, -5.0])
        weights = np.array([5.0, 10.0])
        budget = 20.0

        with pytest.raises(ValueError, match="non-negative"):
            solve_unbounded_knapsack(values, weights, budget)

    def test_invalid_input_zero_weights(self):
        """Test that zero weights raise error."""
        values = np.array([10.0, 20.0])
        weights = np.array([5.0, 0.0])
        budget = 20.0

        with pytest.raises(ValueError, match="positive"):
            solve_unbounded_knapsack(values, weights, budget)

    def test_invalid_input_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        values = np.array([10.0, 20.0])
        weights = np.array([5.0])
        budget = 20.0

        with pytest.raises(ValueError, match="same shape"):
            solve_unbounded_knapsack(values, weights, budget)

    def test_with_stats(self):
        """Test solve_unbounded_knapsack_with_stats."""
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 20.0

        counts, stats = solve_unbounded_knapsack_with_stats(values, weights, budget)

        # Check stats keys
        assert 'total_value' in stats
        assert 'total_weight' in stats
        assert 'n_selected' in stats
        assert 'utilization' in stats
        assert 'feasible' in stats

        # Check feasibility
        assert stats['feasible'], "Solution should be feasible"
        assert stats['total_weight'] <= budget + 0.01

        # Check utilization
        assert 0 <= stats['utilization'] <= 100

    def test_greedy_comparison(self):
        """
        Compare against greedy heuristic.

        DP should always be >= greedy solution.
        """
        values = np.array([60.0, 100.0, 120.0])
        weights = np.array([10.0, 20.0, 30.0])
        budget = 50.0

        # DP solution
        counts_dp = solve_unbounded_knapsack(values, weights, budget)
        value_dp = np.sum(counts_dp * values)

        # Greedy solution (by value/weight ratio)
        ratios = values / weights
        greedy_idx = np.argmax(ratios)
        counts_greedy = np.zeros_like(values, dtype=np.int32)
        counts_greedy[greedy_idx] = int(budget / weights[greedy_idx])
        value_greedy = np.sum(counts_greedy * values)

        # DP should be at least as good as greedy
        assert value_dp >= value_greedy - 0.01, "DP should beat or match greedy"


class TestValidateSolution:
    """Test suite for solution validation."""

    def test_valid_solution(self):
        """Test validation of a valid solution."""
        counts = np.array([1, 2, 0])
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 30.0

        assert validate_solution(counts, values, weights, budget)

    def test_invalid_exceeds_budget(self):
        """Test validation rejects solution exceeding budget."""
        counts = np.array([5, 5, 5])  # Way over budget
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 30.0

        assert not validate_solution(counts, values, weights, budget)

    def test_invalid_negative_counts(self):
        """Test validation rejects negative counts."""
        counts = np.array([1, -1, 0])
        values = np.array([10.0, 20.0, 15.0])
        weights = np.array([5.0, 10.0, 7.5])
        budget = 30.0

        assert not validate_solution(counts, values, weights, budget)


if __name__ == "__main__":
    # Run tests with: python tests/test_knapsack.py
    pytest.main([__file__, "-v"])