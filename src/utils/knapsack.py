# src/utils/knapsack.py

"""
Unbounded knapsack solver for MS-MEQE.

Implements dynamic programming solution for the unbounded knapsack problem:
    maximize    sum_k c_k * v_k
    subject to  sum_k c_k * w_k <= W
                c_k >= 0, integer

Uses principled weight discretization to handle fractional weights while
maintaining optimality guarantees.

Usage:
    from msmeqe.utils.knapsack import solve_unbounded_knapsack

    values = np.array([10.0, 20.0, 15.0])
    weights = np.array([5.0, 10.0, 7.5])
    budget = 20

    counts = solve_unbounded_knapsack(values, weights, budget)
    # counts = [0, 2, 0]  # Take item 1 twice
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def solve_unbounded_knapsack(
        values: np.ndarray,
        weights: np.ndarray,
        budget: float,
        precision: float = 0.01,
) -> np.ndarray:
    """
    Solve unbounded knapsack problem with fractional weights.

    Algorithm:
        1. Discretize weights to integers: w_int[k] = ceil(w[k] / precision)
        2. Scale budget: W_int = floor(W / precision)
        3. Run standard unbounded knapsack DP
        4. Backtrack to recover counts

    Discretization preserves optimality if precision is small enough.
    Complexity: O(m * W_int) where m = number of items

    Args:
        values: np.ndarray (m,) - value v_k for each item (non-negative)
        weights: np.ndarray (m,) - weight w_k for each item (positive)
        budget: float - capacity constraint W (positive)
        precision: float - discretization precision (default: 0.01)
                  Smaller = more accurate but slower

    Returns:
        counts: np.ndarray (m,) - optimal frequency c_k for each item

    Raises:
        ValueError: If inputs have invalid shapes or values

    Example:
        >>> values = np.array([60, 100, 120])
        >>> weights = np.array([10, 20, 30])
        >>> counts = solve_unbounded_knapsack(values, weights, 50)
        >>> print(counts)  # [0, 0, 1] or similar optimal solution
    """
    # === INPUT VALIDATION ===

    if not isinstance(values, np.ndarray) or not isinstance(weights, np.ndarray):
        raise ValueError("values and weights must be numpy arrays")

    if values.shape != weights.shape:
        raise ValueError(
            f"values and weights must have same shape: "
            f"{values.shape} vs {weights.shape}"
        )

    if len(values.shape) != 1:
        raise ValueError(f"values and weights must be 1D arrays, got shape {values.shape}")

    m = len(values)

    if m == 0:
        return np.zeros((0,), dtype=np.int32)

    if budget <= 0:
        logger.debug("Budget <= 0, returning zero counts")
        return np.zeros(m, dtype=np.int32)

    if np.any(values < 0):
        raise ValueError("All values must be non-negative")

    if np.any(weights <= 0):
        raise ValueError("All weights must be positive")

    if precision <= 0:
        raise ValueError(f"Precision must be positive, got {precision}")

    # === DISCRETIZE WEIGHTS ===

    v = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)

    # Discretize weights: w_int[k] = ceil(w[k] / precision)
    # Using ceil ensures we don't underestimate capacity usage
    w_int = np.ceil(w / precision).astype(np.int32)
    w_int = np.maximum(w_int, 1)  # Ensure all weights >= 1

    # Scale budget down (conservative)
    W_int = int(np.floor(budget / precision))
    W_int = max(W_int, 1)  # Ensure capacity >= 1

    logger.debug(
        "Knapsack: m=%d items, budget=%.4f → W_int=%d, precision=%.4f",
        m, budget, W_int, precision
    )
    logger.debug(
        "Weight range: [%.4f, %.4f] → [%d, %d]",
        w.min(), w.max(), w_int.min(), w_int.max()
    )

    # === UNBOUNDED KNAPSACK DP ===

    # dp[cap] = maximum value achievable with capacity cap
    dp = np.full(W_int + 1, -np.inf, dtype=np.float32)
    dp[0] = 0.0

    # choice[cap] = which item was chosen to achieve dp[cap]
    choice = np.full(W_int + 1, -1, dtype=np.int32)

    # Fill DP table
    for cap in range(1, W_int + 1):
        for i in range(m):
            wi = w_int[i]
            if wi <= cap and dp[cap - wi] != -np.inf:
                new_val = v[i] + dp[cap - wi]
                if new_val > dp[cap]:
                    dp[cap] = new_val
                    choice[cap] = i

    # === BACKTRACK TO RECOVER SOLUTION ===

    counts = np.zeros(m, dtype=np.int32)
    cap = W_int

    # Backtrack from dp[W_int]
    iterations = 0
    max_iterations = W_int + 1  # Safety limit

    while cap > 0 and choice[cap] != -1:
        if iterations >= max_iterations:
            logger.warning("Backtracking exceeded max iterations, stopping")
            break

        i = choice[cap]
        counts[i] += 1
        cap -= w_int[i]
        iterations += 1

    # === VERIFY SOLUTION ===

    total_value = float(np.sum(counts * v))
    total_weight = float(np.sum(counts * w))
    n_selected = int(np.sum(counts > 0))

    # Check capacity constraint (allow small tolerance due to discretization)
    if total_weight > budget + precision:
        logger.warning(
            "Solution exceeds budget: %.4f > %.4f (tolerance=%.4f). "
            "Consider reducing precision.",
            total_weight, budget, precision
        )

    utilization = 100.0 * total_weight / budget if budget > 0 else 0.0

    logger.debug(
        "Knapsack solved: %d/%d items selected, "
        "value=%.4f, weight=%.4f/%.4f (%.1f%% utilization)",
        n_selected, m, total_value, total_weight, budget, utilization
    )

    return counts


def solve_unbounded_knapsack_with_stats(
        values: np.ndarray,
        weights: np.ndarray,
        budget: float,
        precision: float = 0.01,
) -> Tuple[np.ndarray, dict]:
    """
    Solve unbounded knapsack and return solution statistics.

    Same as solve_unbounded_knapsack but also returns detailed statistics.

    Args:
        values, weights, budget, precision: Same as solve_unbounded_knapsack

    Returns:
        counts: Optimal item frequencies
        stats: Dictionary containing:
            - total_value: Sum of values
            - total_weight: Sum of weights
            - n_selected: Number of items selected (count > 0)
            - utilization: Budget utilization percentage
            - feasible: Whether solution satisfies capacity constraint
    """
    counts = solve_unbounded_knapsack(values, weights, budget, precision)

    total_value = float(np.sum(counts * values))
    total_weight = float(np.sum(counts * weights))
    n_selected = int(np.sum(counts > 0))
    utilization = 100.0 * total_weight / budget if budget > 0 else 0.0
    feasible = total_weight <= budget + precision

    stats = {
        'total_value': total_value,
        'total_weight': total_weight,
        'n_selected': n_selected,
        'utilization': utilization,
        'feasible': feasible,
    }

    return counts, stats


def validate_solution(
        counts: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        budget: float,
        tolerance: float = 1e-6,
) -> bool:
    """
    Validate that a knapsack solution is feasible.

    Args:
        counts: Item frequencies
        values: Item values
        weights: Item weights
        budget: Capacity constraint
        tolerance: Numerical tolerance for capacity check

    Returns:
        True if solution is feasible, False otherwise
    """
    if counts.shape != values.shape or counts.shape != weights.shape:
        logger.error("Shape mismatch in solution validation")
        return False

    if np.any(counts < 0):
        logger.error("Negative counts in solution")
        return False

    total_weight = float(np.sum(counts * weights))

    if total_weight > budget + tolerance:
        logger.error(
            "Solution violates capacity: %.4f > %.4f",
            total_weight, budget
        )
        return False

    return True