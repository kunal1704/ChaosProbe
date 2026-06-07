"""Trajectory construction and normalization utilities for ChaosProbe.

This module is the bridge between transformer embedding matrices and the
trajectory-descriptor layer. It normalizes token-by-dimension embedding matrices
into the unit interval and generates the deterministic skew-tent trajectory used
as the reference path for descriptor extraction.
"""

from __future__ import annotations

import numpy as np


def generate_skew_tent_trajectory(
    initial_condition: float,
    threshold: float,
    length: int,
) -> np.ndarray:
    """Generate a deterministic skew-tent trajectory.

    The recurrence follows the ChaosFEX skew-tent sampler:

    - if ``x < threshold``, then ``x_next = x / threshold``
    - otherwise, ``x_next = (1 - x) / (1 - threshold)``

    Returns a 1D ``float64`` array of length ``length`` whose first element is
    the supplied initial condition.
    """

    initial_condition = float(initial_condition)
    threshold = float(threshold)

    if not np.isfinite(initial_condition):
        raise ValueError("initial_condition must be finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not 0 < initial_condition < 1:
        raise ValueError("initial_condition must satisfy 0 < initial_condition < 1")
    if not 0 < threshold < 1:
        raise ValueError("threshold must satisfy 0 < threshold < 1")
    if not isinstance(length, int):
        raise ValueError("length must be an integer")
    if length < 10:
        raise ValueError("length must be at least 10")

    trajectory = np.zeros(length, dtype=np.float64)
    trajectory[0] = np.float64(initial_condition)

    for idx in range(1, length):
        previous = trajectory[idx - 1]
        if previous < threshold:
            trajectory[idx] = previous / threshold
        else:
            trajectory[idx] = (1.0 - previous) / (1.0 - threshold)

    return trajectory


def normalize_to_unit_interval(x: np.ndarray, mode: str = "global") -> np.ndarray:
    """Normalize a 2D embedding matrix to ``[0, 1]``.

    ``global`` scales over the full matrix, ``token`` scales each row, and
    ``dimension`` scales each column. Constant groups map to zeros, which keeps
    downstream descriptor extraction finite and shape-preserving.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("x must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    if mode == "global":
        normalized = _min_max(values)
    elif mode == "token":
        normalized = _min_max(values, axis=1, keepdims=True)
    elif mode == "dimension":
        normalized = _min_max(values, axis=0, keepdims=True)
    else:
        raise ValueError("mode must be one of: global, token, dimension")

    return np.clip(normalized, 0.0, 1.0).astype(np.float64, copy=False)


def _min_max(
    values: np.ndarray,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    """Apply min-max scaling, returning zeros for constant groups."""

    minimum = np.min(values, axis=axis, keepdims=keepdims)
    maximum = np.max(values, axis=axis, keepdims=keepdims)
    span = maximum - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=np.float64),
        where=span != 0,
    )
