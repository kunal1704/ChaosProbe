"""Baseline descriptor generators for ChaosProbe.

These helpers create non-chaotic or perturbed alternatives for comparing
descriptor tensors against controlled reference conditions.
"""

from __future__ import annotations

import numpy as np

from chaosprobe.descriptors import compute_descriptors


def shuffled_trajectory_descriptors(
    values: np.ndarray,
    trajectory: np.ndarray,
    epsilon: float,
    threshold: float,
    seed: int = 42,
) -> np.ndarray:
    """Compute descriptors after shuffling a copy of the trajectory."""

    trajectory_copy = _validate_trajectory(trajectory).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(trajectory_copy)
    return compute_descriptors(values, trajectory_copy, epsilon, threshold)


def random_trajectory_descriptors(
    values: np.ndarray,
    trajectory_length: int,
    epsilon: float,
    threshold: float,
    seed: int = 42,
) -> np.ndarray:
    """Compute descriptors using a seeded uniform random trajectory."""

    if not isinstance(trajectory_length, int):
        raise ValueError("trajectory_length must be an integer")
    if trajectory_length < 10:
        raise ValueError("trajectory_length must be at least 10")

    rng = np.random.default_rng(seed)
    trajectory = rng.uniform(0.0, 1.0, size=trajectory_length).astype(np.float64)
    return compute_descriptors(values, trajectory, epsilon, threshold)


def gaussian_value_baseline(
    values: np.ndarray,
    trajectory: np.ndarray,
    epsilon: float,
    threshold: float,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Compute descriptors after adding clipped Gaussian value noise."""

    value_matrix = _validate_values(values)
    trajectory_vector = _validate_trajectory(trajectory)
    noise_scale = _validate_noise_scale(noise_scale)

    rng = np.random.default_rng(seed)
    noisy_values = np.clip(
        value_matrix + rng.normal(0.0, noise_scale, size=value_matrix.shape),
        0.0,
        1.0,
    )
    return compute_descriptors(noisy_values, trajectory_vector, epsilon, threshold)


def uniform_value_baseline(
    values: np.ndarray,
    trajectory: np.ndarray,
    epsilon: float,
    threshold: float,
    noise_scale: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Compute descriptors after adding clipped uniform value noise."""

    value_matrix = _validate_values(values)
    trajectory_vector = _validate_trajectory(trajectory)
    noise_scale = _validate_noise_scale(noise_scale)

    rng = np.random.default_rng(seed)
    noisy_values = np.clip(
        value_matrix
        + rng.uniform(-noise_scale, noise_scale, size=value_matrix.shape),
        0.0,
        1.0,
    )
    return compute_descriptors(noisy_values, trajectory_vector, epsilon, threshold)


def _validate_values(values: np.ndarray) -> np.ndarray:
    value_matrix = np.asarray(values, dtype=np.float64)
    if value_matrix.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not np.all(np.isfinite(value_matrix)):
        raise ValueError("values must contain only finite values")
    if np.any(value_matrix < 0) or np.any(value_matrix > 1):
        raise ValueError("values must lie in [0, 1]")
    return value_matrix


def _validate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    trajectory_vector = np.asarray(trajectory, dtype=np.float64)
    if trajectory_vector.ndim != 1:
        raise ValueError("trajectory must be a 1D array")
    if not np.all(np.isfinite(trajectory_vector)):
        raise ValueError("trajectory must contain only finite values")
    return trajectory_vector


def _validate_noise_scale(noise_scale: float) -> float:
    noise_scale = float(noise_scale)
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise_scale must be finite and greater than 0")
    return noise_scale
