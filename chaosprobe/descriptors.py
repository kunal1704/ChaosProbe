"""Descriptor extraction for ChaosProbe.

The descriptor tensor is the scientific object of study and is preserved with
shape ``T x D x 4``.
"""

from __future__ import annotations

import numpy as np


TTSS_CHANNEL = 0
ENERGY_CHANNEL = 1
TIME_TO_MATCH_CHANNEL = 2
ENTROPY_CHANNEL = 3
NUM_DESCRIPTOR_CHANNELS = 4


def compute_descriptors(
    values: np.ndarray,
    trajectory: np.ndarray,
    epsilon: float,
    threshold: float,
) -> np.ndarray:
    """Compute TTSS, energy, time-to-match, and entropy descriptors."""

    value_matrix = np.asarray(values, dtype=np.float64)
    trajectory_vector = np.asarray(trajectory, dtype=np.float64)
    _validate_descriptor_inputs(value_matrix, trajectory_vector, epsilon, threshold)

    descriptors = np.zeros(
        (*value_matrix.shape, NUM_DESCRIPTOR_CHANNELS),
        dtype=np.float64,
    )

    for token_idx in range(value_matrix.shape[0]):
        for dim_idx in range(value_matrix.shape[1]):
            value = value_matrix[token_idx, dim_idx]
            match_indices = np.flatnonzero(np.abs(trajectory_vector - value) < epsilon)
            match_idx = int(match_indices[0]) if match_indices.size else len(trajectory_vector)

            if match_idx > 0:
                path = trajectory_vector[:match_idx]
            else:
                path = trajectory_vector[:1]

            ttss = float(np.count_nonzero(path > threshold) / len(path))
            descriptors[token_idx, dim_idx, TTSS_CHANNEL] = ttss
            descriptors[token_idx, dim_idx, ENERGY_CHANNEL] = float(np.sum(path**2))
            descriptors[token_idx, dim_idx, TIME_TO_MATCH_CHANNEL] = float(match_idx)
            descriptors[token_idx, dim_idx, ENTROPY_CHANNEL] = _binary_entropy(ttss)

    return descriptors


def flatten_descriptors(desc: np.ndarray) -> np.ndarray:
    """Convert a ``(T, D, C)`` descriptor tensor into ``(T, D * C)``."""

    descriptor_tensor = np.asarray(desc)
    if descriptor_tensor.ndim != 3:
        raise ValueError("desc must be a 3D descriptor tensor")

    tokens, dimensions, channels = descriptor_tensor.shape
    return descriptor_tensor.reshape(tokens, dimensions * channels)


def channel_view(desc: np.ndarray, channel: int) -> np.ndarray:
    """Return a single descriptor channel as a ``(T, D)`` view."""

    descriptor_tensor = np.asarray(desc)
    if descriptor_tensor.ndim != 3:
        raise ValueError("desc must be a 3D descriptor tensor")
    if not 0 <= channel < descriptor_tensor.shape[2]:
        raise IndexError("channel is out of range")

    return descriptor_tensor[:, :, channel]


def _validate_descriptor_inputs(
    values: np.ndarray,
    trajectory: np.ndarray,
    epsilon: float,
    threshold: float,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("values must lie in [0, 1]")
    if trajectory.ndim != 1:
        raise ValueError("trajectory must be a 1D array")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must contain only finite values")

    epsilon = float(epsilon)
    threshold = float(threshold)
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not epsilon > 0:
        raise ValueError("epsilon must be greater than 0")
    if not 0 < threshold < 1:
        raise ValueError("threshold must satisfy 0 < threshold < 1")


def _binary_entropy(probability: float) -> float:
    if probability == 0.0 or probability == 1.0:
        return 0.0

    complement = 1.0 - probability
    return float(
        -probability * np.log2(probability)
        - complement * np.log2(complement)
    )
