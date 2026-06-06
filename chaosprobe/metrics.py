"""Representation metrics for ChaosProbe descriptor analysis."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def mean_cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Average row-wise cosine similarity for matrices with the same shape."""

    matrix_a, matrix_b = _validate_same_shape_2d(A, B)
    numerator = np.sum(matrix_a * matrix_b, axis=1)
    denominator = np.linalg.norm(matrix_a, axis=1) * np.linalg.norm(matrix_b, axis=1)
    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )
    return float(np.mean(similarities))


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Compute linear centered kernel alignment for two 2D representations."""

    matrix_a, matrix_b = _validate_same_rows_2d(A, B)
    centered_a = matrix_a - np.mean(matrix_a, axis=0, keepdims=True)
    centered_b = matrix_b - np.mean(matrix_b, axis=0, keepdims=True)

    cross = np.linalg.norm(centered_a.T @ centered_b, ord="fro") ** 2
    norm_a = np.linalg.norm(centered_a.T @ centered_a, ord="fro")
    norm_b = np.linalg.norm(centered_b.T @ centered_b, ord="fro")
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    return float(cross / denominator)


def pairwise_distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman correlation between condensed pairwise distance vectors."""

    matrix_a, matrix_b = _validate_same_rows_2d(A, B)
    if matrix_a.shape[0] < 3:
        return 0.0

    distances_a = pdist(matrix_a, metric="euclidean")
    distances_b = pdist(matrix_b, metric="euclidean")
    if np.all(distances_a == distances_a[0]) or np.all(distances_b == distances_b[0]):
        return 0.0

    correlation = spearmanr(distances_a, distances_b).correlation
    if correlation is None or not np.isfinite(correlation):
        return 0.0
    return float(correlation)


def neighborhood_preservation(A: np.ndarray, B: np.ndarray, k: int = 5) -> float:
    """Compare k-nearest-neighbor overlap between two representations."""

    matrix_a, matrix_b = _validate_same_rows_2d(A, B)
    n_rows = matrix_a.shape[0]
    if n_rows <= 1:
        return 0.0
    if not isinstance(k, int):
        raise ValueError("k must be an integer")

    k = min(k, n_rows - 1)
    if k <= 0:
        return 0.0

    neighbors_a = _nearest_neighbors(matrix_a, k)
    neighbors_b = _nearest_neighbors(matrix_b, k)
    overlaps = [
        len(set(row_a).intersection(row_b)) / k
        for row_a, row_b in zip(neighbors_a, neighbors_b)
    ]
    return float(np.mean(overlaps))


def anisotropy(A: np.ndarray) -> float:
    """Average pairwise cosine similarity excluding self-similarity."""

    matrix = _validate_2d(A, "A")
    if matrix.shape[0] < 2:
        return 0.0

    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    cosine = np.divide(
        matrix @ matrix.T,
        denominator,
        out=np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
        where=denominator != 0,
    )
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(cosine[mask]))


def descriptor_channel_summary(desc: np.ndarray) -> list[dict[str, float]]:
    """Return JSON-serializable summary statistics for each descriptor channel."""

    descriptor_tensor = _validate_descriptor_tensor(desc)
    summaries = []
    for channel in range(descriptor_tensor.shape[2]):
        channel_values = descriptor_tensor[:, :, channel]
        summaries.append(
            {
                "channel": int(channel),
                "mean": float(np.mean(channel_values)),
                "std": float(np.std(channel_values)),
                "min": float(np.min(channel_values)),
                "max": float(np.max(channel_values)),
            }
        )
    return summaries


def compute_channel_metrics(
    original: np.ndarray,
    desc: np.ndarray,
    k: int = 5,
) -> list[dict[str, float]]:
    """Compare an original representation against each descriptor channel."""

    original_matrix = _validate_2d(original, "original")
    descriptor_tensor = _validate_descriptor_tensor(desc)
    if original_matrix.shape != descriptor_tensor.shape[:2]:
        raise ValueError("original must have the same T x D shape as desc")

    original_anisotropy = anisotropy(original_matrix)
    metrics = []
    for channel in range(descriptor_tensor.shape[2]):
        channel_matrix = descriptor_tensor[:, :, channel]
        metrics.append(
            {
                "channel": int(channel),
                "mean_cosine_similarity": mean_cosine_similarity(
                    original_matrix,
                    channel_matrix,
                ),
                "linear_cka": linear_cka(original_matrix, channel_matrix),
                "pairwise_distance_correlation": pairwise_distance_correlation(
                    original_matrix,
                    channel_matrix,
                ),
                "neighborhood_preservation": neighborhood_preservation(
                    original_matrix,
                    channel_matrix,
                    k=k,
                ),
                "anisotropy_original": original_anisotropy,
                "anisotropy_channel": anisotropy(channel_matrix),
            }
        )
    return metrics


def _nearest_neighbors(matrix: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :k]


def _validate_same_shape_2d(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = _validate_same_rows_2d(A, B)
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("A and B must have the same shape")
    return matrix_a, matrix_b


def _validate_same_rows_2d(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_a = _validate_2d(A, "A")
    matrix_b = _validate_2d(B, "B")
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError("A and B must have the same number of rows")
    return matrix_a, matrix_b


def _validate_2d(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_descriptor_tensor(desc: np.ndarray) -> np.ndarray:
    descriptor_tensor = np.asarray(desc, dtype=np.float64)
    if descriptor_tensor.ndim != 3:
        raise ValueError("desc must be a 3D descriptor tensor")
    if not np.all(np.isfinite(descriptor_tensor)):
        raise ValueError("desc must contain only finite values")
    return descriptor_tensor
