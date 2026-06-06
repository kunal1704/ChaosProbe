import numpy as np
import pytest

from chaosprobe.metrics import (
    anisotropy,
    compute_channel_metrics,
    descriptor_channel_summary,
    linear_cka,
    mean_cosine_similarity,
    neighborhood_preservation,
    pairwise_distance_correlation,
)


@pytest.fixture
def matrix():
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.25],
            [1.0, 1.0, 0.75],
            [0.5, 0.25, 1.0],
        ],
        dtype=np.float64,
    )


def test_identical_matrices_have_high_similarity(matrix):
    assert mean_cosine_similarity(matrix, matrix) == pytest.approx(1.0)
    assert linear_cka(matrix, matrix) == pytest.approx(1.0)
    assert pairwise_distance_correlation(matrix, matrix) == pytest.approx(1.0)
    assert neighborhood_preservation(matrix, matrix, k=2) == pytest.approx(1.0)


def test_anisotropy_returns_finite_scalar(matrix):
    value = anisotropy(matrix)

    assert isinstance(value, float)
    assert np.isfinite(value)


def test_descriptor_channel_summary_returns_one_entry_per_channel(matrix):
    desc = np.stack([matrix, matrix + 1.0, matrix * 2.0, np.zeros_like(matrix)], axis=2)

    summary = descriptor_channel_summary(desc)

    assert len(summary) == 4
    for idx, channel_summary in enumerate(summary):
        assert channel_summary["channel"] == idx
        for key in ["mean", "std", "min", "max"]:
            assert isinstance(channel_summary[key], float)


def test_compute_channel_metrics_returns_channel_wise_metrics(matrix):
    desc = np.stack([matrix, matrix + 0.1, matrix * 2.0, np.zeros_like(matrix)], axis=2)

    metrics = compute_channel_metrics(matrix, desc, k=2)

    assert len(metrics) == 4
    for idx, channel_metrics in enumerate(metrics):
        assert channel_metrics["channel"] == idx
        for key in [
            "mean_cosine_similarity",
            "linear_cka",
            "pairwise_distance_correlation",
            "neighborhood_preservation",
            "anisotropy_original",
            "anisotropy_channel",
        ]:
            assert isinstance(channel_metrics[key], float)


def test_zero_vectors_are_handled_safely():
    zeros = np.zeros((3, 2), dtype=np.float64)

    assert mean_cosine_similarity(zeros, zeros) == 0.0
    assert linear_cka(zeros, zeros) == 0.0
    assert pairwise_distance_correlation(zeros, zeros) == 0.0
    assert anisotropy(zeros) == 0.0


def test_neighborhood_preservation_handles_small_sample_counts():
    one_row = np.array([[1.0, 2.0]], dtype=np.float64)

    assert neighborhood_preservation(one_row, one_row, k=5) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        mean_cosine_similarity,
        linear_cka,
        pairwise_distance_correlation,
        neighborhood_preservation,
    ],
)
def test_metrics_reject_invalid_dimensions(metric):
    with pytest.raises(ValueError):
        metric(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_mean_cosine_similarity_requires_same_shape(matrix):
    with pytest.raises(ValueError):
        mean_cosine_similarity(matrix, matrix[:, :2])


def test_row_count_must_match_for_geometry_metrics(matrix):
    other = matrix[:3]

    with pytest.raises(ValueError):
        linear_cka(matrix, other)

    with pytest.raises(ValueError):
        pairwise_distance_correlation(matrix, other)

    with pytest.raises(ValueError):
        neighborhood_preservation(matrix, other)


def test_descriptor_helpers_validate_shapes(matrix):
    with pytest.raises(ValueError):
        descriptor_channel_summary(matrix)

    with pytest.raises(ValueError):
        compute_channel_metrics(matrix, np.zeros((4, 2, 4)))
