import numpy as np
import pytest

from chaosprobe.descriptors import (
    channel_view,
    compute_descriptors,
    flatten_descriptors,
)
from chaosprobe.trajectory import generate_skew_tent_trajectory


def test_descriptor_output_shape_dtype_entropy_and_time_to_match():
    values = np.array([[0.1, 0.5], [0.625, 0.9]], dtype=np.float64)
    trajectory = generate_skew_tent_trajectory(0.1, 0.2, 10)

    descriptors = compute_descriptors(
        values=values,
        trajectory=trajectory,
        epsilon=1e-12,
        threshold=0.2,
    )

    assert descriptors.shape == (2, 2, 4)
    assert descriptors.dtype == np.float64
    assert np.all(np.isfinite(descriptors[:, :, 3]))
    assert np.all(descriptors[:, :, 2] >= 0)


def test_descriptor_values_for_first_match_paths():
    values = np.array([[0.1, 0.5]], dtype=np.float64)
    trajectory = np.array([0.1, 0.5, 0.625], dtype=np.float64)

    descriptors = compute_descriptors(values, trajectory, epsilon=1e-12, threshold=0.2)

    np.testing.assert_allclose(descriptors[0, 0], np.array([0.0, 0.01, 0.0, 0.0]))
    np.testing.assert_allclose(descriptors[0, 1], np.array([0.0, 0.01, 1.0, 0.0]))


def test_flatten_descriptors_converts_to_token_by_dimension_channels():
    descriptors = np.zeros((2, 3, 4), dtype=np.float64)

    flattened = flatten_descriptors(descriptors)

    assert flattened.shape == (2, 12)


def test_channel_view_returns_token_by_dimension_matrix():
    descriptors = np.zeros((2, 3, 4), dtype=np.float64)
    descriptors[:, :, 2] = 7.0

    view = channel_view(descriptors, 2)

    assert view.shape == (2, 3)
    np.testing.assert_allclose(view, np.full((2, 3), 7.0))


def test_invalid_values_outside_unit_interval_raise_value_error():
    values = np.array([[0.1, 1.2]], dtype=np.float64)
    trajectory = np.array([0.1, 0.5, 0.625], dtype=np.float64)

    with pytest.raises(ValueError):
        compute_descriptors(values, trajectory, epsilon=0.01, threshold=0.2)


@pytest.mark.parametrize(
    "values, trajectory, epsilon, threshold",
    [
        (np.array([0.1]), np.array([0.1, 0.2]), 0.01, 0.2),
        (np.array([[0.1]]), np.array([[0.1, 0.2]]), 0.01, 0.2),
        (np.array([[0.1]]), np.array([0.1, 0.2]), 0.0, 0.2),
        (np.array([[0.1]]), np.array([0.1, 0.2]), 0.01, 1.0),
        (np.array([[np.nan]]), np.array([0.1, 0.2]), 0.01, 0.2),
        (np.array([[0.1]]), np.array([np.inf, 0.2]), 0.01, 0.2),
        (np.array([[0.1]]), np.array([0.1, 0.2]), np.inf, 0.2),
        (np.array([[0.1]]), np.array([0.1, 0.2]), 0.01, np.nan),
    ],
)
def test_invalid_descriptor_inputs_raise_value_error(
    values,
    trajectory,
    epsilon,
    threshold,
):
    with pytest.raises(ValueError):
        compute_descriptors(values, trajectory, epsilon, threshold)


def test_flatten_and_channel_view_validate_dimensions():
    with pytest.raises(ValueError):
        flatten_descriptors(np.zeros((2, 3)))

    with pytest.raises(ValueError):
        channel_view(np.zeros((2, 3)), 0)

    with pytest.raises(IndexError):
        channel_view(np.zeros((2, 3, 4)), 4)
