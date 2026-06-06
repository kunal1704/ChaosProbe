import numpy as np
import pytest

from chaosprobe.trajectory import (
    generate_skew_tent_trajectory,
    normalize_to_unit_interval,
)


def test_known_skew_tent_trajectory_example():
    trajectory = generate_skew_tent_trajectory(
        initial_condition=0.1,
        threshold=0.2,
        length=10,
    )

    np.testing.assert_allclose(trajectory[:3], np.array([0.1, 0.5, 0.625]))
    assert trajectory.dtype == np.float64


@pytest.mark.parametrize(
    "initial_condition, threshold, length",
    [
        (0.0, 0.2, 10),
        (1.0, 0.2, 10),
        (0.1, 0.0, 10),
        (0.1, 1.0, 10),
        (0.1, 0.2, 9),
    ],
)
def test_invalid_trajectory_parameters_raise_value_error(
    initial_condition,
    threshold,
    length,
):
    with pytest.raises(ValueError):
        generate_skew_tent_trajectory(initial_condition, threshold, length)


def test_global_normalization():
    values = np.array([[1.0, 2.0], [3.0, 5.0]])

    normalized = normalize_to_unit_interval(values, mode="global")

    expected = np.array([[0.0, 0.25], [0.5, 1.0]])
    np.testing.assert_allclose(normalized, expected)
    assert normalized.dtype == np.float64


def test_token_normalization_handles_constant_rows():
    values = np.array([[2.0, 2.0], [1.0, 3.0]])

    normalized = normalize_to_unit_interval(values, mode="token")

    expected = np.array([[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(normalized, expected)


def test_dimension_normalization_handles_constant_columns():
    values = np.array([[2.0, 1.0], [2.0, 3.0]])

    normalized = normalize_to_unit_interval(values, mode="dimension")

    expected = np.array([[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(normalized, expected)


def test_normalization_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        normalize_to_unit_interval(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        normalize_to_unit_interval(np.array([[1.0, 2.0]]), mode="sample")


@pytest.mark.parametrize(
    "values",
    [
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[-np.inf, 1.0]]),
    ],
)
def test_normalization_rejects_non_finite_inputs(values):
    with pytest.raises(ValueError):
        normalize_to_unit_interval(values)
