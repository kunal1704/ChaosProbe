import numpy as np
import pytest

from chaosprobe.baselines import (
    gaussian_value_baseline,
    random_trajectory_descriptors,
    shuffled_trajectory_descriptors,
    uniform_value_baseline,
)
from chaosprobe.trajectory import generate_skew_tent_trajectory


@pytest.fixture
def values():
    return np.array([[0.1, 0.25, 0.5], [0.625, 0.75, 0.9]], dtype=np.float64)


@pytest.fixture
def trajectory():
    return generate_skew_tent_trajectory(0.1, 0.2, 20)


def test_each_baseline_returns_descriptor_tensor(values, trajectory):
    baselines = [
        shuffled_trajectory_descriptors(values, trajectory, 0.05, 0.2),
        random_trajectory_descriptors(values, 20, 0.05, 0.2),
        gaussian_value_baseline(values, trajectory, 0.05, 0.2),
        uniform_value_baseline(values, trajectory, 0.05, 0.2),
    ]

    for desc in baselines:
        assert desc.shape == (2, 3, 4)
        assert desc.dtype == np.float64


@pytest.mark.parametrize(
    "baseline",
    [
        shuffled_trajectory_descriptors,
        gaussian_value_baseline,
        uniform_value_baseline,
    ],
)
def test_same_seed_gives_identical_outputs(values, trajectory, baseline):
    first = baseline(values, trajectory, 0.05, 0.2, seed=7)
    second = baseline(values, trajectory, 0.05, 0.2, seed=7)

    np.testing.assert_allclose(first, second)


def test_random_trajectory_same_seed_gives_identical_outputs(values):
    first = random_trajectory_descriptors(values, 20, 0.05, 0.2, seed=7)
    second = random_trajectory_descriptors(values, 20, 0.05, 0.2, seed=7)

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    "baseline",
    [
        shuffled_trajectory_descriptors,
    ],
)
def test_different_seed_changes_stochastic_outputs(values, trajectory, baseline):
    first = baseline(values, trajectory, 0.05, 0.2, seed=1)
    second = baseline(values, trajectory, 0.05, 0.2, seed=2)

    assert not np.allclose(first, second)


@pytest.mark.parametrize("baseline", [gaussian_value_baseline, uniform_value_baseline])
def test_different_seed_changes_value_noise_outputs(values, trajectory, baseline):
    first = baseline(values, trajectory, 0.05, 0.2, noise_scale=0.2, seed=1)
    second = baseline(values, trajectory, 0.05, 0.2, noise_scale=0.2, seed=2)

    assert not np.allclose(first, second)


def test_random_trajectory_different_seed_changes_output(values):
    first = random_trajectory_descriptors(values, 20, 0.05, 0.2, seed=1)
    second = random_trajectory_descriptors(values, 20, 0.05, 0.2, seed=2)

    assert not np.allclose(first, second)


def test_shuffled_baseline_does_not_mutate_original_trajectory(values, trajectory):
    original = trajectory.copy()

    shuffled_trajectory_descriptors(values, trajectory, 0.05, 0.2, seed=7)

    np.testing.assert_allclose(trajectory, original)


@pytest.mark.parametrize("baseline", [gaussian_value_baseline, uniform_value_baseline])
def test_invalid_noise_scale_raises_value_error(values, trajectory, baseline):
    with pytest.raises(ValueError):
        baseline(values, trajectory, 0.05, 0.2, noise_scale=0.0)


def test_invalid_trajectory_length_raises_value_error(values):
    with pytest.raises(ValueError):
        random_trajectory_descriptors(values, 9, 0.05, 0.2)
