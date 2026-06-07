import argparse
import json

import numpy as np

from experiments.analyze_experiment_001 import (
    analyze_rows,
    bootstrap_mean_difference_ci,
    group_metric_values,
    paired_comparison,
    wilcoxon_p_value,
)


def toy_rows():
    rows = []
    for prompt_id, chaos_value, baseline_value in [
        ("p1", 3.0, 1.0),
        ("p2", 5.0, 2.0),
        ("p3", 7.0, 3.0),
    ]:
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": "toy",
                "method": "chaos",
                "channel": 0,
                "linear_cka": chaos_value,
                "anisotropy_original": 9.0,
            }
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": "toy",
                "method": "random_trajectory",
                "channel": 0,
                "linear_cka": baseline_value,
                "anisotropy_original": 9.0,
            }
        )
    return rows


def test_paired_comparison_computes_expected_mean_difference():
    comparison = paired_comparison(
        model="gpt2",
        baseline="random_trajectory",
        channel=0,
        metric="linear_cka",
        chaos_values=[3.0, 5.0, 7.0],
        baseline_values=[1.0, 2.0, 3.0],
        bootstrap_samples=100,
        seed=7,
    )

    assert comparison["mean_difference"] == 3.0
    assert comparison["chaos_mean"] == 5.0
    assert comparison["baseline_mean"] == 2.0
    assert comparison["count"] == 3


def test_bootstrap_ci_returns_lower_and_upper_bounds():
    lower, upper = bootstrap_mean_difference_ci(
        np.array([1.0, 2.0, 3.0]),
        bootstrap_samples=200,
        seed=7,
    )

    assert lower <= upper
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_zero_differences_are_handled_gracefully():
    comparison = paired_comparison(
        model="gpt2",
        baseline="uniform_value",
        channel=1,
        metric="anisotropy_channel",
        chaos_values=[1.0, 1.0, 1.0],
        baseline_values=[1.0, 1.0, 1.0],
        bootstrap_samples=100,
        seed=42,
    )

    assert comparison["mean_difference"] == 0.0
    assert comparison["p_value"] == 1.0
    assert comparison["cohens_dz"] == 0.0
    assert wilcoxon_p_value(np.zeros(3)) == 1.0


def test_paired_comparison_is_json_serializable():
    comparison = paired_comparison(
        model="gpt2",
        baseline="random_trajectory",
        channel=0,
        metric="linear_cka",
        chaos_values=[2.0, 4.0],
        baseline_values=[1.0, 3.0],
        bootstrap_samples=100,
        seed=42,
    )

    json.dumps(comparison)


def test_grouping_by_method_channel_metric_works_on_toy_rows():
    grouped = group_metric_values(toy_rows())

    assert ("chaos", 0, "linear_cka") in grouped
    assert ("random_trajectory", 0, "linear_cka") in grouped
    assert ("chaos", 0, "anisotropy_original") not in grouped
    assert grouped[("chaos", 0, "linear_cka")]["p1"] == 3.0


def test_analyze_rows_builds_paired_comparison_from_groups():
    comparisons = analyze_rows(
        toy_rows(),
        model="gpt2",
        bootstrap_samples=100,
        seed=42,
    )

    assert len(comparisons) == 1
    assert comparisons[0]["baseline"] == "random_trajectory"
    assert comparisons[0]["metric"] == "linear_cka"
