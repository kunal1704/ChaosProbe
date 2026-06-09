import argparse
import json

import numpy as np

from experiments.analyze_experiment_001 import (
    add_global_fdr_fields,
    add_model_fdr_fields,
    analyze_rows,
    benjamini_hochberg,
    bootstrap_mean_difference_ci,
    group_metric_values,
    paired_comparison,
    write_primary_results_table,
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


def test_benjamini_hochberg_known_toy_values():
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

    np.testing.assert_allclose(q_values, [0.02, 0.04, 0.04, 0.008])


def test_benjamini_hochberg_monotonicity_and_order_behavior():
    p_values = [0.20, 0.01, 0.04, 0.03]
    q_values = benjamini_hochberg(p_values)
    order = np.argsort(p_values)
    sorted_q = np.asarray(q_values)[order]

    assert np.all(sorted_q[:-1] <= sorted_q[1:])
    assert q_values[1] <= q_values[0]


def test_benjamini_hochberg_handles_empty_zero_and_one():
    assert benjamini_hochberg([]) == []
    q_values = benjamini_hochberg([0.0, 1.0])

    assert q_values == [0.0, 1.0]


def test_fdr_fields_are_added_and_respect_alpha():
    rows = [
        {"p_value": 0.01},
        {"p_value": 0.20},
    ]

    add_model_fdr_fields(rows, alpha=0.05)
    add_global_fdr_fields(rows, alpha=0.05)

    assert rows[0]["p_value_fdr_bh_model"] == 0.02
    assert rows[0]["significant_fdr_bh_model"] is True
    assert rows[0]["p_value_fdr_bh_global"] == 0.02
    assert rows[0]["significant_fdr_bh_global"] is True
    assert rows[1]["significant_fdr_bh_model"] is False
    assert rows[1]["significant_fdr_bh_global"] is False


def test_compact_table_formatting_works_on_toy_rows(tmp_path):
    rows = [
        {
            "baseline": "shuffled_trajectory",
            "channel": 0,
            "metric": "anisotropy_channel",
            "chaos_mean": 0.1,
            "baseline_mean": 0.2,
            "mean_difference": -0.1,
            "ci_low": -0.2,
            "ci_high": -0.05,
            "p_value": 0.001,
            "p_value_fdr_bh_model": 0.002,
            "p_value_fdr_bh_global": 0.003,
            "cohens_dz": -1.23456,
            "significant_fdr_bh_global": True,
        },
        {
            "baseline": "gaussian_value",
            "channel": 0,
            "metric": "anisotropy_channel",
            "chaos_mean": 0.1,
            "baseline_mean": 0.2,
            "mean_difference": -0.1,
            "ci_low": -0.2,
            "ci_high": -0.05,
            "p_value": 0.001,
            "p_value_fdr_bh_model": 0.002,
            "p_value_fdr_bh_global": 0.003,
            "cohens_dz": -1.23456,
            "significant_fdr_bh_global": True,
        },
    ]
    output_path = tmp_path / "table.md"

    write_primary_results_table(output_path, rows)
    table = output_path.read_text(encoding="utf-8")

    assert "shuffled_trajectory" in table
    assert "gaussian_value" not in table
    assert "p_value_fdr_bh_global" in table
    assert "True" in table
