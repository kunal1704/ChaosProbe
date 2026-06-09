"""Statistical analysis for Experiment 001 descriptor-baseline metrics.

This script reads Experiment 001 ``per_prompt_metrics.json`` files and performs
paired chaos-vs-baseline comparisons across prompts. It writes representation-
level statistical summaries only; it does not run models, generate text, plot
results, or evaluate downstream performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASELINES = [
    "shuffled_trajectory",
    "random_trajectory",
    "gaussian_value",
    "uniform_value",
]
EXCLUDED_METRICS = {"anisotropy_original"}
PRIMARY_METRICS = [
    "anisotropy_channel",
    "mean_cosine_similarity",
    "linear_cka",
    "pairwise_distance_correlation",
    "neighborhood_preservation",
]
PRIMARY_REPORTING_METRICS = [
    "anisotropy_channel",
    "mean_cosine_similarity",
]
SECONDARY_REPORTING_METRICS = [
    "linear_cka",
    "pairwise_distance_correlation",
    "neighborhood_preservation",
]
CONTROL_BASELINES = {"shuffled_trajectory", "random_trajectory"}


def parse_args() -> argparse.Namespace:
    """Parse CLI settings for Experiment 001 statistical analysis."""

    parser = argparse.ArgumentParser(description="Analyze Experiment 001 outputs.")
    parser.add_argument("--models", nargs="+", default=["gpt2", "distilgpt2"])
    parser.add_argument("--input-dir", type=Path, default=Path("outputs") / "experiment_001")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "experiment_001_stats",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def load_per_prompt_metrics(input_dir: Path, model: str) -> list[dict[str, Any]]:
    """Load per-prompt metric rows for one model output directory."""

    path = input_dir / model / "per_prompt_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing per-prompt metrics file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"per-prompt metrics file must contain a non-empty list: {path}")
    return rows


def group_metric_values(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, float]]:
    """Group metric values by method, channel, metric, and prompt id."""

    grouped: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    identity_keys = {"prompt_id", "category", "method", "channel"}

    for row in rows:
        method = row["method"]
        channel = int(row["channel"])
        prompt_id = row["prompt_id"]
        for key, value in row.items():
            if key in identity_keys or key in EXCLUDED_METRICS:
                continue
            grouped[(method, channel, key)][prompt_id] = float(value)

    return dict(grouped)


def paired_comparison(
    model: str,
    baseline: str,
    channel: int,
    metric: str,
    chaos_values: list[float],
    baseline_values: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute paired mean difference, bootstrap CI, Wilcoxon p, and Cohen dz."""

    chaos_array = np.asarray(chaos_values, dtype=np.float64)
    baseline_array = np.asarray(baseline_values, dtype=np.float64)
    if chaos_array.shape != baseline_array.shape:
        raise ValueError("paired arrays must have the same shape")

    differences = chaos_array - baseline_array
    ci_low, ci_high = bootstrap_mean_difference_ci(differences, bootstrap_samples, seed)

    return {
        "model": model,
        "baseline": baseline,
        "channel": int(channel),
        "metric": metric,
        "chaos_mean": float(np.mean(chaos_array)),
        "baseline_mean": float(np.mean(baseline_array)),
        "mean_difference": float(np.mean(differences)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": wilcoxon_p_value(differences),
        "cohens_dz": cohens_dz(differences),
        "count": int(differences.size),
    }


def bootstrap_mean_difference_ci(
    differences: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a percentile confidence interval for paired differences."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than 0")
    if differences.size == 0:
        raise ValueError("differences must not be empty")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(bootstrap_samples, differences.size))
    sampled_means = np.mean(differences[indices], axis=1)
    lower, upper = np.percentile(sampled_means, [2.5, 97.5])
    return float(lower), float(upper)


def wilcoxon_p_value(differences: np.ndarray) -> float:
    """Return a two-sided Wilcoxon p-value with zero-difference fallback."""

    if differences.size == 0 or np.all(differences == 0):
        return 1.0
    try:
        result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return 1.0
    if not np.isfinite(result.pvalue):
        return 1.0
    return float(result.pvalue)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted q-values in input order."""

    if not p_values:
        return []

    p_array = np.asarray(p_values, dtype=np.float64)
    if not np.all(np.isfinite(p_array)):
        raise ValueError("p_values must contain only finite values")
    if np.any(p_array < 0) or np.any(p_array > 1):
        raise ValueError("p_values must lie in [0, 1]")

    n_values = p_array.size
    order = np.argsort(p_array)
    sorted_p = p_array[order]
    ranks = np.arange(1, n_values + 1, dtype=np.float64)
    sorted_q = sorted_p * n_values / ranks

    # Enforce monotonicity from largest to smallest sorted p-value.
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    q_values = np.empty_like(sorted_q)
    q_values[order] = sorted_q
    return [float(value) for value in q_values]


def cohens_dz(differences: np.ndarray) -> float:
    """Compute Cohen's dz for paired samples."""

    if differences.size < 2:
        return 0.0
    std = np.std(differences, ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(np.mean(differences) / std)


def add_model_fdr_fields(
    comparisons: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    """Add within-model BH q-values and significance flags."""

    q_values = benjamini_hochberg([row["p_value"] for row in comparisons])
    for row, q_value in zip(comparisons, q_values):
        row["p_value_fdr_bh_model"] = float(q_value)
        row["significant_fdr_bh_model"] = bool(q_value <= alpha)
    return comparisons


def add_global_fdr_fields(
    comparisons: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    """Add global BH q-values and significance flags across all models."""

    q_values = benjamini_hochberg([row["p_value"] for row in comparisons])
    for row, q_value in zip(comparisons, q_values):
        row["p_value_fdr_bh_global"] = float(q_value)
        row["significant_fdr_bh_global"] = bool(q_value <= alpha)
    return comparisons


def analyze_rows(
    rows: list[dict[str, Any]],
    model: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build all chaos-vs-baseline comparisons for one model."""

    grouped = group_metric_values(rows)
    comparisons: list[dict[str, Any]] = []

    chaos_keys = [
        key for key in grouped if key[0] == "chaos" and key[2] in PRIMARY_METRICS
    ]
    for _, channel, metric in sorted(chaos_keys):
        chaos_by_prompt = grouped[("chaos", channel, metric)]
        for baseline in BASELINES:
            baseline_by_prompt = grouped.get((baseline, channel, metric), {})
            prompt_ids = sorted(set(chaos_by_prompt).intersection(baseline_by_prompt))
            if not prompt_ids:
                continue

            comparisons.append(
                paired_comparison(
                    model=model,
                    baseline=baseline,
                    channel=channel,
                    metric=metric,
                    chaos_values=[chaos_by_prompt[prompt_id] for prompt_id in prompt_ids],
                    baseline_values=[
                        baseline_by_prompt[prompt_id] for prompt_id in prompt_ids
                    ],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
            )

    return comparisons


def prompt_count(rows: list[dict[str, Any]]) -> int:
    """Count unique prompts represented in per-prompt metric rows."""

    return len({row["prompt_id"] for row in rows})


def write_json(path: Path, payload: Any) -> None:
    """Write JSON output, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_summary(
    path: Path,
    model: str,
    count: int,
    comparisons: list[dict[str, Any]],
    alpha: float,
) -> None:
    """Write a compact Markdown summary for one model's statistical output."""

    significant = [
        row for row in strongest_by_effect(comparisons)
        if row.get("significant_fdr_bh_global")
    ]
    strongest_primary = [
        row for row in strongest_by_effect(comparisons)
        if row["metric"] in PRIMARY_REPORTING_METRICS
    ][:5]

    lines = [
        "# Experiment 001 Statistical Analysis",
        "",
        "## What was compared",
        "Paired chaos-vs-baseline representation metrics across prompts.",
        "",
        "## Model",
        f"- {model}",
        "",
        "## Number of prompts",
        f"- {count}",
        "",
        "## Multiple-comparison correction",
        f"- alpha: {format_number(alpha)}",
        "- Benjamini-Hochberg FDR correction was applied within each model and globally across requested models.",
        "",
        "## Strongest globally FDR-significant effects",
        *format_rows(significant[:5]),
        "",
        "## Strongest primary-reporting-metric effects",
        *format_rows(strongest_primary),
        "",
        "## Caveat",
        "Statistical analysis is representation-level only, not downstream performance.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def strongest_by_effect(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort comparison rows by absolute paired effect size."""

    return sorted(comparisons, key=lambda row: abs(row["cohens_dz"]), reverse=True)


def format_number(value: float) -> str:
    """Format compact table values with four significant digits."""

    return f"{float(value):.4g}"


def format_ci(row: dict[str, Any]) -> str:
    """Format a 95% confidence interval for Markdown tables."""

    return f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"


def format_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Format comparison rows for Markdown summaries."""

    if not rows:
        return ["- No comparisons available."]

    return [
        (
            f"- baseline={row['baseline']}, channel={row['channel']}, "
            f"metric={row['metric']}, mean_difference={row['mean_difference']:.6g}, "
            f"cohens_dz={row['cohens_dz']:.6g}, "
            f"q_global={row.get('p_value_fdr_bh_global', float('nan')):.6g}"
        )
        for row in rows
    ]


def primary_results_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select compact primary-reporting rows for shuffled/random controls."""

    return [
        row for row in comparisons
        if row["baseline"] in CONTROL_BASELINES
        and row["metric"] in PRIMARY_REPORTING_METRICS
    ]


def write_primary_results_table(path: Path, comparisons: list[dict[str, Any]]) -> None:
    """Write a compact Markdown table of primary reporting comparisons."""

    rows = sorted(
        primary_results_rows(comparisons),
        key=lambda row: (row["channel"], row["metric"], row["baseline"]),
    )
    lines = [
        "| channel | metric | baseline | chaos_mean | baseline_mean | mean_difference | 95% CI | p_value | p_value_fdr_bh_model | p_value_fdr_bh_global | cohens_dz | significant_fdr_bh_global |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["channel"]),
                    row["metric"],
                    row["baseline"],
                    format_number(row["chaos_mean"]),
                    format_number(row["baseline_mean"]),
                    format_number(row["mean_difference"]),
                    format_ci(row),
                    format_number(row["p_value"]),
                    format_number(row["p_value_fdr_bh_model"]),
                    format_number(row["p_value_fdr_bh_global"]),
                    format_number(row["cohens_dz"]),
                    str(bool(row["significant_fdr_bh_global"])),
                ]
            )
            + " |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_fdr_effects(path: Path, comparisons: list[dict[str, Any]]) -> None:
    """Write compact top-effect sections for globally significant rows."""

    significant = [
        row for row in strongest_by_effect(comparisons)
        if row.get("significant_fdr_bh_global")
    ]
    anisotropy = [
        row for row in significant if row["metric"] == "anisotropy_channel"
    ][:10]
    controls = [
        row for row in significant if row["baseline"] in CONTROL_BASELINES
    ][:10]

    lines = [
        "# Top FDR-Corrected Effects",
        "",
        "## Top 10 effects by absolute Cohen's dz",
        *format_rows(significant[:10]),
        "",
        "## Top 10 anisotropy_channel effects",
        *format_rows(anisotropy),
        "",
        "## Top 10 shuffled/random trajectory control effects",
        *format_rows(controls),
        "",
        "## Caveat",
        "Representation-level only, not downstream performance.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """Run analysis for all requested models and collect missing-output notes."""

    combined: list[dict[str, Any]] = []
    missing: list[str] = []
    model_outputs: list[tuple[str, int, list[dict[str, Any]]]] = []

    for model in args.models:
        try:
            rows = load_per_prompt_metrics(args.input_dir, model)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        comparisons = analyze_rows(
            rows=rows,
            model=model,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        add_model_fdr_fields(comparisons, args.alpha)
        model_outputs.append((model, prompt_count(rows), comparisons))
        combined.extend(comparisons)

    if combined:
        add_global_fdr_fields(combined, args.alpha)
        for model, count, comparisons in model_outputs:
            model_output_dir = args.output_dir / model
            write_json(model_output_dir / "statistical_comparisons.json", comparisons)
            write_json(
                model_output_dir / "fdr_statistical_comparisons.json",
                comparisons,
            )
            write_primary_results_table(
                model_output_dir / "primary_results_table.md",
                comparisons,
            )
            write_top_fdr_effects(model_output_dir / "top_fdr_effects.md", comparisons)
            write_summary(
                model_output_dir / "summary.md",
                model,
                count,
                comparisons,
                args.alpha,
            )

        write_json(args.output_dir / "combined_statistical_comparisons.json", combined)
        write_json(
            args.output_dir / "combined_fdr_statistical_comparisons.json",
            combined,
        )
        write_primary_results_table(
            args.output_dir / "combined_primary_results_table.md",
            combined,
        )
        write_top_fdr_effects(args.output_dir / "combined_top_fdr_effects.md", combined)

    return combined, missing


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    combined, missing = run_analysis(args)
    for message in missing:
        print(message, file=sys.stderr)
    if not combined:
        raise SystemExit("no statistical comparisons were written")


if __name__ == "__main__":
    main()
