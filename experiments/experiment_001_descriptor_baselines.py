"""Experiment 001: non-plotting descriptor-baseline pipeline.

This script runs the first end-to-end representation-level ChaosProbe workflow:
static prompts -> frozen input embeddings -> normalization -> trajectory
descriptors -> baseline descriptors -> channel-wise metrics -> JSON/Markdown
outputs. It does not generate text, modify prompts, plot results, or evaluate
downstream tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaosprobe.baselines import (
    gaussian_value_baseline,
    random_trajectory_descriptors,
    shuffled_trajectory_descriptors,
    uniform_value_baseline,
)
from chaosprobe.descriptors import compute_descriptors
from chaosprobe.embeddings import get_token_embeddings
from chaosprobe.metrics import compute_channel_metrics, descriptor_channel_summary
from chaosprobe.trajectory import generate_skew_tent_trajectory, normalize_to_unit_interval


PROMPT_PATH = Path(__file__).with_name("prompts_001.json")
DEFAULT_OUTPUT_DIR = Path("outputs") / "experiment_001"
METHODS = [
    "chaos",
    "shuffled_trajectory",
    "random_trajectory",
    "gaussian_value",
    "uniform_value",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI settings for a lightweight or full Experiment 001 run."""

    parser = argparse.ArgumentParser(description="Run ChaosProbe Experiment 001.")
    parser.add_argument("--models", nargs="+", default=["gpt2"], choices=["gpt2", "distilgpt2"])
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--trajectory-len", type=int, default=1000)
    parser.add_argument("--initial-condition", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--normalization",
        default="global",
        choices=["global", "token", "dimension"],
    )
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_prompts(path: Path = PROMPT_PATH) -> list[dict[str, str]]:
    """Load and validate the static neutral prompt dataset."""

    with path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts file must contain a non-empty list")

    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("each prompt must be an object")
        for key in ["id", "category", "text"]:
            if not isinstance(prompt.get(key), str) or not prompt[key].strip():
                raise ValueError(f"each prompt must contain a non-empty {key!r}")

    return prompts


def select_prompts(
    prompts: list[dict[str, str]],
    max_prompts: int,
) -> list[dict[str, str]]:
    """Select a deterministic prefix of prompts for smoke runs."""

    if max_prompts < 0:
        raise ValueError("max_prompts must be greater than or equal to 0")
    if max_prompts == 0:
        return prompts
    return prompts[:max_prompts]


def aggregate_metrics(per_prompt_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-prompt metric rows into mean/std/count summaries."""

    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    identity_keys = {"prompt_id", "category", "method", "channel"}

    for row in per_prompt_metrics:
        for key, value in row.items():
            if key in identity_keys:
                continue
            grouped[(row["method"], row["channel"], key)].append(float(value))

    aggregates = []
    for (method, channel, metric), values in sorted(grouped.items()):
        value_array = np.asarray(values, dtype=np.float64)
        aggregates.append(
            {
                "method": method,
                "channel": int(channel),
                "metric": metric,
                "mean": float(np.mean(value_array)),
                "std": float(np.std(value_array)),
                "count": int(value_array.size),
            }
        )
    return aggregates


def run_model(
    model_name: str,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
) -> Path:
    """Run Experiment 001 for one model and write all output files."""

    output_dir = args.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_result = get_token_embeddings(
        [prompt["text"] for prompt in prompts],
        model_name=model_name,
        layer="input",
        device="cpu",
    )
    trajectory = generate_skew_tent_trajectory(
        args.initial_condition,
        args.threshold,
        args.trajectory_len,
    )

    per_prompt_metrics: list[dict[str, Any]] = []
    descriptor_summaries: list[dict[str, Any]] = []
    sample_tokens: list[dict[str, Any]] = []

    for prompt, item in zip(prompts, embedding_result["items"]):
        normalized = normalize_to_unit_interval(item["embedding"], mode=args.normalization)
        descriptors_by_method = compute_all_descriptors(
            normalized,
            trajectory,
            args,
        )

        # Keep a small token sample for auditability without storing embeddings.
        if len(sample_tokens) < 3:
            sample_tokens.append(
                {
                    "id": prompt["id"],
                    "category": prompt["category"],
                    "text": prompt["text"],
                    "tokens": item["tokens"],
                }
            )

        for method, descriptors in descriptors_by_method.items():
            for metrics in compute_channel_metrics(normalized, descriptors):
                per_prompt_metrics.append(
                    {
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "method": method,
                        **metrics,
                    }
                )
            descriptor_summaries.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "method": method,
                    "channels": descriptor_channel_summary(descriptors),
                }
            )

    config = build_config(model_name, prompts, args)
    aggregate = aggregate_metrics(per_prompt_metrics)

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "sample_tokens.json", sample_tokens)
    write_json(output_dir / "per_prompt_metrics.json", per_prompt_metrics)
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "descriptor_summaries.json", descriptor_summaries)
    write_summary_markdown(output_dir / "summary.md", config)
    return output_dir


def compute_all_descriptors(
    normalized: np.ndarray,
    trajectory: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Compute ChaosProbe and baseline descriptor tensors for one prompt."""

    return {
        "chaos": compute_descriptors(
            normalized,
            trajectory,
            args.epsilon,
            args.threshold,
        ),
        "shuffled_trajectory": shuffled_trajectory_descriptors(
            normalized,
            trajectory,
            args.epsilon,
            args.threshold,
            seed=args.seed,
        ),
        "random_trajectory": random_trajectory_descriptors(
            normalized,
            args.trajectory_len,
            args.epsilon,
            args.threshold,
            seed=args.seed,
        ),
        "gaussian_value": gaussian_value_baseline(
            normalized,
            trajectory,
            args.epsilon,
            args.threshold,
            noise_scale=args.noise_scale,
            seed=args.seed,
        ),
        "uniform_value": uniform_value_baseline(
            normalized,
            trajectory,
            args.epsilon,
            args.threshold,
            noise_scale=args.noise_scale,
            seed=args.seed,
        ),
    }


def build_config(
    model_name: str,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build JSON-serializable run metadata for one model output directory."""

    return {
        "experiment_id": "experiment_001_descriptor_baselines",
        "model_name": model_name,
        "prompt_count": len(prompts),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "trajectory_len": int(args.trajectory_len),
            "initial_condition": float(args.initial_condition),
            "threshold": float(args.threshold),
            "epsilon": float(args.epsilon),
            "normalization": args.normalization,
            "noise_scale": float(args.noise_scale),
            "seed": int(args.seed),
        },
        "methods": METHODS,
        "caveat": "Representation-level experiment only; no downstream task or safety result.",
    }


def write_json(path: Path, payload: Any) -> None:
    """Write indented JSON using the standard library encoder."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_summary_markdown(path: Path, config: dict[str, Any]) -> None:
    """Write a human-readable run summary without scientific interpretation."""

    params = config["parameters"]
    lines = [
        "# Experiment 001 Descriptor Baselines",
        "",
        "## What was tested",
        "A non-plotting representation-level comparison between ChaosProbe descriptors and baseline descriptor constructions.",
        "",
        "## Models used",
        f"- {config['model_name']}",
        "",
        "## Prompt count",
        f"- {config['prompt_count']}",
        "",
        "## Parameters",
        f"- trajectory_len: {params['trajectory_len']}",
        f"- initial_condition: {params['initial_condition']}",
        f"- threshold: {params['threshold']}",
        f"- epsilon: {params['epsilon']}",
        f"- normalization: {params['normalization']}",
        f"- noise_scale: {params['noise_scale']}",
        f"- seed: {params['seed']}",
        "",
        "## Methods compared",
        *[f"- {method}" for method in config["methods"]],
        "",
        "## Metrics computed",
        "- mean_cosine_similarity",
        "- linear_cka",
        "- pairwise_distance_correlation",
        "- neighborhood_preservation",
        "- anisotropy_original",
        "- anisotropy_channel",
        "",
        "## Caveat",
        "Representation-level experiment only, no downstream task or safety result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    prompts = select_prompts(load_prompts(), args.max_prompts)
    for model_name in args.models:
        run_model(model_name, prompts, args)


if __name__ == "__main__":
    main()
