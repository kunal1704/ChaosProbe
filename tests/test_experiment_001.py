import argparse
import json

import numpy as np
import pytest

from experiments.experiment_001_descriptor_baselines import (
    aggregate_metrics,
    build_config,
    compute_all_descriptors,
    load_prompts,
    select_prompts,
)
from chaosprobe.trajectory import generate_skew_tent_trajectory


def test_load_prompts_reads_static_prompt_file():
    prompts = load_prompts()

    assert len(prompts) == 80
    assert {prompt["category"] for prompt in prompts} == {
        "factual",
        "reasoning",
        "mathematical",
        "conversational",
    }


def test_prompts_json_is_plain_list():
    with open("experiments/prompts_001.json", "r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    assert isinstance(prompts, list)
    assert len(prompts) == 80


def test_select_prompts_limits_or_returns_all():
    prompts = [{"id": str(idx), "category": "test", "text": "text"} for idx in range(3)]

    assert len(select_prompts(prompts, 2)) == 2
    assert select_prompts(prompts, 0) == prompts

    with pytest.raises(ValueError):
        select_prompts(prompts, -1)


def test_aggregate_metrics_returns_mean_std_count():
    rows = [
        {"prompt_id": "p1", "category": "x", "method": "chaos", "channel": 0, "linear_cka": 1.0},
        {"prompt_id": "p2", "category": "x", "method": "chaos", "channel": 0, "linear_cka": 0.5},
    ]

    aggregate = aggregate_metrics(rows)

    assert aggregate == [
        {
            "method": "chaos",
            "channel": 0,
            "metric": "linear_cka",
            "mean": 0.75,
            "std": 0.25,
            "count": 2,
        }
    ]


def test_compute_all_descriptors_returns_all_methods():
    args = argparse.Namespace(
        epsilon=0.05,
        threshold=0.2,
        trajectory_len=20,
        noise_scale=0.05,
        seed=7,
    )
    values = np.array([[0.1, 0.5], [0.625, 0.9]], dtype=np.float64)
    trajectory = generate_skew_tent_trajectory(0.1, 0.2, 20)

    descriptors = compute_all_descriptors(values, trajectory, args)

    assert set(descriptors) == {
        "chaos",
        "shuffled_trajectory",
        "random_trajectory",
        "gaussian_value",
        "uniform_value",
    }
    for desc in descriptors.values():
        assert desc.shape == (2, 2, 4)


def test_build_config_is_json_serializable():
    args = argparse.Namespace(
        trajectory_len=20,
        initial_condition=0.1,
        threshold=0.2,
        epsilon=0.05,
        normalization="global",
        noise_scale=0.01,
        seed=42,
    )

    config = build_config("gpt2", [{"id": "p1", "category": "test", "text": "hello"}], args)

    json.dumps(config)
    assert config["model_name"] == "gpt2"
    assert config["prompt_count"] == 1
