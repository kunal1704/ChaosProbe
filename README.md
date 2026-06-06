# ChaosProbe

ChaosProbe is a neurochaos-inspired representation analysis framework for transformer embeddings.

It does **not**:

- generate text
- modify LLM outputs
- claim robustness
- claim safety improvement
- perform jailbreak or prompt-injection experiments

It **does**:

- extract trajectory descriptors from transformer embeddings
- preserve descriptor channels as `T × D × 4`
- compare descriptor spaces to baselines
- support representation geometry analysis

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate ChaosProbe
```

## Future experiment entry point

The initial experiment script is scaffolded but not implemented yet:

```bash
python experiments/experiment_001_descriptor_baselines.py
```

## Project structure

- `chaosprobe/` contains the future Python package for descriptors, trajectories, metrics, and utilities.
- `experiments/` contains experiment entry points and prompt fixtures.
- `docs/` contains working methodology notes and experiment logging templates.
- `paper/` contains rough draft sections for the eventual manuscript.

