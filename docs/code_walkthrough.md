# ChaosProbe Code Walkthrough

## 1. Repository purpose

ChaosProbe is a neurochaos-inspired representation analysis framework for
transformer embeddings. It treats token-level embedding coordinates as values in
the unit interval, derives trajectory-based descriptor tensors, and compares
those tensors against controlled baselines. ChaosProbe does not generate text,
does not modify LLM outputs, and does not make downstream task or safety claims.

## 2. Prompt dataset

Experiment 001 uses `experiments/prompts_001.json`, a static set of 80 neutral
prompts split evenly across factual, reasoning, mathematical, and conversational
categories. The prompts are deterministic fixtures, not generated at runtime,
and they are not adversarial or safety-oriented.

## 3. Embedding extraction

`chaosprobe/embeddings.py` loads supported HuggingFace models (`gpt2` and
`distilgpt2`) and extracts token-level input embeddings with
`model.get_input_embeddings()(input_ids)`. The model is set to evaluation mode,
the extraction runs under `torch.no_grad()`, and the returned embedding matrix
for each prompt has shape `T x D` with dtype `float64`.

## 4. Normalization

`normalize_to_unit_interval()` in `chaosprobe/trajectory.py` maps each embedding
matrix into `[0, 1]`. The current modes are global matrix scaling, per-token row
scaling, and per-dimension column scaling. Constant groups map to zeros so the
downstream descriptor extraction stays finite.

## 5. Chaotic trajectory generation

`generate_skew_tent_trajectory()` creates the deterministic skew-tent reference
trajectory. Starting from an initial condition and threshold, each next value is
computed by the two-branch skew-tent recurrence. This trajectory is the path
against which normalized embedding coordinates are matched.

## 6. Descriptor extraction

`chaosprobe/descriptors.py` computes the `T x D x 4` descriptor tensor. For each
normalized coordinate, the code finds the first trajectory index within
`epsilon`, takes the path prefix, and computes four channels: TTSS, energy,
time-to-match, and entropy. The descriptor tensor is preserved and is not
averaged back into `D` dimensions.

## 7. Baseline descriptor construction

`chaosprobe/baselines.py` builds comparison descriptor tensors with the same
shape as the ChaosProbe descriptors. The current baselines use a shuffled
trajectory, a seeded random trajectory, Gaussian value perturbations, and uniform
value perturbations. These are reference constructions for representation-level
comparison, not downstream model interventions.

## 8. Representation metrics

`chaosprobe/metrics.py` compares `T x D` matrices. Experiment 001 compares the
normalized original embedding matrix against each descriptor channel using
mean cosine similarity, linear CKA, pairwise distance correlation, neighborhood
preservation, and anisotropy. Metric outputs are plain Python floats for JSON
serialization.

## 9. Experiment 001 pipeline

`experiments/experiment_001_descriptor_baselines.py` runs the non-plotting
pipeline. It loads prompts, extracts embeddings, normalizes each embedding
matrix, generates a skew-tent trajectory, computes ChaosProbe and baseline
descriptors, computes channel-wise metrics, aggregates across prompts, and
writes JSON plus Markdown outputs.

## 10. Statistical analysis pipeline

`experiments/analyze_experiment_001.py` reads `per_prompt_metrics.json` outputs
and performs paired chaos-vs-baseline comparisons across prompts. It computes
mean differences, bootstrap confidence intervals, Wilcoxon signed-rank
p-values, paired Cohen's dz effect sizes, Benjamini-Hochberg FDR-corrected
q-values, and compact primary-result tables. This is representation-level
statistics only.

## 11. Output files

Experiment 001 writes per-model outputs under `outputs/experiment_001/<model>/`:
`config.json`, `sample_tokens.json`, `per_prompt_metrics.json`,
`aggregate_metrics.json`, `descriptor_summaries.json`, and `summary.md`.
Statistical analysis writes per-model outputs under
`outputs/experiment_001_stats/<model>/` plus
`outputs/experiment_001_stats/combined_statistical_comparisons.json`.

## 12. What is not implemented yet

ChaosProbe currently does not implement plotting, additional transformer
families, hidden-state extraction, downstream classifiers, LLM generation, prompt
manipulation, jailbreak tests, prompt-injection tests, robustness claims, or
safety claims. Current results are representation-level only and require further
statistical hygiene before paper writing.
