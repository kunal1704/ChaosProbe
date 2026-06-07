# Experiment Log Template

## Experiment ID

## Question

## Hypothesis

## Models

## Data

## Parameters

## Metrics

## Results

## Interpretation

## Caveats

## Next action

---

## Implementation Step: Core descriptor implementation

Skew-tent trajectory generation, unit-interval normalization, and the
`T x D x 4` descriptor tensor implementation were added. Tests were added for
trajectory generation, normalization, descriptor shape and dtype, entropy
finiteness, time-to-match, flattening, channel views, and invalid value
validation.

No scientific results exist yet.

Core validation cleanup: strict trajectory length validation and finite-value
checks added.

---

## Implementation Step: HuggingFace embedding extraction

Token-level input embeddings were implemented for frozen HuggingFace
transformers. GPT-2 and DistilGPT2 are supported by the interface, and no
generation or downstream task logic was added. No scientific results exist yet.

---

## Implementation Step: Baselines and representation metrics

Shuffled trajectory, random trajectory, Gaussian value, and uniform value
baselines were implemented for descriptor tensor comparisons. Channel-wise
representation metrics were added for cosine similarity, linear CKA, pairwise
distance correlation, neighborhood preservation, and anisotropy. No scientific
results exist yet.

---

## Experiment 001 scaffold

A non-plotting end-to-end pipeline was added for Experiment 001.
`prompts_001.json` now contains 80 neutral prompts across factual, reasoning,
mathematical, and conversational categories. The script supports GPT-2 and
DistilGPT2 through the CLI, computes ChaosProbe and baseline descriptor metrics,
and writes JSON plus Markdown outputs. No scientific interpretation is included
unless a successful run is reviewed separately.

---

## Experiment 001 statistical analysis

Paired chaos-vs-baseline comparisons were added for Experiment 001 outputs.
Bootstrap confidence intervals, Wilcoxon signed-rank p-values, and paired effect
sizes were added. No downstream task or safety claims were added.
