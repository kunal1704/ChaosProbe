# ChaosProbe Methodology

## What ChaosProbe Is

ChaosProbe is a neurochaos-inspired dynamical representation analysis framework
for transformer embeddings. Its purpose is to study the geometry and structure
of embeddings by transforming each coordinate into a set of trajectory-derived
descriptors.

## Neurochaos Inspiration

ChaosProbe is inspired by Neurochaos Learning and related ChaosFEX-style ideas
because those methods emphasize rich dynamical summaries rather than simple
static compression. The guiding assumption is that a coordinate-wise trajectory
can encode information that is not visible in a direct pointwise projection.

## Why Trajectory Descriptors

For each coordinate in a transformer embedding matrix `E ∈ R^{T × D}`, ChaosProbe
will eventually derive four descriptors:

- TTSS / threshold-time symbolic statistic
- Energy
- Time-to-match
- Entropy

These descriptors are designed to summarize trajectory behavior from a stimulus
value in `[0,1]` while preserving local dynamical structure.

## Why the Descriptor Tensor Is Preserved

The central object of study is the descriptor tensor with shape `T × D × 4`.
ChaosProbe does not collapse the descriptor channels back into `D` dimensions and
does not average them away. Preserving the full tensor keeps the four descriptor
channels separable so that downstream analysis can compare them directly.

## What Experiment 001 Will Eventually Test

Experiment 001 will eventually compare descriptor-space structure against simple
baselines to ask whether the resulting tensor exposes meaningful geometry in
transformer embeddings. The first experiment is intended to establish the
analysis pipeline, validate the descriptor tensor shape, and define the initial
comparison protocol.

