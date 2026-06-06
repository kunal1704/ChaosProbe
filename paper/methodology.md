# Methodology Outline

## Notation

Let `E ∈ R^{T × D}` denote a transformer embedding matrix, with `T` token
positions and `D` embedding dimensions. ChaosProbe treats each coordinate as a
stimulus/target value in `[0,1]` and derives a descriptor tensor of shape
`T × D × 4`.

## Algorithm Intuition

The project is based on the intuition that a coordinate can be interpreted as a
trajectory generator rather than only a scalar feature. Each coordinate will be
mapped to multiple trajectory-derived statistics so that local dynamical
information is retained.

## Planned Descriptor Tensor

The planned output is a tensor with four channels:

1. TTSS / threshold-time symbolic statistic
2. Energy
3. Time-to-match
4. Entropy

The tensor is preserved in full rather than averaged or compressed back to `D`
dimensions.

## Planned Baselines

The eventual study will compare descriptor tensors to simple baseline
representations, with an emphasis on geometry-preserving analysis rather than
task-specific optimization.

## Planned Evaluation

Evaluation will focus on representation geometry, descriptor separability, and
whether the tensor structure provides informative signal beyond simpler
summaries. Specific metrics and benchmark choices will be defined later.

