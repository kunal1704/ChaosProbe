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
