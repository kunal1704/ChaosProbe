# ChaosProbe Research V1 Status

## Current research question

Are ChaosProbe trajectory descriptor tensors distinguishable from baseline
descriptor constructions when applied to frozen transformer input embeddings?

## What has been implemented

- Core skew-tent trajectory generation and unit-interval normalization.
- Descriptor extraction into preserved `T x D x 4` tensors.
- HuggingFace input embedding extraction for GPT-2 and DistilGPT2.
- Four descriptor baselines: shuffled trajectory, random trajectory, Gaussian
  value perturbation, and uniform value perturbation.
- Representation metrics and an Experiment 001 non-plotting pipeline.
- Paired statistical analysis with bootstrap confidence intervals, Wilcoxon
  signed-rank p-values, and paired effect sizes.

## What has been tested

The repository includes pytest coverage for trajectory generation,
normalization, descriptor extraction, HuggingFace API validation, baselines,
metrics, Experiment 001 helpers, and statistical-analysis helpers. Tests use
deterministic NumPy fixtures where possible and avoid requiring model downloads
for unit coverage.

## What results exist so far

Experiment 001 has generated representation-level outputs for GPT-2 and
DistilGPT2 in the local ignored `outputs/` directory. Statistical summaries have
also been generated locally. These outputs are useful for review, but they are
not committed because generated experiment outputs remain gitignored.

## Current strongest finding

The current local summaries show large paired differences for several
chaos-vs-baseline comparisons, especially anisotropy-related descriptor-channel
metrics against shuffled and random trajectory baselines. This should be treated
as preliminary representation-level evidence, not as a finalized scientific
claim.

## Current caveats

- FDR or other multiple-comparison correction is still pending.
- The current metrics are representation-level and do not evaluate downstream
  task performance.
- The current code does not test safety, robustness, jailbreak behavior, or
  prompt-injection behavior.
- Only input embeddings are analyzed; hidden-state layers are not implemented.
- The prompt set is small and neutral, designed for a first controlled pass.

## Next required steps before paper writing

- Add multiple-comparison correction such as FDR for Experiment 001 statistics.
- Decide which metrics are primary before interpreting results.
- Re-run experiments with fixed environment metadata and archived outputs.
- Review prompt coverage and model coverage before expanding claims.
- Write a results section only after statistical hygiene and reproducibility
  checks are complete.
