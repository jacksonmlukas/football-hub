---
name: Model change
about: Any change that alters a prediction
labels: ''
---

<!-- For a change to one of the LLM forecasters study's pinned models, use the "LLM forecaster model change" form (model-change.yml) instead. -->

## What changes

## Hypothesis
What should improve, and by how much? State it before running anything.

## Evaluation plan
- [ ] Temporal split, not random k-fold
- [ ] Benchmarked against the closing line
- [ ] Log-loss delta with bootstrap interval
- [ ] Conformal coverage checked after the change

## Gate
Which `model-eval` gate does this affect? What happens if it fails?
