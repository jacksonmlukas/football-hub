---
name: Model change
about: Any change that alters a prediction
labels: ''
---

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
