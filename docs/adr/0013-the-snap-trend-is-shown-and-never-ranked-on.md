# The snap trend is shown, and never ranked on

**Status:** accepted 2026-08-25.

**Decision.** `docs/snap-trend-signal.md` establishes a real signal. It may be **displayed**
beside a waiver decision. It may **not** become the sort order of one, and no module ranks on
it.

## The signal is real

Snap share over weeks *w−1..w* against weeks *1..w−2*, predicting points in *w+1..w+3*,
2022-25: partial **r = +0.236** at anchor week 10 controlling for season-to-date PPG *and*
rest-of-season ECR. Twelve of twelve season-anchor cells positive. Adding the ECR control
changes it by less than 0.01, so consensus prices none of it. A 200-permutation placebo centres
on zero. It is stronger in the waiver-relevant tail.

This is the first of six screens in this repo to survive.

## Ranking on it is worse than not having it

The obvious use — each week take the best available player by snap delta — was gated against
the obvious alternative, best available by consensus rank. Forty paired decisions, players
owned in under 50% of leagues, scored on the next three weeks:

    top-1 by snap delta   -3.81 points vs by ECR, wins 1/4 seasons
    top-3 by snap delta   -3.35 points vs by ECR, wins 1/4 seasons

**Both arms are worse than the thing they were supposed to improve on.** The reason is
structural rather than statistical, and it is the whole point of this ADR: the screen measured
a *partial* correlation. It says snap delta adds information **given** ECR. Ranking by delta
alone discards ECR, which is not what it said and never was.

## The rule it does support cannot be validated

Using ECR to choose the candidate set and delta to choose within it — which is what a partial
correlation actually implies:

| rule | vs best-by-ECR | se | seasons won |
|---|---|---|---|
| top-5 by ECR, then best delta | +1.32 | 0.5 | 2/4 |
| top-10 by ECR, then best delta | +0.66 | 0.2 | 3/4 |
| top-20 by ECR, then best delta | +0.95 | 0.3 | 3/4 |

Directionally positive everywhere and significant nowhere. The standard deviation of a single
waiver decision is about 20 points over three weeks, so resolving a one-point edge at 2 se needs
roughly **1,800 paired decisions — about 44 seasons**. Four exist.

**So this is inconclusive, and reporting it as "the signal does not work" would be wrong.** The
screen is well powered because it uses ~1,400 players per anchor; the decision is not, because
it uses one claim per week.

## Consequences

* The signal may be shown next to a waiver decision, with its size.
* Nothing sorts on it. A future review proposing a waiver module ranked by snap delta should
  read the middle section: that was measured and it is worse than consensus.
* **Amended 2026-08-27.** The combined rule -- ECR chooses the candidate set, snap delta
  chooses within it -- is now adopted as a **provisional rule** under
  [ADR-0014](0014-a-provisional-rule-may-act-where-no-gate-can-run.md), logged per claim. That
  does not contradict this ADR: nothing sorts on snap delta, which is what "never ranked on"
  meant and still means. What changed is that declining to use a screened signal *forever*,
  because its horizon never arrives, was itself a decision -- and it was being made by default
  rather than deliberately.
* The combined rule is not refuted, and is not adoptable either. If it is ever wanted, the
  honest path is a decision rule declared in advance and left to accumulate across real seasons,
  not another retrospective slice of the same four.

## This is the same disposition as ADR-0010, for a different reason

[ADR-0010](0010-edge-is-displayed-but-never-ranked-on.md) holds that `edge` is displayed and
never ranked on because it **cannot be validated** — ESPN does not retain the historical ADP it
would need. Here the signal *is* validated, and the *decision built on it* cannot be. Different
failure, same disposition, and the pair is worth keeping side by side: a real measurement is not
a ranking, and the step between them has its own evidential burden.
