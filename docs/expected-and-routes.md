# Two things the field said we were missing. Both are nulls here.

**Run 2026-08-29**, pre-registered in [what-the-field-knows.md](what-the-field-knows.md) and
committed before either was executed. Both came back null, and one of them came back null in a
way I had not enumerated.

---

## 1. Expected points in the weekly priors — null, slightly negative

**The change, as pre-registered.** Opportunity counts stay realised — *a target is not an
estimate* — and everything below the opportunity is replaced with `ff_opportunity`'s expected
variant: `receptions → receptions_exp`, `receiving_yards → rec_yards_gained_exp`, and the
rushing and passing equivalents. The substitution moved **74%** of receiving priors, 47% of
rushing, 12% of passing; `targets_prior` is unchanged, by design.

**The result.**

| contrast | realised priors | expected priors |
|---|---|---|
| the week — weekly vs `f = 1` | +0.0103 (2.8 se) | **+0.0128 (3.6 se)** |
| the rebuild — `f = 1` vs flat | **+0.0634 (5.3 se)** | +0.0581 (3.2 se) |
| both together | +0.0737 (5.9 se) | +0.0708 (3.8 se) |

| gate | realised | expected |
|---|---|---|
| frozen | **+0.711** [+0.313, +1.129] | **+0.082** [−0.372, +0.534] |
| churn | −2.006 [−2.761, −1.229] | −3.068 [−3.927, −2.258] |

**Against the pre-registration**, which said *"the rebuild arm improves materially — at least
+0.03 MAE — and the gate verdicts do not change"*:

- **The accuracy half is wrong.** The rebuild got *worse*, by 0.005.
- **The verdict half held.** SHOW on both gates either way.

**Why it plausibly fails here, stated as a hypothesis and not a finding.** The literature's
claim is about the stability of expected points against realised, and stability buys the most
where the sample is thinnest. But the prior here is already a **multi-week average**, which
does much of the same smoothing — by six games the realised mean has removed most of the
efficiency noise xFP removes in one.

Weakly consistent with that, and exploratory rather than pre-registered:

| games played before the week | n | realised MAE | expected MAE | gain |
|---|---|---|---|---|
| 3 | 2,218 | 4.451 | 4.413 | **+0.038** |
| 4–5 | 3,939 | 4.470 | 4.472 | −0.001 |
| 6–8 | 4,795 | 4.633 | 4.629 | +0.005 |
| 9+ | 3,716 | 5.069 | 5.113 | **−0.044** |

It helps at three games and hurts at nine-plus, which is the right shape for the mechanism and
far too small to rescue the headline. **Not adopted.** The join stays in the tree behind
`--expected`, per ADR-0007.

---

## 2. Pass-play participation — null, and a twin rather than an upgrade

**The pre-registration said:** *"`route_trend` clears, and `snap_trend` dies once controlled
for it — access dominating presence. If both survive independently, they are measuring
different things and the framing above is wrong."*

**Neither happened.** What happened is a third thing:

    corr(snap_trend, route_trend) = +0.917
    corr(offense_pct, route_pct)  = +0.963

Put in the same joint screen the two **annihilate each other** — snap_trend falls to +0.017 at
1.4 se, route_trend to −0.001 — and the screen loses a signal it had. That is a fact about
collinearity, not about either signal.

Screened against the *other* survivors rather than against each other, both clear and the
**snap version is the stronger**:

| | alone | vs the other survivors |
|---|---|---|
| snap_trend | +0.038 (2.8 se), 5/5 | **+0.0425 (3.04 se), 5/5** |
| route_trend | +0.032 (2.4 se), 5/5 | +0.0341 (2.51 se), 5/5 |

**So the access-beats-presence distinction does not survive here.** A hypothesis for why,
untested: the distinction the literature draws is about *levels* — a back with 80% of snaps and
30% of routes — and a **trend** differences the level out. If a player's run/pass snap mix is
stable within a season, the two trends have to move together, and at 0.96 on the underlying
shares they do.

**Not adopted, and deliberately not left in the default screen.** `ROUTE_TREND` is defined and
excluded from `FEATURES` with a test pinning that: a collinear twin in the control set destroys
a real signal, and leaving it in would have quietly cost the repo `snap_trend`. The builder
stays with its harness behind `--routes`.

### What it cost and what it bought

The build is `participation` at play level — 46k–51k plays a season, `offense_players`
exploded — and it is the slowest thing that has ever been in the panel, which is why it is
opt-in. What it bought is the knowledge that this repo's cheapest in-season signal is already
the better of the two measurements, and a `route_pct` column that is now trivially available
should a *level* question ever need it.

---

## What both have in common

Two well-supported outside findings, each verified as genuinely absent from the model, each
implemented faithfully, each null on this data. That is not an argument against reading the
literature — it is what reading it is for. The alternative was carrying both as folklore.

It also sharpens the thing [what-the-field-knows.md](what-the-field-knows.md) opened with:
**the binding constraint here is not projection accuracy.** Two features that the field says
improve projections did not improve this projection, and the one that slightly improved the
*week* contrast (+0.0103 → +0.0128) still made the lineup decision worse.

## Reproduce

```bash
uv run python -m hub.models.weekly --fit --expected
uv run python -m hub.models.weekly_screen --run --routes
uv run python -m hub.season.weekly_gate --run --seasons 2021,2022,2023,2024,2025 \
  --drafts 40 --shrink mae-market --expected
```
