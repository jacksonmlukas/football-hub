# What the field knows, and the two things it says we are leaving on the table

**Written 2026-08-29**, after fifteen measurements and an outside literature scan. Successor to
[where-to-look-next.md](where-to-look-next.md), which was written at fourteen and before any of
the weekly work. Same rules: nothing here is committed to, and every proposal carries the gate
it would have to clear.

## The frame

Fifteen measurements; consensus has won roughly fourteen. The two that worked — the injury
designation and the snap-share trend — both asked consensus about something published on a
**Monday**. Every null asked it about information consensus had had all summer.

The outside literature says the same thing from the other side. Wharton's 2026 study
benchmarked pure machine-learning projections against ESPN's for the 2025 season and found
**industry projections still win on absolute error while algorithmic models are competitive on
ranking**. That is the exact shape of [weekly-blend-gate.md](weekly-blend-gate.md): this repo
beat its own flat projection by +0.074 MAE at 5.9 se and then lost the lineup decision to a
free ranking.

**The binding constraint here has never been projection accuracy.** It is worth saying plainly
before proposing anything that improves accuracy.

---

## 1. There is a weekly expected-points table in the store and the model never opens it

**Verified 2026-08-29:** the only mention of `ff_opportunity` in `hub/models/weekly.py` or
`hub/models/weekly_screen.py` is a *comment*. The panel is built entirely from `player_stats` —
**realised** counts. Meanwhile the store holds weekly `receptions_exp`,
`rec_yards_gained_exp`, `rush_yards_gained_exp`, `pass_yards_gained_exp`,
`rec_touchdown_exp` and eleven more.

The literature is unusually unanimous: expected fantasy points are *"by far the more stable and
predictive metric"*, and Fantasy Points Over Expected — the residual — regresses week to week.

**This repo already proved half of it independently.** `td_rate_prior` came back at −0.040
across 5/5 seasons in [weekly-screen.md](weekly-screen.md): a player whose points came from
touchdowns rather than yards scores *less* next week. That is FPOE regressing, found by a
different route, and then the model was built on the regressing quantity anyway.

What the current model structurally cannot see: **a six-target week at three yards downfield
and a six-target week at fifteen yards downfield are identical to it.** Expected points encode
air yards, target depth and field position — the *quality* of an opportunity, where the model
only counts it.

### Pre-registered, before the run

**The change.** In the priors, the expected variant replaces the realised one:

| stays realised | becomes expected |
|---|---|
| `targets`, `carries`, `attempts` | `receptions` → `receptions_exp` |
| | `receiving_yards` → `rec_yards_gained_exp` |
| | `rushing_yards` → `rush_yards_gained_exp` |
| | `passing_yards` → `pass_yards_gained_exp` |

Opportunity counts stay realised because **a target is not an estimate** — he was thrown at or
he was not. Everything downstream of the opportunity is an efficiency, and efficiency is what
regresses.

**How it is judged.** It is a feature change to an already-gated model, so it re-runs what
exists and changes nothing else: the Gate A diagnostic, then both gates, at the settings ADR-0017
fixed (2021–2025 scoring 2022–2025, 40 rosters, `mae-market`).

**Pre-stated expectation.** The rebuild arm improves materially — **at least +0.03 MAE** on the
`f = 1` vs flat contrast — and **the gate verdicts do not change**. If a verdict flips to ADOPT
on a feature substitution, treat the run as suspect: nothing in fifteen measurements suggests
one join is worth that.

---

## 2. Route participation is the sharper version of the one signal that lasted

The framing worth stealing verbatim: **snap share tells you presence, route participation tells
you access, and target share tells you whether the quarterback used that access.** PFF's
expected-target work reports that a *predicted*-target share is more stable than raw target
share by **+0.18 R²**.

This repo's one durable in-season signal is a **snap-share** trend — +0.043 joint, 5/5 seasons.
Snap share counts run snaps a receiver was never going to be thrown to. Route participation
does not, and it is the same quantity measured better.

**Verified available, and better than feared.** `nflreadpy.load_participation` carries
`offense_players` (the gsis ids on the field) and `route`:

| season | plays | with `offense_players` | non-empty `route` |
|---|---|---|---|
| 2021 | 50,714 | 91.3% | 37.3% |
| 2022 | 50,150 | 91.4% | 36.2% |
| 2023 | 46,168 | 100.0% | 42.5% |
| 2024 | 45,919 | 100.0% | 41.6% |
| 2025 | 45,184 | 100.0% | 41.8% |

**A caveat that shapes the definition.** `route` is one value per *play*, not per player — it is
the charted route on that play, so a per-player route type is not derivable. What is derivable
is **pass-play participation**: the share of his team's route-charted plays on which he was on
the field. That is the "access" quantity, it is available for all five seasons, and calling it
"route participation" would be claiming a precision the data does not have.

### Pre-registered, before the run

**The feature.** `route_pct` = plays where the player is in `offense_players` **and** the play
has a non-empty `route`, over his team's plays with a non-empty route, per game. Then
`route_trend` = mean(w−3…w−1) − mean(w−6…w−4), exactly the shape `snap_trend` already uses, on
the same calendar-week grid, dark before week 8 for the same reason.

**How it is judged.** Through the existing screen, against points and against Usage, with the
same controls and the same two-half bar. **And head to head with `snap_trend` in the joint
screen** — which is the question that matters, because these measure the same thing and only
one of them should survive.

**Pre-stated expectation.** `route_trend` clears, and **`snap_trend` dies once controlled for
it** — access dominating presence. If both survive independently, they are measuring different
things and the framing above is wrong.

---

## Three more, not being run now

**The winner's curse has a standard fix and it is not shrinkage.** Measured 2026-08-29: median
within-player weekly sd is 5.06, and predictive sd is **+36% at one game played** against
twelve, +7% at four. [ADR-0012](adr/0012-the-lineup-optimiser-waits-for-real-variance.md)
measured per-player *outcome* volatility beyond the positional constant at ±9.3% and called it
not estimable — **parameter uncertainty is a different quantity, four times larger at the thin
end, and trivially estimable, because it is just `n`.** The waiver fix is to rank adds by a
lower confidence bound, `mean − z·se`, rather than by the mean: the textbook treatment for
taking the maximum of hundreds of noisy estimates, and it does not touch the mean, which was
tried twice and failed twice.

**ADR-0012 is narrower than it reads.** Its re-run clause was withdrawn because volatility is
not estimable. Parameter uncertainty is not a function of `mu` — it is a function of `n` —
which is the exact condition the ADR named as what would revive the lineup optimiser. Whether
+4% at the thick end is enough to move a lineup is untested and cheap to test.

**The objective is not points.** Everything here is scored in points per team-week, but a league
is won on *wins*, which is a nonlinear function of points. A +0.7 edge is worth more when it
flips a close matchup than when it pads a blowout, and the literature's exploitable asymmetry —
*for the underdog variance is a friend, for the favourite it is the enemy* — is already
implemented in `lineup.optimize` and inert only for want of real variance.

## What the field says to stop doing

- **Better preseason projections.** Wharton: industry beats machine learning on error. This
  repo: five preseason screens, five nulls.
- **More market-derived features.** The implied total died in the joint screen. Consensus reads
  the same lines on the same Saturday.
- **Positional-mean shrinkage.** Measured twice here — null, then three times worse
  ([weekly-shrinkage.md](weekly-shrinkage.md)) — and the literature agrees on the mechanism: it
  over-shrinks the studs.

## Sources

Wharton Sports Analytics, *Beyond the Expert* (2026) · SumerSports, *Sticky Stats* · PFF,
*"Coach, I was open"* · Sharp Football, *Expected Fantasy Points* · ffverse `ffopportunity` ·
Fantasy Football Analytics on projection uncertainty and lineup optimisation under uncertainty.
