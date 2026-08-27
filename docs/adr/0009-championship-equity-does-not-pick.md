# Championship equity does not pick

**Status:** accepted 2026-08-24.

**Decision.** The draft market picks, with consensus behind it when ESPN publishes no ADP.
Championship equity — the nested draft-and-season simulation in `hub.draft.optimize` — is not
in the draft-night output at all. The code stays, because the harness calls it and because
retesting it later should not start from nothing.

## Why, in one number

P0b, 2026-08-24. Two arms on the same room and the same seed, over 2022–25, scored on realised
weekly points:

    n=80   optimizer - market = -19.66 points per team game
           95% CI [-23.16, -16.20]      P(optimizer better) 0.0%

**Re-run 2026-08-25**, unchanged code path, after a day of work elsewhere in the repo:

    n=80   optimizer - market = -19.13 points per team game
           95% CI [-22.31, -15.75]   P(optimizer better) 0.0%

The original -19.66 sits well inside that interval, so the verdict reproduces. The two are not
identical because the historical boards are rebuilt from live ffopportunity and consensus each
run, and both move as nflverse backfills; the seed fixes the simulation, not the inputs. Worth
knowing that this gate is a re-measurement and not a recorded constant.

**Spot-checked again 2026-08-27 after the bracket fix**, and *not* re-measured. Giving the
playoffs their own draws (see `season.seed_table` / `season.champion`) changes
`champion_probability`, which is arm B's objective and nothing arm A touches -- so unlike the
re-run above, this one could genuinely have moved the number.

A full re-run was started and abandoned: a single draft across four seasons exceeds ten
minutes, so the 20-draft harness is a multi-hour job, not the ~30 minutes assumed when it was
launched. `--diagnose` exists for exactly this question -- "run before and after a change to
the objective and diff" -- and answers it for a thousandth of the cost:

| pick | before | after |
|---|---|---|
| 3 | Christian McCaffrey RB +5.39% | Christian McCaffrey RB +4.98% |
| 22 | Chris Olave WR +1.84% | Chris Olave WR +1.88% |
| 27 | Travis Etienne Jr. RB +1.44% | Travis Etienne Jr. RB +1.02% |
| 46 | Javonte Williams RB +3.12% | Javonte Williams RB +2.94% |
| 51 | Parker Washington WR +1.20% | **Michael Wilson WR** +1.48% |
| 70 | Travis Etienne Jr. RB +2.31% | Travis Etienne Jr. RB +3.25% |

Five of six turns name the same player, the sixth swaps one mid-round receiver for another,
and the lifts move by less than half a point against a 250-sim standard error. The tripwire
trips identically in both -- equity names a filled RB over an empty QB/WR/TE at 27, 46 and 70.

**So the decision stands, and the honest statement of its evidence is unchanged: -19.66 and
-19.13 are both measurements of the old bracket.** No point estimate exists for the new one.
That is a gap worth naming rather than papering over, and closing it means budgeting hours for
the harness, not minutes. What the spot-check establishes is narrower and sufficient: the fix
does not move what equity recommends, so it cannot plausibly move a 19-point verdict.

Losing in all four seasons, winning 9 of 80 drafts. `config_digest` 9975101f; paired rows in
`data/processed/p0b_paired.parquet`; harness in `hub.draft.backtest`.

The rule that produced this action was fixed before the run, and had three branches, including
one that would have promoted equity back to the headline.

## Why this is surprising, which is why it is written down

The repo contains a real season simulator: talent drawn once per season, a square-root weekly
spread law fitted at an exponent of 0.498, per-position skew, teammate correlation applied by
Cholesky, a 14-week schedule, a six-team playoff. Every piece of it is measured. It is
reasonable to assume that something that elaborate must pick better than "take the best
available player who fills a hole".

It does not, and the reason is visible in the rosters it builds. In one 2024 draft it took
McCaffrey, Kamara, Mixon, Ekeler and Conner — five running backs — and finished with four
receivers in a three-receiver league. The simulator is not broken. Its *lift ordering* is not
accurate enough to beat following consensus and filling your starting slots.

## What was considered and rejected

- **Blame the inputs.** Arm B scores seasons on prior-season xFP while the live board uses
  `proj_blend`, so the tested optimizer is fed worse numbers than the shipped one. Real, and
  listed in `backtest.LIMITATIONS` before the run. But a twenty-point gap is not an
  input-quality gap: better `mu` sharpens *which* running back, not *how many*.
- **Keep it as a tiebreaker.** That was its role, and it is the worst place for it. A
  tiebreaker only acts when the objective is the sole thing deciding, so a bad one is wrong
  precisely when it matters most.
- **Wait for P1.** P1 (break the circularity) was gated on equity beating the market. It
  loses, so P1 does not fire.

## Consequences

- The no-ADP path improves rather than degrading. It used to point at the equity table; it now
  falls back to `market_pick(by="ecr")` — which is arm A, the arm that won.
- `--win-prob`, `--no-win-prob` and `--sims` are gone from the board CLI.
- `tripwire` in `hub.draft.backtest` is the standing check on whether an objective is fit to
  pick with. It fired on this preference the same morning and was talked past once; see its
  docstring, which records that.
- Reopening this means re-running `hub.draft.backtest`, not re-arguing from first principles
  about how good a season simulator ought to be.
