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

The spot-check establishes something narrower than a re-measurement and sufficient for the
decision: the fix does not move what equity recommends, so it cannot plausibly move a 19-point
verdict.

### Measured on the new bracket, 2026-08-27: -17.30

The full harness was then run to completion on the fixed bracket -- 3h20m, same 20 drafts x 4
seasons, same board snapshot, n=80:

    n=80   optimizer - market = -17.30 points per team game
           95% CI [-20.45, -14.07]   P(optimizer better) 0.0%
           REMOVE

| measurement | bracket | result |
|---|---|---|
| original | recycled weeks | **-19.66**, CI [-23.16, -16.20] |
| re-run 2026-08-27 | recycled weeks | **-19.13**, CI [-22.31, -15.75] |
| **this run** | **own weeks** | **-17.30**, CI [-20.45, -14.07] |

**The verdict is unchanged and the gap named above is now closed.** The point estimate moved
1.83 points between the two brackets, which is well inside every interval here -- each of the
three sits comfortably in the other two's CI -- so this is not evidence that the bracket
mattered much. It moved in the direction the fix predicts, though: giving the playoffs their
own draws decouples a title from a week 1 result, and that helps the arm whose objective is
championship probability. Helping arm B by two points against a seventeen-point deficit is the
shape of a real but small correction.

Nothing about ADR-0009 changes. What changes is that its number is now a measurement of the
code that exists.

Losing in all four seasons, winning 9 of 80 drafts. `config_digest` 9975101f; paired rows in
`data/processed/p0b_paired.parquet`; harness in `hub.draft.backtest`.

The rule that produced this action was fixed before the run, and had three branches, including
one that would have promoted equity back to the headline.

> **Restated 2026-09-11 (#52): the intervals above are over the wrong unit, no magnitude
> is quotable, and the verdict is untouched.** Every interval above is bootstrapped over the
> eighty (season, draft) rows. With the season as the unit of replication — four, not eighty
> — the published run's interval is **[−26.68, −11.45]** and its MDE 14.77 rather than 5.03
> (#45; `experiment.summarise` now clusters on the season and prints the MDE;
> [gate-power.md](../gate-power.md)). Re-run on the repaired harness at a fixed data digest
> the effect returned **−12.48** and then **−11.59**, having moved eight points across 270
> commits with the input bytes unchanged (#190, open), and every run to date was made on a
> harness with a foresight leak that favoured arm B — the arm that lost — so the losses are if
> anything understated (#195, fixed; #194 re-measures). So none of −19.66, −19.13, −17.30 or
> the later figures is quoted as the size of the effect, and where the record cites −19.66 it
> reads *as first measured*. What every run agrees on, and what REMOVE rests on: worse in 4 of
> 4 held-out seasons, an interval excluding zero, `P(optimizer better)` 0.0%. The figures
> above keep their original text.

> **Restated 2026-09-12 (#194, #245, #246): the run-to-run spread is 0.3, the current tree
> reads −11.07, and the verdict stands on the seasons.** #194 ran `e14ab56` at two root
> seeds on one Board digest: **−12.19** and **−11.88**, per-season differences 1.72 / 0.62 /
> 0.62 / 0.76 — so the eight-point historical drift is code, not noise. Then the shipped
> tree (`24bbe45`: `TALENT_CV` net of absence #235, the nickname crosswalk #246 that had
> been scoring four 2022 draftees as zero, byes #226, the imputed-player spread #87, a
> different Board digest for all four reasons) on the same seasons, seed and budget:
>
> | | `e14ab56` seed 0 | `e14ab56` seed 7 | **shipped, 2026-09-12** |
> |---|---|---|---|
> | optimizer − market | −12.19 | −11.88 | **−11.07** |
> | 95% percentile CI | [−18.33, −6.05] | [−17.09, −6.67] | **[−16.92, −5.22]** |
> | 95% t CI, 3 df | [−22.06, −2.32] | [−20.34, −3.42] | **[−20.32, −1.82]** |
> | 2022 / 2023 / 2024 / 2025 | −19.95 / −5.38 / −6.72 / −16.71 | −18.23 / −6.00 / −7.34 / −15.95 | **−17.52 / −5.07 / −5.38 / −16.31** |
> | join failures (market / optimizer) | — | — | 0.0% / 0.5%, floor 2% |
> | verdict | REMOVE, 4/4 | REMOVE, 4/4 | **REMOVE, 4/4** |
>
> The current tree sits about a point nearer zero than `e14ab56` — two of the 0.5-point
> run-to-run standard errors — with 2022, the season whose draftees were being scored as
> zero, moving most. That is the direction #246 predicted and about its size; #235's refit
> is the other candidate and the two are not separated by one run. **What this ADR rests on
> has not moved in any run ever made**: worse in 4 of 4 seasons, an interval excluding zero,
> `P(optimizer better)` 0.0%. The −17.30 above was measured on the arm and the constants of
> 2026-08-27 and reads *as first measured*; the eight-point history (#190) is attributed to
> the named changes rather than re-derived, because the published run recorded no digest to
> re-derive from. Paired rows: `data/processed/gate/p245_shipped_seed0.parquet`.

> **Restated 2026-09-12 (#260): the season draw was narrowed to the rostered union, which is
> a total re-draw, and the −11.07 above was measured on the draw before it.**
> `season.simulate_weeks` drew correlated weekly points for every Board row (~450) on every
> season simulation and read back the ~168 rostered players at the lineup; it now draws over
> the rostered union in Board order. The model is unchanged -- same constants, same
> correlation blocks factored per team over their rostered members, same lineup rule -- but a
> player's talent and weekly draws are now a function of his position in the union rather
> than his Board row, and `docs/gate-power.md` says what any change to the drawn width is:
> a re-pairing of every player with his noise, not a perturbation. On the frozen 200-row
> Board at the #197 pin's budget, arm A's roster did not move and arm B's did at three of six
> picks (McCaffrey over Lamb at 1, Jacobs over Prescott at 4, Marquise Brown over Ekeler at
> 6); the pin is re-pinned with that cause. **The shipped-constants figure in the table above
> must be re-run at the new draw**, and read against #194's run-to-run spread of 0.31: a
> movement inside a few of the 0.5-point run-to-run standard errors is the re-pairing and
> nothing else, and a larger one needs its own explanation. That run needs the four Boards
> under `data/` and hours of wall-clock, neither of which the change was made with, so it is
> the maintainer's next action and is not claimed here -- `docs/method.md` rule 13, an open
> incident until the number beside −11.07 exists. Nothing this ADR rests on is touched by a
> re-pairing: worse in 4 of 4, an interval excluding zero, `P(optimizer better)` 0.0%.
> Time per arm-B draft on the frozen Board, 12 x 250: 83.28 s to 77.01 s, on a 200-row
> instrument where the union is 84% of the Board; the production Boards are ~450 rows and
> the union the same 168, so the saving there is larger and is measured with the re-run.

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

> **Re-scored 2026-09-07 (issue #51): this ADR and [ADR-0007](0007-measurements-that-steer-the-product-are-committed-code.md)
> pull opposite ways, and neither is wrong.** *"The code stays"* above is the REMOVE this ADR
> ordered, applied in `e56873c`; ADR-0007 independently requires a measurement that steered the
> product to stay committed and re-runnable. Both hold. What neither says is **where** such code
> should live, and a call-site census run on 2026-09-06 shows what that gap costs: roughly a
> thousand lines — `win_probability`, `champion_probability`, `rank_tiers`, `tag_for`,
> `cost_of_waiting`, all of `hub/draft/leverage.py` — sit inside the draft package downstream of
> no product decision, indistinguishable at a glance from code the product reads.
>
> The consequence is not tidiness. Nothing that ships reads it, so nothing that ships could
> complain, and two defects accumulated in it for 270 commits without any product-level signal.
> `backtest.py:227` still names *"the shipped 12 × 250"*, and there is no shipped path for the
> referent to point at.
>
> **Neither ADR is reopened by this.** The verdict stands, the code stays, and the question is
> only where it lives so a reader can tell the exhibit from the product. **Owned by #198**,
> which names the conflict explicitly and fills the gap rather than re-litigating either side.
> `hub.draft.cohort`'s use of `simulate_remaining_draft` is *not* part of the removed arm — it
> serves the other two Gates.

> **Restated 2026-09-11 (#198): it lives in `hub.exhibits.championship_equity`.** The
> objective this ADR gated — `win_probability`, `_lift_frame`, `rank_tiers` and, from
> `hub.draft.season`, `champion_probability` — is there verbatim, and `hub.draft.backtest`
> imports it from there as the one production reader, which is what *"reopening this means
> re-running `hub.draft.backtest`"* above needs. `hub.draft.optimize` now opens on the room,
> the seeding tree and THE PICK, which is what the product reads; its *"pinned at the shipped
> 12 × 250"* sentence in `compare` is corrected to say the budget is the measurement's own.
> `tag_for` is deleted (no caller, no measurement), `hub/draft/leverage.py` is
> `hub.exhibits.leverage`, and `cost_of_waiting` was never part of this arm — it is
> `pick_value`'s column and the product reads it. The numbers above are unchanged and were not
> re-run: a move of code is not a re-measurement. What *has* moved the arm since, and says so,
> is #235's `TALENT_CV` refit; `tests/unit/test_backtest.py`'s frozen-Board pin (#197)
> records the roster before and after.
