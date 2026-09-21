# football-hub

Fantasy football and NFL forecasting, built as measurement infrastructure first.

The interesting problem is not predicting football. It is knowing whether your predictions
are any good, in a domain with ~285 NFL games a season, an efficient market as the benchmark,
and no ability to run an A/B test. This repo is organised around that problem — and the most
useful thing in it is the record of what got measured and then *removed*.

**The method is written up in one place: [docs/method.md](docs/method.md)** — the two tests,
the thirteen rules, and the specific mistake behind each one.

## What this repo has actually concluded

Every one of these was built, measured against a rule fixed before the numbers, and acted on.

| Thing | Verdict | Evidence |
|---|---|---|
| Championship equity — nested draft + season simulation optimising P(win) | **Removed from the draft output** | Lost in 4 of 4 held-out seasons, n=80; −19.66 pts/team-game vs following the market *as first measured* — season-clustered CI [−26.68, −11.45], and re-runs at fixed inputs returned −17.30, −12.48 and −11.59, so no magnitude is quoted ([ADR-0009](docs/adr/0009-championship-equity-does-not-pick.md), restated 2026-09-11; [gate-power.md](docs/gate-power.md)). *(Restated 2026-09-21, #358: the verdict stands; this design could detect effects down to 11.17 pts/team-game at 80% power, k=4 seasons; [detectable-effects.md](docs/detectable-effects.md) row 6.)* *(Restated 2026-09-21 under rule 13, from #376's run: **REMOVE as measured under the superseded rule; SHOW under the tie-aware rule** (ADR-0019, #335) at won 0, tied 1, lost 3 of 4 — per season 2022 −15.35, 2023 −8.41, 2024 −5.99, 2025 −19.86, and 2024 is inside its own noise at 20 rooms. The **action** — championship equity does not pick — stands on judgment and three losses of four; the **verdict word** no longer does. Not a rehabilitation: the claim against it is downgraded from *worse in every season* to *worse in most, and the instrument cannot resolve the rest* — the first published removal to weaken under the fixed gate, and the cost S1 always carried.)* |
| VOR ordering | **Demoted to context** | −5.06 pts/team-game ([market-value.md](docs/market-value.md)). *(Restated 2026-09-21, #358: the verdict stands; this design's detectable effect could not be computed — only a pooled figure across 3 seasons × 40 drafts is published, no per-season breakdown to derive s from; [detectable-effects.md](docs/detectable-effects.md) row 7.)* |
| `edge` (consensus vs ADP), the repo's original signal | **Displayed, never sorted on** | Structurally unmeasurable — needs historical ADP that ESPN does not retain ([ADR-0010](docs/adr/0010-edge-is-displayed-but-never-ranked-on.md)). *(Restated 2026-09-21, #358: confirmed — no detectable-effect bound exists for this row; [detectable-effects.md](docs/detectable-effects.md) row 8. Not estimated.)* |
| Lineup optimiser | **Inert until variance is real** | +0.00 pts/game; `sd = k·√mu`, so it is handed no information sorting lacks ([ADR-0012](docs/adr/0012-the-lineup-optimiser-waits-for-real-variance.md)). *(Restated 2026-09-21, #358: the verdict stands; this design's detectable-effect bound could not be computed — the gate builds its paired frame from the network at run time and persists nothing; [detectable-effects.md](docs/detectable-effects.md) row 10.)* |
| Five candidate draft signals (expected-vs-actual points, recency-weighted, depth-chart climb ×2, age) | **All null** | Screened before any implementation ([signal-screens.md](docs/signal-screens.md)). *(Restated 2026-09-21, #358: at 80% power this design could detect partial r of 0.092 (expected-vs-actual points, measured +0.21), 0.122 (depth-chart climb, next season, measured +0.008), 0.209 (depth-chart climb, rest-of-season) and 0.103 (age, measured +0.0076); the recency-weighted variant's bound could not be computed. [detectable-effects.md](docs/detectable-effects.md) rows 1, 3, 4, 5.)* |
| Weekly injury cost — retention by designation and practice report | **Adopted** — the one that cleared its gate | Beats bench-the-ruled-out by 0.170 MAE at 3.8 se, n=3687 ([weekly-injury.md](docs/weekly-injury.md)) |

What survives and decides the pick is the draft market, corrected for three effects with
fitted coefficients and a bounded adjustment ([ADR-0011](docs/adr/0011-the-pick-ranks-on-corrected-adp.md)).

**Demonstrable market edge is an explicit non-goal.** Six measured attempts failed. The system
audits itself against markets; it does not try to beat them.

## What it does

| Component | Method | Status |
|---|---|---|
| Draft board | FantasyPros consensus + ffopportunity expected points + ESPN ADP, joined on normalised names | shipping |
| The pick | Best available filling a starting slot, on ADP corrected for touchdown luck, durability and injury, clamped at 20% of ADP | shipping |
| Live poller | Recomputes replacement level from what is left; detects positional runs; bounded output for a 90-second clock | shipping |
| Backtest harness | Replays past drafts on contemporaneous boards, scores rosters on realised weekly points | shipping |
| Player prediction | Fitted weekly spread law (`sd = k·√mu`, exponent 0.498 ± 0.012), per-position skew, teammate correlation by Cholesky | shipping |
| Component projection | Fantasy points as an aggregate of sampled counts and yards, not a directly projected total | shipping |
| Survivor | Integer program over the season (pulp/CBC), maximising log-probability of surviving | shipping |
| Conformal intervals | Hand-rolled split conformal on a rolling window of strictly earlier weeks | shipping |
| Team ratings | Passthrough returning the market prior. Deliberately a placeholder — see `hub/models/ratings.py` | **not built** |
| Player props, awards, staking | Roadmap. Free-tier odds credits cannot support props | **not built** |

## How a thing earns its place

Two tests, deliberately not interchangeable:

- A **signal** claims to predict outcomes beyond what consensus already knows. It is
  **screened**: partial correlation against expert consensus rank. Six preseason hypotheses
  screened, five null and one — the in-season snap-share trend — positive; the weekly screen
  that followed cleared three more on the next-week horizon. The index is
  [signal-screens.md](docs/signal-screens.md).
- A **model** produces a projection or a decision. It is **gated**: does it beat the simplest
  thing that already works? Championship equity was gated against following the market and
  lost.

Confusing the two is how a well-built, well-tested model ships while being confidently worse
than a one-line rule. `CONTEXT.md` defines both.

## Evaluation methodology

The short version. The long version, with the incident behind each rule, is
[docs/method.md](docs/method.md).


- **Decision rules are pre-registered.** Every gate's branches are fixed, in code, before the
  numbers — including the branch where the elaborate thing loses. `verdict()` functions are
  unit-tested so the rule cannot be quietly reinterpreted afterwards.
- **Outcomes, not projections.** Backtests score realised weekly points. Scoring against the
  same projection a strategy ranks on makes any projection-follower look prescient.
- **Temporal integrity.** Historical boards are reconstructed from the last consensus scrape
  *before* that season opened. Week *t* is never predicted using week *t+1*.
- **Provenance.** `config_digest` hashes the resolved config *and* all 35 fitted constants, so
  refitting a coefficient moves the model version ([ADR-0006](docs/adr/0006-fitted-constants-live-with-their-provenance.md)).
  It previously claimed to and did not.
- **Gates fire against their author.** Two did, in one day, and both are recorded — including
  one that was wrongly argued past and one that could not fail because its treatment arm had
  information its control arm lacked.

## Reliability

- **Data contracts** (`src/hub/contracts.py`) assert schema, nullability, uniqueness and
  plausible ranges at every ingest boundary. Violations raise and the pipeline serves
  last-good state.
- **Graceful degradation** is a requirement. Every optional stage of the board build fails
  independently and reports which; the board still builds.
- **Quota guards.** CFBD's free tier is 1,000 calls a month, so the fetch layer refuses
  per-team and per-game loops by construction.

## Development

```bash
uv sync --all-extras
cp .env.example .env      # ESPN cookies for ADP; .env is gitignored
make draft                # build the draft board
uv run pytest -q
```

A `PostToolUse` hook runs `pyrefly` and the full suite on every edit, so a change that breaks
either is rejected at the point of writing. A second hook blocks reading data files into an
agent's context, which is the most expensive mistake available in this repo.

## Docs worth reading first

- [`CONTEXT.md`](CONTEXT.md) — the glossary. Three different things get called "the market"
  here, and the entry exists to stop that.
- [`docs/architecture.md`](docs/architecture.md) — the index of all twenty-five decision
  records, each with what it rests on and whether that still holds.
- [`docs/decisions.md`](docs/decisions.md) — the working journal, including corrections issued
  against this project's own conclusions.
- [`docs/next.md`](docs/next.md) — what is open, what is closed, and why.
- [`docs/improvements.md`](docs/improvements.md) — a researched backlog. Every item is
  grounded in a number computed in this repo, and names whether it is a signal to
  screen or a model to gate.

## Data sources and licensing

nflverse, FantasyPros via `nflreadpy`, ESPN's public fantasy endpoints, CFBD, The Odds API.
None of it is redistributed: `data/` is gitignored in full and the pre-flip check verifies no
data file is reachable anywhere in history, not just at HEAD.

## License

MIT for the code. The data is not mine to license.
