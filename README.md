# football-hub

Fantasy football and NFL forecasting, built as measurement infrastructure first.

The interesting problem is not predicting football. It is knowing whether your predictions
are any good, in a domain with ~285 NFL games a season, an efficient market as the benchmark,
and no ability to run an A/B test. This repo is organised around that problem — and the most
useful thing in it is the record of what got measured and then *removed*.

## What this repo has actually concluded

Every one of these was built, measured against a rule fixed before the numbers, and acted on.

| Thing | Verdict | Evidence |
|---|---|---|
| Championship equity — nested draft + season simulation optimising P(win) | **Removed from the draft output** | −19.66 pts/team-game vs following the market, 95% CI [−23.16, −16.20], n=80 ([ADR-0009](docs/adr/0009-championship-equity-does-not-pick.md)) |
| VOR ordering | **Demoted to context** | −5.06 pts/team-game ([market-value.md](docs/market-value.md)) |
| `edge` (consensus vs ADP), the repo's original signal | **Displayed, never sorted on** | Structurally unmeasurable — needs historical ADP that ESPN does not retain ([ADR-0010](docs/adr/0010-edge-is-displayed-but-never-ranked-on.md)) |
| Lineup optimiser | **Inert until variance is real** | +0.00 pts/game; `sd = k·√mu`, so it is handed no information sorting lacks ([ADR-0012](docs/adr/0012-the-lineup-optimiser-waits-for-real-variance.md)) |
| Five candidate draft signals (expected-vs-actual points, recency-weighted, depth-chart climb ×2, age) | **All null** | Screened before any implementation ([signal-screens.md](docs/signal-screens.md)) |

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
  **screened**: partial correlation against expert consensus rank. Five screened, five null.
- A **model** produces a projection or a decision. It is **gated**: does it beat the simplest
  thing that already works? Championship equity was gated against following the market and
  lost.

Confusing the two is how a well-built, well-tested model ships while being confidently worse
than a one-line rule. `CONTEXT.md` defines both.

## Evaluation methodology

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
- [`docs/architecture.md`](docs/architecture.md) — index of twelve decision records.
- [`docs/decisions.md`](docs/decisions.md) — the working journal, including corrections issued
  against this project's own conclusions.
- [`docs/next.md`](docs/next.md) — what is open, what is closed, and why.

## Data sources and licensing

nflverse, FantasyPros via `nflreadpy`, ESPN's public fantasy endpoints, CFBD, The Odds API.
None of it is redistributed: `data/` is gitignored in full and the pre-flip check verifies no
data file is reachable anywhere in history, not just at HEAD.

## License

MIT for the code. The data is not mine to license.
