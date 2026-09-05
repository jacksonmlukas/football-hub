# Contract fixtures

Frozen payloads for `tests/contracts/`. Contract tests run against these rather than the
live API, because live-API tests in CI are flaky and prove nothing about our parsing. The
nightly job in `tests/golden/` is what actually catches an upstream rename.

## Provenance

Two kinds of fixture, and the filename says which. There is a third answer to the provenance
question and it has no file here: a contract that no test validates against any frozen
payload. Six of the eleven in `hub.contracts` are in that position, so what this directory
records about them is nothing, and `verified_against_live=None` is how they say so — the
absence of a row below is not evidence of a capture. See
`tests/contracts/test_every_contract_is_applied.py`, which resolves which payload each
contract is actually checked against and requires the flag to match.

| Fixture | Source |
|---|---|
| `nflverse_pbp.json` | **Captured** 2026-08-23, real 2025 play-by-play, first 8 rows of `PBP_COLS` |
| `nflverse_ff_opportunity.json` | **Captured** 2026-08-23, real 2025 weekly ff_opportunity |
| `nflverse_schedules.json` | **Captured** 2026-08-23, real 2025 games with a published spread |
| `espn_scoreboard.json` | **Captured** 2026-08-23 from the public scoreboard endpoint |
| `cfbd_games.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `cfbd_lines.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `odds_spreads.synthetic.json` | **Hand-built** from The Odds API's documented response shape |

The `.synthetic` suffix is not decoration. Those three were written by hand because no
`CFBD_API_KEY` or `ODDS_API_KEY` exists on this machine, so **they prove our parser handles
the shape we believe the API returns, not the shape it actually returns.** A synthetic
fixture cannot catch a rename, which is the failure contracts exist for. Replace them with
real captures the day a key is added; until then treat those two sources as unverified
against reality and read the `tests/golden/` job as the thing that would close the gap.

CFBD prohibits redistributing its data. A hand-built two-row shape sample is not a dataset,
but a real capture would be -- if you replace these, keep them minimal.
