# Contract fixtures

Frozen payloads for `tests/contracts/`. Contract tests run against these rather than the
live API, because live-API tests in CI are flaky and prove nothing about our parsing. The
nightly job in `tests/golden/` is what actually catches an upstream rename.

## Provenance

Two kinds of fixture, and the filename says which. There is a third answer to the provenance
question and it has no file here: a contract that no test validates against any frozen
payload. Five of the eleven in `hub.contracts` are in that position, so what this directory
records about them is nothing, and `verified_against_live=None` is how they say so — the
absence of a row below is not evidence of a capture. It was six until 2026-09-05: a row here
is only evidence a validation can reach, and `espn_scoreboard.json` was recorded below as a
capture while the one test reading it asserted fields by hand, so the resolver saw nothing
and the contract said `None` beside its own evidence. See
`tests/contracts/test_every_contract_is_applied.py`, which resolves which payload each
contract is actually checked against and requires the flag to match.

| Fixture | Source |
|---|---|
| `nflverse_pbp.json` | **Captured** 2026-08-23, real 2025 play-by-play, first 8 rows of `PBP_COLS` |
| `nflverse_ff_opportunity.json` | **Captured** 2026-08-23, real 2025 weekly ff_opportunity |
| `nflverse_schedules.json` | **Captured** 2026-08-23, real 2025 games with a published spread |
| `espn_scoreboard.json` | **Captured** 2026-08-23 from the public scoreboard endpoint, trimmed — see below |
| `cfbd_games.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `cfbd_lines.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `odds_spreads.synthetic.json` | **Hand-built** from The Odds API's documented response shape |

The `.synthetic` suffix is not decoration. Those three were written by hand because no
`CFBD_API_KEY` or `ODDS_API_KEY` exists on this machine, so **they prove our parser handles
the shape we believe the API returns, not the shape it actually returns.** A synthetic
fixture cannot catch a rename, which is the failure contracts exist for. Replace them with
real captures the day a key is added; until then treat those two sources as unverified
against reality and read the `tests/golden/` job as the thing that would close the gap.

`espn_scoreboard.json` is a capture with fields removed: two events, each carrying only
`id`, `date`, `name`, event-level `status`, and one competition's competitors. The values and
their nesting are ESPN's, so it is evidence about the shape of a real response — but only
about the part that was kept. It does not carry `competitions[0].status`, which is where
`hub.fetch.espn.live_state` reads a game's state from, so feeding this file to `live_state`
raises `KeyError: 'status'` (measured 2026-09-05). **That is not evidence that the endpoint
omits the field**, only that this capture cannot answer the question; a trimmed capture is
silent about what was trimmed. `tests/contracts/test_source_contracts.py` therefore lifts the
four contracted columns out of the payload itself, and pins the gap in a test that goes red
the day someone re-captures with that field present.

CFBD prohibits redistributing its data. A hand-built two-row shape sample is not a dataset,
but a real capture would be -- if you replace these, keep them minimal.
