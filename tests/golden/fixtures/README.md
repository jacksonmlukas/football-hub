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

**A row here is also not evidence about the part of a payload that was trimmed away.** A
capture cut down past the path the production reader takes still reads as a capture from
that table, and cannot fail when the source changes shape on the path the code walks — which
is what `espn_scoreboard.json` was until #73. So the rule for trimming is: remove anything
you like except a path a reader takes.

| Fixture | Source |
|---|---|
| `nflverse_pbp.json` | **Captured** 2026-08-23, real 2025 play-by-play, first 8 rows of `PBP_COLS` |
| `nflverse_ff_opportunity.json` | **Captured** 2026-08-23, real 2025 weekly ff_opportunity |
| `nflverse_schedules.json` | **Captured** 2026-08-23, real 2025 games with a published spread |
| `espn_scoreboard.json` | **Captured** 2026-09-05 from the public scoreboard endpoint, trimmed — see below |
| `cfbd_games.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `cfbd_lines.synthetic.json` | **Hand-built** from CFBD's documented response shape |
| `odds_spreads.synthetic.json` | **Hand-built** from The Odds API's documented response shape |

The `.synthetic` suffix is not decoration. Those three were written by hand because no
`CFBD_API_KEY` or `ODDS_API_KEY` exists on this machine, so **they prove our parser handles
the shape we believe the API returns, not the shape it actually returns.** A synthetic
fixture cannot catch a rename, which is the failure contracts exist for. Replace them with
real captures the day a key is added; until then treat those two sources as unverified
against reality and read the `tests/golden/` job as the thing that would close the gap.

## `espn_scoreboard.json`

Re-captured on 2026-09-05 by one unauthenticated GET of

    https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=200

— the URL `hub.fetch.espn.scoreboard("cfb")` builds, with the User-Agent `_get` sends first.
The scoreboard endpoint needs no key, which is why the live overlay may refresh every ten
minutes with no secret and why this is the one source here that can be re-captured on demand.
One request, not a loop.

**Why the college board and not the NFL one.** The NFL board on 2026-09-05 held sixteen
Week 1 events and every one of them was `pre`, so an NFL capture that day could not carry an
in-progress game — and the in-progress path is the whole reason the overlay exists. The
college board that afternoon held games in all three states at once. The response shape is
the same endpoint either way, and `live_state` reads both leagues through the same code.

**What was kept, and what was trimmed.** Four of the 99 events, chosen to hold every state
and both situation cases:

| Event | State | Why it is in the file |
|---|---|---|
| ECU at Alabama | `in` | `situation` carries `possession` and `downDistanceText` |
| North Texas at Indiana | `in` | live, but `situation` carried neither — ESPN sent it between plays |
| Tennessee State at Georgia | `pre` | scores are `"0"`, not absent |
| San José State at USC | `post` | a final |

Each event keeps `id`, `date`, `name` and one competition holding `status.type`
(`state`, `shortDetail`), its competitors (`homeAway`, `score`, `team.abbreviation`) and a
`situation` reduced to the two fields the reader reads. Everything else ESPN sent — venue,
odds, broadcasts, links, and a `situation.lastPlay` block naming players and linking their
headshots — is gone. It is third-party data a fixture does not need, and dropping it is the
same instinct as the CFBD note below.

The **event-level `status`** is deliberately not kept, though the real response carries one.
`live_state` takes a game's state from the competition, and a second copy of it on the event
is exactly what let a test read a path production does not: the previous capture kept only
the event-level copy, so putting it through `live_state` raised `KeyError: 'status'` and the
contract test had to lift columns out of the payload beside the reader instead of through it.
The trim now follows the reader.

That is what makes this file able to fail. `tests/contracts/test_source_contracts.py` runs it
through `espn.live_state`, and three shape changes tried against it on 2026-09-05 — renaming
`team.abbreviation`, renaming `homeAway`, and dropping `competitions[].status` — each stop
every event resolving and raise; two events sharing an id fails the contract. An `id` arriving
as a number is *not* caught, because `SCOREBOARD_TYPES` coerces it back to `Utf8` first, in
production as well as here.

CFBD prohibits redistributing its data. A hand-built two-row shape sample is not a dataset,
but a real capture would be -- if you replace these, keep them minimal.
