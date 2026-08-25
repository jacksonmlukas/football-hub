# Draft night runbook — Thu Sep 3, 9:00 PM, slot 3

Rehearsed 2026-08-24. **Re-rehearsed end to end 2026-08-25**, which corrected the timings
below and found one real failure: with no network the board did not degrade, it raised
`ConnectionError` and printed a traceback instead of a pick. Fixed — see *no network at all*
under [When something breaks](#when-something-breaks).

## Before you sit down

```bash
make draft                 # rebuild the board; ~4s warm, ~40s on a cold nflverse cache
uv run python -m hub.draft.board --reset
```

The reset matters. Draft state persists on disk, and starting a real draft on top of leftover
picks from a rehearsal is a silent way to be wrong all night.

Once that has run once, **the venue's wifi can fail and the tool still works.** Every later
command falls back to the board written by this build. Which is the real reason to run it.

**Rebuild the board on the day.** ADP moves. The poller prints the board's age at startup for
exactly this reason — if it says anything other than a small number of hours, you are drafting
off stale market data and nothing else will tell you.

## The loop

Once per turn, in this order:

```bash
uv run python -m hub.draft.board --sync --pick 22
```

`--sync` pulls the draft from ESPN. **It may return nothing** — mock rooms publish no picks
to the API and it is untested whether the real draft does
([decisions.md](decisions.md)). If it reports 0 picks, type them instead:

```bash
uv run python -m hub.draft.board --taken "Ja'Marr Chase, Bijan Robinson" --pick 22
```

Names are matched loosely — punctuation, suffixes and case do not matter — and a name that
does not match the board is reported with a suggestion. **Read that warning.** An unmatched
pick leaves that player on your board as available.

Duplicates are safe: entering a pick twice is ignored rather than double-counted.

### Or run the poller

```bash
uv run python -m hub.draft.live --poll 10
```

Same recommendation, refreshed automatically, plus live replacement level and run detection.
It reads the board; it does not build one. If it dies, the board and `--pick` still work.

## Timing, measured

| command | time |
|---|---|
| `--pick N` | **5.0s** |
| `--taken "..." --pick N` | **4.6s** |
| `make draft` | **5.2s** |
| any of the above, no network | **0.5s** (serves the last good board) |

A 90-second clock has room for all of it several times over, which is why this still is not
worth optimising.

**The earlier 0.4s figures were wrong and did not reproduce.** Every invocation rebuilds the
board from source — there is no cached fast path, and `--pick` prints `loading ffopportunity`
every time — so the real cost is a full build. Measured three consecutive runs at 5.10 / 4.91 /
5.11s. The conclusion is unchanged and the number is now honest.

This used to read *16s with championship equity, 3s without*; equity was removed on
2026-08-24 ([ADR-0009](adr/0009-championship-equity-does-not-pick.md)) and the flags with it.

## Reading the output

**THE PICK** is the best available player who fills an unfilled starting slot, ranked on
**corrected ADP** — the market's ordering, moved by what this repo has measured the market to
miss. It is what beat every alternative that could be measured: the room, VOR ordering
(−5.06 pts/team-game) and championship equity (−19.66). Take it unless something below argues
otherwise.

The line says which market chose it:

| label | meaning |
|---|---|
| `by draft market, corrected` | normal — ADP, adjusted for our corrections |
| `by draft market (ADP)` | the board predates corrected ADP; rebuild it |
| `by consensus (ECR) -- no ADP today` | ESPN gave no ADP. This is arm A of the backtest and it is a good fallback, not a degraded one |

**The bracketed corrections are now folded into the ranking, bounded.** That changed on
2026-08-24. Touchdown luck, games missed and a current designation each carry a fitted
coefficient, and a correction may move a player up to 20% of his own ADP — about one round at
ADP 60, and almost nothing at the top of round one, where consensus is tightest.

They are still shown, because the size matters to you even when the clamp limits how much it
moves the board. On the live board 54 of 452 players move: 47 later, 7 earlier — running
*cold* on touchdowns marks a player up.

**"Also close" is context, not a ranking.** It sorts by value over replacement, which was
measured 5.06 points a team-game worse than the market. It is there to show you what else is
near and how near, with `cost_of_waiting` and the weeks 15–17 SoS column beside it. Do not
draft off its order.

`edge` is displayed and never sorted on. It cannot be validated — it needs historical ADP,
which ESPN does not retain ([ADR-0010](adr/0010-edge-is-displayed-but-never-ranked-on.md)).

## When something breaks

| symptom | what it means | what to do |
|---|---|---|
| `--sync` reports 0 picks | ESPN is not publishing mid-draft | type picks with `--taken` |
| `by consensus (ECR) -- no ADP today` | no ESPN ADP on the board | nothing. This is the designed fallback and it is the arm that won the backtest |
| `THE PICK unavailable` | the board has neither ADP nor ECR, so it did not build | serve `site/data/draft_board.json` — the top 300 by ECR, which covers all 192 picks |
| `NOT ON THE BOARD: 'x'` | mistyped pick | re-enter with the suggested spelling, then `--undo 1` the bad one |
| board build fails on a source | SoS, touchdown luck, durability and the two league checks each degrade on their own | it will say which, and print `built without: ...`; the board still builds |
| `BUILD FAILED: ...` then `serving the last good board` | **no network at all**, or ffverse/ESPN down. The two spine fetches have no in-build fallback, so the whole build is skipped | nothing. It prints how old the board is and carries on to THE PICK. ADP is that stale; every other column is a season-long number and does not move. Rebuild when the network returns |
| poller says the board is many hours old | you did not rebuild today | `make draft`, then restart the poller |
| wrong pick recorded | — | `--undo N` removes the last N |

## What is deliberately not automated

Whether to take the pick when a correction disagrees with the market by more than the clamp
allows. The clamp bounds how far a measurement may move a player; it does not decide whether
you believe the measurement. On the first live run the market wanted Jahmyr Gibbs at ADP 1.4
and our touchdown measurement said he ran +2.17 points a game hot — at ADP 1.4 the clamp
permits a move of 0.3 picks, so the board still says take him and the note still says he ran
hot. Two views, both shown. That one is yours.
