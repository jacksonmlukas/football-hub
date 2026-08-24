# Draft night runbook — Thu Sep 3, 9:00 PM, slot 3

Rehearsed 2026-08-24. Every command below was timed and every failure path was tried.

## Before you sit down

```bash
make draft                 # rebuild the board; ~40s
uv run python -m hub.draft.board --reset
```

The reset matters. Draft state persists on disk, and starting a real draft on top of leftover
picks from a rehearsal is a silent way to be wrong all night.

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

## Timing, measured

| command | time |
|---|---|
| `--pick N` (with championship equity) | **16s** |
| `--pick N --no-win-prob` | **3s** |
| `--taken "..."` | **4s** |

A 90-second clock has room for the full version. If the room is moving fast, drop
`--win-prob`: the equity table is a tiebreaker, not the recommendation.

## Reading the output

**THE PICK** is best available by ADP that fills an unfilled starting slot. It is what the
market would do, and it is what beat the room in the backtest ([next.md](next.md), P0). Take
it unless something below argues otherwise.

The bracketed corrections next to it are where our own measurements say the market is wrong —
touchdown luck, games missed last season, a current designation. **They are not priced into
ADP and they are not folded into the ranking.** They are yours to weigh.

**Championship equity** is a tiebreaker with a stated record: even with the market on
realised outcomes, +0.04 pts/team-game, 95% CI [−3.64, +3.58], n=36. Use it to choose among
close calls, not to override the market on its own.

Read the *lift*, never the level. Absolute P(win) is inflated because the season is scored on
the same projection the greedy ranks on.

## When something breaks

| symptom | what it means | what to do |
|---|---|---|
| `--sync` reports 0 picks | ESPN is not publishing mid-draft | type picks with `--taken` |
| `THE PICK unavailable` | no ESPN ADP on the board | use the equity table; the market has nothing to say without ADP |
| `NOT ON THE BOARD: 'x'` | mistyped pick | re-enter with the suggested spelling, then `--undo 1` the bad one |
| board build fails on a source | SoS, touchdown luck, durability and the scoring check each degrade on their own | it will say which; the board still builds |
| wrong pick recorded | — | `--undo N` removes the last N |

## What is deliberately not automated

Whether to take the market's pick when a correction disagrees with it. On the first live run
the market wanted Jahmyr Gibbs at ADP 1.4, our touchdown measurement said he ran +2.17 points
a game hot, and equity put him in the avoid tier. Three views, all shown, none of them
authoritative enough to overrule the others automatically.
