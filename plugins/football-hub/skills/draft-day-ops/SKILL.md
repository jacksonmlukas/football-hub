---
name: draft-day-ops
description: Use when running or preparing for a live fantasy draft — building the board, polling the ESPN draft feed, detecting positional runs, or deciding the next pick. Triggers on "draft board", "who should I take", "positional run", "draft day", "my pick is up".
---

# Draft Day Ops

Operating procedure for a live 12-team full-PPR ESPN draft. Latency matters: you have roughly
90 seconds per pick, so every answer must be one command away.

## Before the draft

1. `make draft` rebuilds the board. Re-run the morning of, so ECR reflects late news.
2. Verify `.env` has `ESPN_S2`, `ESPN_SWID`, `ESPN_LEAGUE_ID`. Without them the ADP diff is dark
   and you lose the main edge.
3. Confirm roster slots came from the league API rather than the hardcoded default:
   `python -m hub.draft.board --show-slots`

## During the draft

Run the live board in a second terminal. Do not ask Claude to recompute mid-draft; the poller
already does it.

```
python -m hub.draft.live --poll 10
```

Answer picks from three columns, in priority order:

| Column | Meaning | Use it for |
|---|---|---|
| `edge` | ECR rank minus ESPN ADP | Main signal. Positive means the room lets him fall. |
| `vor` | xFP/gm above positional replacement | Breaking ties inside a tier. |
| `sd` / `worst` | ECR dispersion | Rounds 1-4 favor low `sd`. Round 10+ favors high `worst`. |

## Positional run detection

A run is three or more picks at one position inside the last five. When one fires, value at that
position collapses for the next round and rises everywhere else. The live board flags it.

**Do not chase a run.** Take the best player at a position nobody just drafted. The run has
already moved the market against you at the position being run.

## Rules

- Never draft on `edge` alone above round 3. Consensus is tight at the top, so a large `edge`
  there usually means a name-matching failure rather than an opportunity.
- Rookies have null `xfp`. Expected, not a bug. Rank them on ECR alone.
- If the poller dies, the static board in `site/data/draft_board.json` is still correct. Use it.

## When not to use

Not for waiver or trade decisions after the draft. Those go through `weekly-slate`.
