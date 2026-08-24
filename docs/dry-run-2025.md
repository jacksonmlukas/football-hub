# Dry run: replaying the 2025 draft

Phase 2.5 of `docs/foundation-plan.md`, run 2026-08-23, eleven days before the draft.

The done-when is unusual and worth restating: **"you find at least one bug. If you find
none, the dry run was not realistic."** Three surfaced. All three were invisible to a
green test suite, and all three were found by reading the poller's output at the seven
picks that are actually mine rather than by asserting on it.

## Method

`make draft` for a fresh board, then `hub.draft.live --replay 2025` over the real 192-pick
draft, stopping at picks 3, 22, 27, 46 and 51 to read what the board would have told me.

## What broke

### 1. Unmatched picks were silent

`state.unmatched()` existed and `board.py` printed it. `live.py` — the module actually
running during a draft — never called it. A pick whose name does not normalise to a board
entry leaves that player **on the board**, so the poller keeps recommending someone who is
already gone, with no warning.

Reproduced by drafting a player under a name the normaliser cannot reach: three picks
recorded, one unmatched, and the board's top recommendation was a player who had just been
taken.

The 2025 draft has 28 such names against today's board. Most are benign — 13 D/ST and 10
kickers, both excluded from the board by design. But three are genuine skill players
(Ricky Pearsall, Jayden Higgins, Adam Thielen) who simply are not in the 2026 top 452. The
count being mostly benign is exactly why it needs surfacing rather than suppressing: on
draft day a real mismatch would look identical to these.

**Fix:** `refresh()` returns `unmatched`, and `render()` leads with it.

### 2. `my_slot` was accepted and never used

The poller took a slot argument, used it to work out which pick was next, and never once
looked at what that slot already held.

Replaying to pick 51 I was holding **RB3 WR1, with no QB and no TE** — and the board
offered four more WRs. `season.py` had already established that a surplus player scores
exactly zero, because you can only start three; a board recommending a fifth WR over an
empty TE slot is recommending a player worth nothing.

**Fix:** `refresh()` returns the roster; `render()` prints `you hold: RB3 WR1 | need QB,
WR, TE` and stars every candidate that fills an empty starting slot.

### 3. Below-replacement players could top the board

From round 4 the ranking switches to `edge`, and nothing checked value. At pick 46 the
board opened with **Jordan Love at VOR −1.0** — below replacement, which means worse than
what will sit on waivers all season — purely because his edge was large.

This is the same class of error as the round-3 rule added earlier, one round further on.
The earlier fix stopped edge ranking rounds 1–3; it did not stop edge ranking *bad players*
from round 4.

**Fix:** `rank()` filters `vor_live > 0` before sorting, whatever the key.

## What held

- **Latency.** 192 picks, slowest refresh 53ms against a 2000ms budget. The fixes cost
  7ms → 53ms, still 38x margin.
- **Live replacement level.** RB falls 11.0 → 9.1 → 8.5 → 7.1 → 6.6 across the first five
  rounds, and the remaining backs reprice against it.
- **Run detection.** WR x50, RB x39, QB x2 over the real draft. The WR-heavy pattern is what
  a full-PPR room does.
- **Pick schedule and mode.** 3, 22, 27, 46, 51 with scarcity/value alternating correctly.

## Live poller run (2026-08-23)

The section below used to say `--poll` had never run against a live draft. It has now been
run against the real 2026 league, in the state it is actually in: authenticated, reachable,
draft not yet started. Three more bugs, none of which replay could have surfaced because
they are all about the loop rather than the board.

### 4. Twenty-two seconds of nothing when piped

Run with stdout to a pipe, the poller produced **no output at all**. Python block-buffers a
pipe, and nothing in the loop flushed. In a terminal it is line-buffered and fine, which is
exactly why replay and the unit tests never caught it -- but `--poll 10 | tee draft.log` is
a reasonable way to keep a record of draft night, and it would have shown a blank file.

**Fix:** every print in the loop flushes.

### 5. The board scrolled off screen

`sync_from_espn` announced "draft is empty (not started?)" on every pass. At 10s over a
three-hour draft that is roughly a thousand identical lines pushing the one thing you need
to read out of view. The change guard suppressed re-rendering the board but not the sync
message underneath it.

**Fix:** the poller syncs quietly and prints a heartbeat every two minutes instead --
enough to tell a working poller from a hung one at 2am, which is a distinction worth
ninety seconds.

### 6. SIGTERM skipped the fallback message

Only `KeyboardInterrupt` was handled. A poller killed rather than Ctrl-C'd exited without
saying where the static board is -- which is the one thing you need at the moment it dies.

**Fix:** `SIGTERM` handled too; both paths print the fallback and exit 0.

### The degradation chain, verified

If ESPN sync fails mid-draft the operator records picks by hand and the poller keeps
working: `board --taken "..."` writes to disk, `sync_from_espn` falls back to that on an
empty or unreachable draft, and the board reprices. Confirmed end to end -- two manual
picks removed the right players and repriced the top of the board.

## What this did not test

The poller has now run against the live league, but **not against a live draft** -- there
is not one until Sep 3. What remains unexercised is the only part that needs picks to
arrive: the transition when `n_taken` starts moving, mid-draft reconnection after a dropped
request, and behaviour if ESPN returns a partial pick list. The sync, the loop, the
interval, the flush, the shutdown path and the manual fallback are all now confirmed
against the real league.
