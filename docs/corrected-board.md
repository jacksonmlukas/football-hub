# The board that drafted, against the corrected board

Issue #50, computed 2026-09-11 after every correction in the pin-reprice-correct programme
(#39, #41, #48) and the ones that followed it (#155, #180, #226, #235). One diff, stated
once. The movement #86 will make is not folded in here and carries its own restatement.

## What the baseline is, and what it is not

**The board that drafted was never persisted.** The draft ran Thursday 3 September at
21:00 CT (02:00Z on the 4th). The board archive (`data/processed/boards/`) holds three full
builds on 2026-08-30 between 02:36Z and 02:40Z, nothing between then and the draft, and no
build on draft night; the next full build is 17:22Z on the 4th, fifteen hours after. No
commit landed between 2026-08-30 (`973719f`) and 04:07Z on the 4th. So the last full build
before the draft — `board-20260830T024026.parquet`, 449 players, 30 columns, ADP of the
30th — is the board that drafted as far as the record can say, and it is the baseline.

**Rebuilding it on the code that drafted does not reproduce it**, which is the first
acceptance criterion and is not met. On `973719f` with the archived board's own ESPN
columns (ADP, projection, designation) standing in for the fetch and the consensus taken
as of the 31st: `ecr` and `adp` reproduce exactly on all 449 players; `xfp_per_game`
differs on 116 of them, by up to 4.7, because the nflverse `ff_opportunity` table for 2025
has moved upstream since the 30th and nothing pins it to an as-of; and the consensus
returns 1,357 rows against the archived 449, which moves replacement level and so
`vor_proj` on every player. A Board digest exists now (#196, #197), so the next diff has a
baseline the archive can vouch for; this one does not.

## The diff, with the drafted inputs held fixed

Two boards differ by more than the corrections: the market moved for nine days. To see
the corrections alone, today's code was run on the drafted inputs — the same ESPN columns
and the consensus as of the 31st — and that board compared to the baseline by
corrected-ADP rank.

| | value |
|---|---|
| players priced in both | 449 |
| inside the first 168 (14 rounds), rank changes | **120 of 168** |
| mean absolute move | 3.2 places |
| largest | 27 (Daniel Jones, QB, 161 → 134) |
| moved a round or more (≥ 12) | 6 |

By position inside 168, mean signed move (negative = earlier now) and mean absolute:

| pos | n | mean | mean abs |
|---|---|---|---|
| QB | 26 | **−3.8** | 5.9 |
| RB | 54 | +0.9 | 1.1 |
| TE | 22 | +0.8 | 1.0 |
| WR | 66 | +0.6 | 2.3 |

The corrections move quarterbacks earlier and almost nothing else: Purdy 107 → 90, Stroud
159 → 144, Prescott 78 → 67, Dart 79 → 68, Burrow 58 → 51, Jones 161 → 134; Willis the
other way, 150 → 164. That is the durability term (#183, #235) pricing the position whose
seasons vary most in games played, and it is the whole of the signal — RB and TE move a
place on average.

**At the drafter's own turns** (slot 3 of 12), the player each board ranks at that overall
pick by corrected ADP:

| pick | drafted board | corrected board |
|---|---|---|
| 3 | Ja'Marr Chase | Ja'Marr Chase |
| 22 | A.J. Brown | Brock Bowers |
| 27 | George Pickens | George Pickens |
| 46 | Ladd McConkey | Ladd McConkey |
| 51 | D'Andre Swift | Joe Burrow |
| 70 | George Kittle | Rhamondre Stevenson |
| 75 | Michael Pittman Jr. | Michael Pittman Jr. |
| 94 | Jonathon Brooks | Jake Ferguson |
| 99 | Jakobi Meyers | Wan'Dale Robinson |
| 118 | Khalil Shakir | Josh Downs |

Four of ten turns name the same player; the six that differ are one-to-three-place swaps
in a dense part of the board, not a different draft. **The roster actually drafted** moves
at most two places on any of its fourteen skill players (Chase 3 → 3, Collins 25 → 26,
Rice 26 → 25, Jacobs 31 → 33, Tuten 59 → 60, McLaurin 61 → 62, Golden 90 → 91, Brooks
94 → 96, Andrews 103 → 104, Concepcion 122 → 122, Love 132 → 132, Rodriguez 144 → 143,
Lloyd 145 → 147, Boston 151 → 151). Nothing this programme corrected would have changed
a pick that was made.

## The diff against the board as served today

For the record, the same comparison against the served board of 2026-09-11 — today's
ADP, today's data — is 142 of 168 ranks changed, mean absolute move 4.8, and 13 by a round
or more. The extra movement is the market's: Josh Jacobs 31 → 67 and MarShawn Lloyd
145 → 100 are injury news priced by ESPN's ADP in the nine days since, not corrections.
Boston and Tucker have left ESPN's draftable pool since, and on the served board a player
with no ADP carries `NaN` rather than null in `adp_corrected` -- 293 rows -- which sorts
them to the tail; filed as #243.

## The lambda sweep, re-run

`docs/lambda-sweep.md` names `hub.draft.evaluate` as the decisive metric for
`projection_lambda = 0.0`, and #41 replaced that harness's opponent model with the unified
pick-noise base (fitted where the drafted sample is two-sided, #155). Re-run 2026-09-11,
144 drafts per cell, six holdouts:

| lam | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | mean | t | neg |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | −42.6 | +6.2 | +0.7 | +7.2 | +23.8 | +3.8 | −0.1 | −0.02 | 1/6 |
| 0.04 | −115.0 | −14.3 | −4.3 | +3.9 | +16.5 | +20.6 | −15.4 | −0.75 | 3/6 |
| **0.08** | −177.8 | −5.9 | −65.4 | −37.0 | +11.1 | +18.8 | **−42.7** | **−1.43** | 4/6 |
| 0.16 | −149.5 | −84.4 | −159.5 | −112.8 | +3.9 | −2.3 | −84.1 | −2.90 | 5/6 |
| 0.24 | −145.1 | −165.8 | −260.4 | −192.5 | −59.8 | −20.7 | −140.7 | −3.92 | 6/6 |
| 0.32 | −97.9 | −268.9 | −319.5 | −253.1 | −113.1 | −50.1 | −183.8 | −4.08 | 6/6 |

Best lambda per season: **0.00, 0.00, 0.00, 0.00, 0.02, 0.00** — five of six select exactly
zero where four did before, and the pooled reading at 0.08 is −42.7 (t −1.43) against the
published −45.3 (t −1.20). **The previous winner still wins**, by a little more.
