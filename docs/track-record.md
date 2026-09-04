# Public track record

Published warts and all. Three design rules make the difference between a credible record and a
tout account.

## 1. Predictions are committed before kickoff, or they do not count

The value of a public record is entirely in the timestamp. A results page assembled after the
fact proves nothing, and any reader who matters knows that.

```
hub.publish --predictions --week N   # writes site/data/preds_wk{N}.json
git commit -m "wk{N} predictions"    # BEFORE first kickoff
```

The git history is the pre-registration. A reader can check that the commit predates the game.
Nothing else you can do for free carries that much credibility.

The Sunday Actions job must run and commit before the earliest kickoff, not on a "sometime
Sunday" cron. Set it early and accept the slack.

### What this does and does not prove

The commit proves *when*. It does not prove *from what*, and the two are worth keeping apart.

Every prediction carries the market number it was priced from, which source that was, and --
for a dated snapshot -- when that snapshot was captured. A reader can see all of it. What a
reader cannot currently do is fetch the input back and check it independently, and the reason
differs by source:

* a **snapshot** is immutable, and unpublished. `.gitignore` excludes the processed store as
  redistributed third-party data the repo cannot publish.
* the **moving field** is published, and has moved. nflverse keeps one current `spread_line`
  per game and no history, so the lookahead value is gone.

So the record is checkable on its timestamp and corroborable on neither of its inputs. That is
stated in every weekly artifact, under `provenance`, rather than left for a reader to discover.
Publishing a derived per-game summary of the snapshots would change the first of those; nothing
changes the second.

## 2. The page is generated, never edited

`hub.publish --track-record` reads `data/processed/preds/**/*.parquet` and regenerates the whole
page every week. No hand-curation, no ability to quietly drop a bad week. If you can edit it,
a reader has to trust you, and the point is that they should not have to.

## 3. Show calibration, not accuracy

This is what separates evaluation infrastructure from a picks account:

- **Reliability diagram**, ten bins, with counts per bin
- **Log-loss and Brier** against the market-only baseline, with the delta and a bootstrap interval
- **Conformal coverage**: nominal versus empirical, tracked weekly
- **Every number carries its interval.** With ~285 NFL games a season, most weekly deltas are
  noise, and showing that you know it is the credibility signal.
- **A wrong-predictions section**, not buried. The biggest misses, with what the model saw.

## The honest risk

A market-anchored model will mostly track the market. The record will look unexciting, and there
will be a stretch where it looks bad. Both are fine and both are the point.

Mitigation is not spin, it is sample-size honesty. If every figure ships with an interval wide
enough to contain a bad four-week run, a bad four-week run reads as noise because it is noise.
The failure mode to avoid is publishing point estimates all season and then reaching for
"small sample" only when the numbers turn.

## What to skip

No units, no bankroll, no ROI. Edge ranks last among this project's objectives and a public P&L
invites exactly the wrong reading of the work. Log closing lines and CLV privately — costs
nothing now, impossible to reconstruct later — and keep the public page about calibration.
