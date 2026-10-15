"""Does a margin sd fitted by weeks-out price a snapshot spread better than `MARGIN_SD`? (#249)

**Read `docs/lookahead-margin-sd.md` first.** The design, the buckets, the unit, the ceiling,
the power and the action each verdict takes were written there, in the commit that precedes this
one, before any 2026 margin residual existed. This docstring says only where each piece lives.

`MARGIN_SD` (12.741, `hub.models.market`) was fitted on closing spreads. The survivor grid prices
a distant week from a snapshot spread posted weeks earlier, so the margin around it is plausibly
wider. This measures whether a sd fitted per weeks-out bucket, on strictly earlier weeks, scores
better on held-out log loss than `MARGIN_SD` does. It reads `hub.models.margin` and
`hub.models.market` and **edits neither**, and **nothing imports this module**: it wires no price,
and the survivor consumer the ticket also asks for is HELD while #379 is dormant.

**The gate cannot return ADOPT or REMOVE on a plausible effect in 2026, and that is named.**
`docs/lookahead-margin-sd.md`'s power table puts the MDE 6 to 27 times the gain an sd 6-18%
wider than the close would earn, so the branch this takes is NOT-RUNNABLE, an exemption in
`docs/method.md` rule 16's sense: it decides nothing, neither that the lookahead sd equals the
closing one nor that it is wider. What the data can speak to is the dispersion, so `describe`
prints each bucket's sd with its interval beside `MARGIN_SD` as a description, not as a gate.

**The cluster is the week.** 2026 is the only season with snapshots, so a season cluster is one
observation and has no t interval. A `Harness` is season-clustered by construction and cannot
run this design, which is why this calls `experiment.run_gate` directly, as
`hub.season.weekly_forward` does for the same reason.

    uv run python -m hub.models.lookahead_margin --run
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from math import exp, sqrt
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.closure import closure_of
from hub.declare import not_an_input
from hub.ledger import Ledger, recipe
from hub.models.experiment import Actions, Ceiling, GateRun, expanding_weeks, run_gate
from hub.models.margin import fit, home_win_prob, home_won

# The incumbent. Imported rather than restated, so the two cannot drift apart.
from hub.models.market import MARGIN_SD
from hub.models.scoring_rules import log_loss

SEASON = 2026

# Weeks between a snapshot's `captured_at` and the kickoff, where one bucket ends and the next
# begins: b0 [0, 1), b1 [1, 2), b2 [2, 4), b3 [4, 8), b4 [8, inf). Fixed in the pre-registration
# before the first residual was computed. No function here takes the edges as an argument, so a
# caller cannot tune them after seeing data, and a test holds this against the document.
BUCKET_EDGES: tuple[float, ...] = not_an_input(
    (1.0, 2.0, 4.0, 8.0),
    "the bucket edges of a pre-registered gate, in weeks out; they shape which rows a "
    "comparison scores and no prediction reads them")
N_BUCKETS = len(BUCKET_EDGES) + 1

# Fewer earlier rows than this and a bucket is not fitted: it inherits the sd of the nearest
# bucket nearer the kickoff. Chosen, not fitted -- the relative error of a standard deviation
# over n rows is about 1 / sqrt(2(n - 1)), 13% at 30.
MIN_FIT_N = not_an_input(
    30, "the fewest earlier (game, bucket) rows a bucket needs before its own sd is fitted, "
        "pre-registered; a harness threshold that no prediction reads")

WEEK_SECONDS = 7 * 24 * 3600

# z for a two-sided 5% test plus z for 80% power: the pre-registered pair of docs/gate-power.md.
Z_ALPHA_POWER = not_an_input(
    1.96 + 0.84, "the sum of the two-sided 5% and 80%-power normal quantiles the ratio's MDE is "
                 "stated at; it sizes an experiment and nothing here predicts")

# The unit of independence is the week and the repeated measure within it is the game.
WITHIN: tuple[str, ...] = ("game_id",)
CLUSTER_WEEK: tuple[str, ...] = ("week",)

# The code this comparison runs: its own module, whose import closure is hashed into the ledger
# key (`Harness.arm_roots`' role, for a gate that is not a `Harness`).
ARM_ROOTS = ("hub.models.lookahead_margin",)

CEILING_ARM = ("each bucket's sd fitted in sample on every scored row of that bucket, "
               "flattered by construction")

ACTIONS = Actions(
    adopt="The bucketed sd scores better on held-out log loss than MARGIN_SD; reported to the "
          "maintainer, and nothing here wires it into a price.",
    remove="The bucketed sd scores worse than MARGIN_SD: MARGIN_SD stays on every row.",
    show="MARGIN_SD stays on every row; the gate could not establish that a bucketed sd pays.")

NAME = "lookahead_margin_sd"


class Fit(NamedTuple):
    """One bucket's sd for one held-out week, and how it was got."""

    sd: float
    kind: str      # "closing" (b0), "fitted", or "inherited" (too thin to fit)
    n: int


def weeks_out() -> pl.Expr:
    """Weeks from a snapshot's `captured_at` to its game's kickoff; positive when it precedes."""
    return (pl.col("kickoff") - pl.col("captured_at")).dt.total_seconds() / WEEK_SECONDS


def bucket_of(w: pl.Expr) -> pl.Expr:
    """The bucket index for a weeks-out expression, off `BUCKET_EDGES`."""
    out = pl.lit(len(BUCKET_EDGES), dtype=pl.Int64)
    for i in reversed(range(len(BUCKET_EDGES))):
        out = pl.when(w < BUCKET_EDGES[i]).then(pl.lit(i, dtype=pl.Int64)).otherwise(out)
    return out


def label(b: int) -> str:
    """`[1, 2)` for bucket 1: the weeks-out range a bucket spans."""
    lo = 0.0 if b == 0 else BUCKET_EDGES[b - 1]
    hi = "inf" if b >= len(BUCKET_EDGES) else f"{BUCKET_EDGES[b]:g}"
    return f"[{lo:g}, {hi})"


def build_rows(lines: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """One row per scored (game, bucket): the last snapshot captured in the bucket.

    `lines` carries `game_id`, `close_spread`, `captured_at`; `games` carries `game_id`,
    `season`, `week`, `kickoff`, `result`. A snapshot at or after kickoff is dropped -- the
    lookahead guard -- and a game without a result, or tied, is not scored (`margin.home_won`).
    `resid` is `result - spread`, the quantity `margin.fit` takes the dispersion of.
    """
    keep = ["game_id", "season", "week", "kickoff", "result"]
    joined = (lines.select("game_id", "close_spread", "captured_at")
                   .drop_nulls("close_spread")
                   .join(games.select(keep), on="game_id", how="inner")
                   .with_columns(weeks_out().alias("weeks_out"))
                   .filter(pl.col("weeks_out") > 0))
    scored = home_won(joined).with_columns(bucket_of(pl.col("weeks_out")).alias("bucket"))
    last = (scored.sort(["game_id", "bucket", "captured_at"])
                  .group_by(["game_id", "bucket"], maintain_order=True).last())
    return (last.select("season", "week", "game_id", "bucket",
                        pl.col("close_spread").alias("spread"), "result", "home_won",
                        (pl.col("result") - pl.col("close_spread")).alias("resid"))
                .with_columns(pl.col("week").cast(pl.Int64), pl.col("season").cast(pl.Int64))
                .sort(["season", "week", "game_id", "bucket"]))


def bucket_sds(past: pl.DataFrame, *, min_n: int | None = None) -> dict[int, Fit]:
    """Each bucket's sd from `past` alone. b0 is `MARGIN_SD`; a bucket with fewer than
    `MIN_FIT_N` rows inherits the nearest bucket nearer the kickoff, so the chain ends at the
    closing sd and a thin bucket can never widen a price on evidence it does not have."""
    floor = MIN_FIT_N if min_n is None else min_n
    out = {0: Fit(MARGIN_SD, "closing", int(past.filter(pl.col("bucket") == 0).height))}
    for b in range(1, N_BUCKETS):
        rows = past.filter(pl.col("bucket") == b)
        if rows.height >= max(floor, 2):
            out[b] = Fit(fit(rows)["sd"], "fitted", rows.height)
        else:
            out[b] = Fit(out[b - 1].sd, "inherited", rows.height)
    return out


def _row_loss(spread: np.ndarray, sd: np.ndarray, won: np.ndarray) -> np.ndarray:
    """Per-row log loss of `Phi(spread / sd)` against the outcome, each row its own sd."""
    return np.array([log_loss(home_win_prob(spread[i:i + 1], float(sd[i])), won[i:i + 1])
                     for i in range(len(spread))])


def walk_forward(rows: pl.DataFrame) -> pl.DataFrame:
    """The paired frame: one row per held-out (game, bucket) at bucket 1 or beyond.

    For each week with an earlier week behind it (`expanding_weeks`: strictly earlier, never the
    same week), fit `bucket_sds` on the earlier rows and score the week's rows under both arms.
    `diff = ll_A - ll_B` is positive when the bucketed sd is better; `ceiling_diff` is
    `ll_A - ll_oracle`, the oracle being each bucket's sd fitted in sample on every row of it.
    b0 is the closing end, priced as `MARGIN_SD` in both arms and left out of the gate.
    """
    schema = {"season": pl.Int64, "week": pl.Int64, "game_id": pl.Utf8, "bucket": pl.Int64,
              "sd_b": pl.Float64, "kind": pl.Utf8, "diff": pl.Float64,
              "ceiling_diff": pl.Float64}
    oracle = bucket_sds(rows, min_n=2)
    frames = []
    for season, week, past, now in expanding_weeks(rows):
        sds = bucket_sds(past)
        now = now.filter(pl.col("bucket") >= 1)
        if now.is_empty():
            continue
        spread = now["spread"].to_numpy().astype(float)
        won = now["home_won"].to_numpy().astype(float)
        buckets = now["bucket"].to_list()
        a = _row_loss(spread, np.full(len(spread), MARGIN_SD), won)
        b = _row_loss(spread, np.array([sds[k].sd for k in buckets]), won)
        o = _row_loss(spread, np.array([oracle[k].sd for k in buckets]), won)
        frames.append(pl.DataFrame({
            "season": [season] * now.height, "week": [week] * now.height,
            "game_id": now["game_id"], "bucket": buckets,
            "sd_b": [sds[k].sd for k in buckets], "kind": [sds[k].kind for k in buckets],
            "diff": a - b, "ceiling_diff": a - o}, schema=schema))
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def describe(rows: pl.DataFrame) -> pl.DataFrame:
    """The fit the ticket's first criterion asks for, in sample, as a description and not a gate.

    One row per bucket: n, the sd (fitted, or inherited when n is under `MIN_FIT_N`), its 95%
    interval `sd +/- 1.96 se` with `se = sd / sqrt(2(n - 1))`, the ratio to `MARGIN_SD`, the
    smallest ratio that n resolves at 80% power (`exp(Z / sqrt(2(n - 1)))`), and whether the
    interval excludes `MARGIN_SD` from above. A bucket with no rows reports NaN, not zero.
    """
    sds = bucket_sds(rows)
    out = []
    for b in range(N_BUCKETS):
        n = int(rows.filter(pl.col("bucket") == b).height)
        f = sds[b]
        usable = n >= 2 and b >= 1 and f.kind == "fitted"
        se = f.sd / sqrt(2 * (n - 1)) if usable else float("nan")
        lo, hi = (f.sd - 1.96 * se, f.sd + 1.96 * se) if usable else (float("nan"),) * 2
        out.append({"bucket": b, "weeks_out": label(b), "n": n, "sd": f.sd, "kind": f.kind,
                    "se": se, "lo": lo, "hi": hi, "ratio": f.sd / MARGIN_SD,
                    "mde_ratio": exp(Z_ALPHA_POWER / sqrt(2 * (n - 1))) if n >= 2
                    else float("nan"),
                    "above": bool(usable and lo > MARGIN_SD)})
    return pl.DataFrame(out)


def describe_lines(table: pl.DataFrame) -> list[str]:
    lines = [f"  {'bucket':<8}{'weeks out':<11}{'n':>4}  {'sd':>7}  {'95% interval':<17}"
             f"{'ratio':>6}  {'MDE ratio':>9}  note"]
    for r in table.iter_rows(named=True):
        ci = (f"[{r['lo']:.2f}, {r['hi']:.2f}]" if r["kind"] == "fitted" and r["bucket"]
              and r["n"] >= 2 else "-")
        mde = "-" if r["mde_ratio"] != r["mde_ratio"] else f"{r['mde_ratio']:.2f}"
        note = {"closing": "the closing end, MARGIN_SD",
                "fitted": "fitted" + (", interval excludes MARGIN_SD from above"
                                      if r["above"] else ""),
                "inherited": (f"INHERITED from the nearer bucket (n {r['n']} < {MIN_FIT_N})"
                              if r["n"] else "INHERITED, no rows")}[r["kind"]]
        lines.append(f"  b{r['bucket']:<7}{r['weeks_out']:<11}{r['n']:>4}  {r['sd']:>7.3f}  "
                     f"{ci:<17}{r['ratio']:>6.2f}  {mde:>9}  {note}")
    return lines


def span_lines(lines: pl.DataFrame) -> list[str]:
    """What the archive holds, so a run states the coverage it was given and not only its span."""
    if lines.is_empty():
        return ["  archive: no snapshots"]
    at = lines["captured_at"]
    first, last = at.min(), at.max()
    days = (last - first).days  # type: ignore[operator]
    weeks = lines.select(pl.col("captured_at").dt.truncate("1w").n_unique()).item()
    return [f"  archive: {first:%Y-%m-%d} to {last:%Y-%m-%d} ({days} days, {days / 7:.1f} weeks), "
            f"{at.n_unique()} capture instants in {weeks} calendar weeks"]


def gate_run(paired: pl.DataFrame, *, seasons: Sequence[int] = (SEASON,), seed: int = 0,
             ledger: Ledger | None = None) -> GateRun:
    """The comparison through the one rule (`experiment.run_gate`), clustered on the week."""
    ceiling = None
    if "ceiling_diff" in paired.columns and paired.height:
        ceiling = Ceiling(CEILING_ARM, paired["ceiling_diff"].to_numpy())
    return run_gate(
        paired.drop("ceiling_diff", strict=False), cluster=CLUSTER_WEEK, within=WITHIN,
        actions=ACTIONS, name=NAME, arm_a="bucketed sd", arm_b="MARGIN_SD",
        unit="nats per game-bucket", places=4, ceiling=ceiling, seed=seed, ledger=ledger,
        recipe=recipe(seasons=list(seasons), edges=list(BUCKET_EDGES), min_fit_n=MIN_FIT_N),
        code_modules=closure_of(ARM_ROOTS))


def load_2026() -> tuple[pl.DataFrame, pl.DataFrame]:  # pragma: no cover - reads the store
    """The season's snapshot archive and its schedule with results, as `build_rows` takes them."""
    from hub import store
    from hub.fetch import nflverse
    from hub.schedule import kickoff_expr
    sched = nflverse.load("schedules", [SEASON])
    games = sched.select("game_id", "season", "week", kickoff_expr(), "result")
    return store.lines(SEASON), games


def main(argv: Sequence[str] | None = None, *,
         load: Callable[[], tuple[pl.DataFrame, pl.DataFrame]] | None = None,
         ledger: Ledger | None = None) -> int:
    """`load` and `ledger` are #403's seam: the one store/network read and the ledger written to."""
    ap = argparse.ArgumentParser(
        prog="hub.models.lookahead_margin",
        description="A margin sd fitted by weeks-out against MARGIN_SD, on the snapshot archive.")
    ap.add_argument("--run", action="store_true", help="measure and write one ledger entry")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0
    try:
        lines, games = (load or load_2026)()
    except Exception as e:
        return unavailable("hub.models.lookahead_margin", "the snapshot archive and schedule", e)
    rows = build_rows(lines, games)
    print("\n".join(span_lines(lines)))
    print(f"  {rows.height} scored (game, bucket) rows over {rows['game_id'].n_unique()} games")
    if rows.is_empty():
        print("  nothing scored -- no snapshot precedes a played game", file=sys.stderr)
        return 1
    print("\n  the fit, in sample, as a description (not the gate):")
    print("\n".join(describe_lines(describe(rows))))
    paired = walk_forward(rows)
    run = gate_run(paired, ledger=ledger)
    label_, sentence = run.verdict
    print(f"\n  {label_}: {sentence}")
    print("\n".join(run.lines))
    print("\n  Nothing is wired: no price, page or survivor row reads this module. The survivor "
          "consumer of #249 is HELD while #379 is dormant.")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
