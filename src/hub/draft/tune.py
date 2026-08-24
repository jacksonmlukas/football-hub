"""Set `projection_lambda` from evidence instead of judgment.

`hub.draft.projection` nudges consensus by last season's regression signal:

    adj_ecr = ecr * exp(-lam * z)

`lam` was 0.08, chosen by reasoning rather than measurement. This builds the holdout that
can answer the question: take 2024's expected-vs-actual gap, apply it to the 2025 preseason
board, and score the result against what 2025 actually did.

Two things about the design matter more than the number it produces.

**Zero has to be reachable.** The honest answer for a weak signal is "do not adjust", so
the sweep always includes 0.0 and every result is reported as a delta against the untouched
consensus board. A tuner that cannot return zero is a machine for fitting noise.

**The harness is validated before it is trusted.** `tests/unit/test_tune.py` plants a signal
that genuinely predicts and checks the sweep finds it, then plants pure noise and checks it
returns zero. A tuner that cannot recover a known answer cannot be believed about an
unknown one.

One holdout is one season. Whatever this returns is a weak estimate, and the write-up says
so rather than implying a precision the design cannot support.

    uv run python -m hub.draft.tune --sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SWEEP_OUT = ROOT / "docs" / "lambda-sweep.md"

# Spanning "ignore the signal" to "nudge hard". 0.08 is the incumbent, kept in the grid so
# the write-up can say what it would have scored.
DEFAULT_GRID: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32)

# A 12-team league drafts ~192 players, but the first 50 picks decide a season. Scoring the
# whole board would let deep-bench noise drown the part that matters.
TOP_N = 50

# A 12-team league drafts 16 rounds. Beyond that the ordering is not a draft decision.
DRAFTABLE = 192


def apply(df: pl.DataFrame, lam: float) -> pl.DataFrame:
    """The projection adjustment, as a pure function of (ecr, z, lam)."""
    z = pl.col("z_regress").fill_null(0.0).clip(-3.0, 3.0)
    return df.with_columns((pl.col("ecr") * np.e ** (-lam * z)).alias("adj_ecr"))


def score(df: pl.DataFrame, lam: float, top_n: int = TOP_N) -> dict[str, float]:
    """How good a board this lambda produces, against what actually happened.

    Two metrics because they answer different questions. Spearman asks whether the whole
    ordering improved. `top{N}_points` asks the question a drafter actually has: if I take
    the first N names on this board, how many points do I end up with?
    """
    if df.is_empty():
        return {"spearman": float("nan"), "spearman_pool": float("nan"),
                f"top{top_n}_points": 0.0}

    ranked = apply(df, lam).sort("adj_ecr")
    pred = np.arange(1, ranked.height + 1, dtype=float)
    actual = ranked["actual_points"].fill_null(0.0).to_numpy()

    # Spearman is Pearson on ranks. Negated because a better board means a LOW adj_ecr
    # pairs with HIGH points, and a positive number reading as "good" is worth the line.
    actual_rank = actual.argsort().argsort().astype(float)
    if pred.std() == 0 or actual_rank.std() == 0:
        rho = float("nan")
    else:
        rho = -float(np.corrcoef(pred, actual_rank)[0, 1])

    # Spearman restricted to the draftable pool as well as over everything. A 12-team
    # league makes 192 picks, so the ordering of players ranked 400th is noise no draft
    # ever consults -- but it is a third of the board and it can dominate a full-board
    # correlation. Reporting both keeps that visible instead of choosing silently.
    pool = min(DRAFTABLE, ranked.height)
    pr, ar = pred[:pool], actual_rank[:pool]
    rho_pool = (float("nan") if pr.std() == 0 or ar.std() == 0
                else -float(np.corrcoef(pr, ar)[0, 1]))

    return {"spearman": rho,
            "spearman_pool": rho_pool,
            f"top{top_n}_points": float(actual[:top_n].sum())}


# Resamples used to ask whether an improvement is distinguishable from noise.
BOOTSTRAP = 300

# How many standard errors an improvement must clear before it counts. Two is the usual
# bar and it is the right one here: consensus is a strong prior and moving off it should
# require evidence, not a favourable draw.
MIN_SIGMA = 2.0


def sweep(df: pl.DataFrame, lams: Sequence[float] = DEFAULT_GRID,
          top_n: int = TOP_N, bootstrap: int = BOOTSTRAP, seed: int = 0) -> pl.DataFrame:
    """Score every lambda against the untouched board, with a bootstrap standard error.

    The point estimate alone is not enough, and the first version of this proved it: on a
    holdout built from pure noise, lambda 0.05 beat 0.00 on captured points purely by
    chance. Any perturbation reshuffles which players land in the top N, and on one
    realisation some reshuffle wins. So each lambda is re-scored over resampled holdouts
    and reported with the spread, and the selection rule below reads both.
    """
    base = score(df, 0.0, top_n)[f"top{top_n}_points"]
    col = f"top{top_n}_points"

    deltas: dict[float, np.ndarray] = {}
    rho_deltas: dict[float, np.ndarray] = {}
    if not df.is_empty() and bootstrap:
        rng = np.random.default_rng(seed)
        idx = [rng.integers(0, df.height, df.height) for _ in range(bootstrap)]
        samples = [df[list(i)] for i in idx]
        base_scores = [score(x, 0.0, top_n) for x in samples]
        for lam in lams:
            got = [score(x, lam, top_n) for x in samples]
            deltas[float(lam)] = np.array(
                [g[col] - b[col] for g, b in zip(got, base_scores)])
            rho_deltas[float(lam)] = np.array(
                [g["spearman"] - b["spearman"] for g, b in zip(got, base_scores)])

    rows = []
    for lam in lams:
        s = score(df, lam, top_n)
        d, rd = deltas.get(float(lam)), rho_deltas.get(float(lam))
        rows.append({
            "lam": float(lam),
            "spearman": s["spearman"],
            "spearman_pool": s["spearman_pool"],
            col: s[col],
            "delta_vs_consensus": s[col] - base,
            "delta_mean": float(d.mean()) if d is not None else 0.0,
            "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))) if d is not None and len(d) > 1
                        else 0.0,
            "rho_delta_mean": float(rd.mean()) if rd is not None else 0.0,
            "rho_delta_se": (float(rd.std(ddof=1) / np.sqrt(len(rd)))
                             if rd is not None and len(rd) > 1 else 0.0),
        })
    return pl.DataFrame(rows)


def best_lambda_spearman(swept: pl.DataFrame, min_sigma: float = MIN_SIGMA) -> float:
    """Select on whole-board rank correlation rather than top-50 points.

    This is the better-powered criterion and the reason is structural: Spearman reads every
    player, while top-50 points rides on the two or three sitting nearest the cut. Six
    seasons of the points metric came to roughly fifteen player-outcomes; the same six
    seasons of Spearman come to six times the whole board.
    """
    if swept.is_empty() or "rho_delta_mean" not in swept.columns:
        return 0.0
    credible = swept.filter(
        (pl.col("lam") > 0)
        & (pl.col("rho_delta_mean") > min_sigma * pl.col("rho_delta_se"))
        & (pl.col("rho_delta_mean") > 0))
    if credible.is_empty():
        return 0.0
    return float(credible.sort(["rho_delta_mean", "lam"],
                               descending=[True, False])["lam"][0])


def best_lambda(swept: pl.DataFrame, top_n: int = TOP_N,
                min_sigma: float = MIN_SIGMA) -> float:
    """The best lambda whose improvement clears the noise, else zero.

    Zero is a real answer, not a failure mode. Consensus aggregates hundreds of analysts;
    a nudge that cannot be distinguished from a lucky resample has not earned the right to
    move it.
    """
    if swept.is_empty():
        return 0.0
    col = f"top{top_n}_points"
    if "delta_mean" not in swept.columns:
        return float(swept.sort([col, "lam"], descending=[True, False])["lam"][0])

    credible = swept.filter(
        (pl.col("lam") > 0)
        & (pl.col("delta_mean") > min_sigma * pl.col("delta_se"))
        & (pl.col("delta_mean") > 0))
    if credible.is_empty():
        return 0.0
    return float(credible.sort(["delta_mean", "lam"], descending=[True, False])["lam"][0])


# FantasyPros changed its page taxonomy between 2020 and 2021. Before the change the
# redraft board was `redraft-offense`; after, `redraft-overall`. The two never coexist in
# any preseason, so there is no overlap year in which to check they are the same board.
# A 2020 result is therefore NOT interchangeable with the others and must be reported
# separately rather than pooled as if it were.
BOARD_PAGE = "redraft-overall"
LEGACY_BOARD_PAGE = "redraft-offense"
LEGACY_THROUGH = 2020


def board_page(board_season: int) -> str:
    return LEGACY_BOARD_PAGE if board_season <= LEGACY_THROUGH else BOARD_PAGE


def holdout(signal_season: int = 2024, board_season: int = 2025,
            page: str | None = None) -> pl.DataFrame:
    """Last season's regression signal, this season's preseason board, this season's truth.

    The board is the LAST preseason ECR snapshot before the season opened, because that is
    what a drafter would actually have had in hand.
    """
    import nflreadpy as nfl

    from hub.draft.projection import regression_signal
    from hub.draft.state import _norm

    sig = (nfl.load_ff_opportunity(seasons=[signal_season], stat_type="weekly")
           .filter(pl.col("player_id").is_not_null())
           .group_by(["full_name", "position"])
           .agg(pl.col("total_fantasy_points_exp").sum().alias("xfp"),
                pl.col("total_fantasy_points").sum().alias("fp")))
    sig = regression_signal(sig).select(
        pl.col("full_name"), pl.col("z_regress"))

    # Bounded on BOTH sides. Without a lower bound, a player ranked in a previous
    # preseason but not this one carries his stale rank forward and is scored as though
    # the market still rated him. Harmless-looking, and it would quietly differ between
    # season pairs -- which is exactly what a multi-season comparison must not do.
    board = (nfl.load_ff_rankings("all")
             .filter((pl.col("page_type") == (page or board_page(board_season)))
                     & (pl.col("scrape_date") < f"{board_season}-09-01")
                     & (pl.col("scrape_date") >= f"{board_season}-07-01")
                     & pl.col("ecr").is_not_null())
             .sort("scrape_date", descending=True)
             .unique(subset=["player"], keep="first")
             .select(pl.col("player"), pl.col("ecr")))

    truth = (nfl.load_ff_opportunity(seasons=[board_season], stat_type="weekly")
             .filter(pl.col("player_id").is_not_null())
             .group_by("full_name")
             .agg(pl.col("total_fantasy_points").sum().alias("actual_points")))

    key = pl.col("player").map_elements(_norm, return_dtype=pl.Utf8).alias("_k")
    sig_k = sig.with_columns(
        pl.col("full_name").map_elements(_norm, return_dtype=pl.Utf8).alias("_k")).drop("full_name")
    truth_k = truth.with_columns(
        pl.col("full_name").map_elements(_norm, return_dtype=pl.Utf8).alias("_k")).drop("full_name")

    return (board.with_columns(key)
            .join(sig_k, on="_k", how="left")
            .join(truth_k, on="_k", how="inner")   # no truth, no evidence
            .drop("_k")
            .sort("ecr"))


def report(swept: pl.DataFrame, df: pl.DataFrame, top_n: int = TOP_N) -> list[str]:
    best = best_lambda(swept, top_n)
    incumbent = swept.filter(pl.col("lam") == 0.08)
    lines = [
        f"  holdout: {df.height} players with a 2025 finish and a preseason rank",
        f"  with a 2024 signal: {int(df['z_regress'].is_not_null().sum())}",
        "",
        f"  {'lam':>6} {'spearman':>10} {'top' + str(top_n) + ' pts':>12} {'vs consensus':>14}",
    ]
    for r in swept.iter_rows(named=True):
        lines.append(f"  {r['lam']:>6.2f} {r['spearman']:>10.4f} "
                     f"{r[f'top{top_n}_points']:>12,.0f} "
                     f"{r['delta_vs_consensus']:>+14,.1f}")
    lines.append("")
    lines.append(f"  best by points captured: lam={best:.2f}")
    if incumbent.height:
        lines.append(f"  incumbent 0.08 scored {incumbent['delta_vs_consensus'][0]:+,.1f} "
                     f"against consensus")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.tune",
        description="Set projection_lambda from a holdout instead of judgment.")
    ap.add_argument("--sweep", action="store_true", help="run the holdout sweep")
    ap.add_argument("--signal-season", type=int, default=2024)
    ap.add_argument("--board-season", type=int, default=2025)
    a = ap.parse_args(argv)
    if not a.sweep:
        ap.print_help()
        return 0

    df = holdout(a.signal_season, a.board_season)
    swept = sweep(df)
    print(f"  projection_lambda sweep: {a.signal_season} signal -> "
          f"{a.board_season} board -> {a.board_season} results")
    print("\n".join(report(swept, df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
