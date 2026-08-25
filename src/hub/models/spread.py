"""Is weekly spread a property of the player, or only of his mean?

`hub.models.predict.moments` sets `sd = K[position] * sqrt(mu)`. The exponent is right --
0.498 +/- 0.012, about 42 se from 1, see docs/weekly-spread.md -- and that fit is not in
question here. What is in question is `K[position]`: a *constant* per position, which makes
`sd` a deterministic increasing function of `mu`.

That is the structural complaint in ADR-0012. If `sd` rises monotonically with `mu`, then
ordering players by `mu` and ordering them by any mean-variance objective give the same
answer, so the lineup optimiser can never disagree with the projection. It was measured at
+0.00 points a game, and this is why.

For the optimiser to have anything to optimise, two players with the *same* mean must be
able to have different spreads. This module asks whether they do, and whether that
difference is predictable a year ahead.

docs/weekly-spread.md already names the suspect:

    Within-season trend counts as spread. A player whose role grows across a season shows
    that growth as weekly variance. Some of k is really usage drift.

So the candidates are ordered by how much they assume:

* `positional` -- `k = K[position]`. **The shipped model, and the one to beat.**
* `own_k` -- the player's own prior-season `k`, shrunk toward `K[position]`. Assumes only
  that volatility is a persistent property of a player. Needs no usage data at all, and if
  this fails there is little reason to expect usage features to succeed.
* `usage` -- `K[position]` scaled by a fitted function of prior-season role: target share,
  snap share, air-yards share, touchdown rate, and the within-season drift of snap share
  that the quote above blames.

**The gate, fixed before any of this was run:** a candidate is adopted only if it beats
`positional` on held-out mean absolute error of predicted `sd`, in **every** held-out
season, and the paired difference clears 2 standard errors. Beating it on average while
losing a season is not enough -- that is how a fit gets adopted on one lucky year.

Both arms are always given the *same* `mu`, so the comparison isolates the spread question
from projection error and cannot favour either arm.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.models.predict import WEEKLY_K, WEEKLY_K_POOLED

POSITIONS = ("QB", "RB", "WR", "TE")

# Matching docs/weekly-spread.md's sample exactly, so the two measurements are comparable.
# Below 8 games the sd is a handful of numbers; below 3 ppg the ratio sd/sqrt(mu) is
# dominated by whether he happened to score once.
MIN_GAMES = 8
MIN_PPG = 3.0

CANDIDATES = ("positional", "own_k", "usage")

# Standard errors the paired difference must clear before a candidate replaces the shipped
# model. Two is the repo's usual bar; the gain has to be real, not just consistently signed.
MIN_SE = 2.0

# Prior-season role features for the `usage` arm. `drift` is the within-season slope of
# snap share, which is the term docs/weekly-spread.md accuses of masquerading as spread.
FEATURES = ("tgt_share", "ay_share", "td_pg", "snap_pct", "drift", "log_mu")

_TD_COLS = ("receiving_tds", "rushing_tds", "passing_tds")


def _positional_k(pos: pl.Expr) -> pl.Expr:
    return pos.cast(pl.Utf8).fill_null("").replace_strict(
        WEEKLY_K, default=WEEKLY_K_POOLED, return_dtype=pl.Float64)


def _opt(df: pl.DataFrame, c: str) -> pl.Expr:
    """A column that may not exist in this slice of nflverse, as nulls rather than a crash."""
    return pl.col(c).cast(pl.Float64) if c in df.columns else pl.lit(None, pl.Float64)


def _slope(x: str, y: str) -> pl.Expr:
    """OLS slope of `y` on `x` within a group, written out rather than via `pl.cov`.

    Returns null for a group with no spread in `x` -- a one-week player has no trend, and
    dividing by a zero variance would put an infinity into the feature matrix.
    """
    xm, ym = pl.col(x).mean(), pl.col(y).mean()
    var = (pl.col(x) - xm).pow(2).sum()
    cov = ((pl.col(x) - xm) * (pl.col(y) - ym)).sum()
    return pl.when(var > 0).then(cov / var).otherwise(None)


def snap_usage(snaps: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Per player-season snap share and its within-season drift, keyed on `player_id`.

    `snap_counts` keys on `pfr_player_id` while everything else in this repo keys on
    nflverse's `gsis_id`, so this join is unavoidable. Players it cannot match come back
    absent rather than zero: a missing snap share is unknown, and zero would assert he
    never played.
    """
    need = {"season", "week", "pfr_player_id", "offense_pct"}
    missing = need - set(snaps.columns)
    if missing:
        raise ValueError(f"snap_usage needs {sorted(missing)}")
    if not {"pfr_id", "gsis_id"} <= set(crosswalk.columns):
        raise ValueError("crosswalk needs pfr_id and gsis_id")

    s = snaps
    if "position" in s.columns:
        s = s.filter(pl.col("position").is_in(POSITIONS))
    pct = pl.col("offense_pct").cast(pl.Float64)
    # nflverse has shipped this both as a fraction and as a percentage. Detect rather than
    # assume: a share above 1.5 can only be the percent form.
    top = s.select(pl.col("offense_pct").cast(pl.Float64).max()).item() if s.height else None
    if top is not None and float(top) > 1.5:
        pct = pct / 100.0

    xw = (crosswalk.select(pl.col("pfr_id").cast(pl.Utf8),
                           pl.col("gsis_id").cast(pl.Utf8).alias("player_id"))
          .drop_nulls().unique(subset=["pfr_id"]))
    s = (s.with_columns(pct.alias("_pct"), pl.col("week").cast(pl.Float64).alias("_wk"))
         .join(xw, left_on=pl.col("pfr_player_id").cast(pl.Utf8),
               right_on="pfr_id", how="inner"))
    return s.group_by(["season", "player_id"]).agg(
        pl.col("_pct").mean().alias("snap_pct"),
        _slope("_wk", "_pct").alias("drift"))


def player_seasons(stats: pl.DataFrame, snaps: pl.DataFrame | None = None,
                   crosswalk: pl.DataFrame | None = None, *,
                   min_games: int = MIN_GAMES, min_ppg: float = MIN_PPG) -> pl.DataFrame:
    """One row per qualifying player-season: its realised spread, and the role that produced it."""
    need = {"season", "week", "player_id", "position", "fantasy_points_ppr"}
    missing = need - set(stats.columns)
    if missing:
        raise ValueError(f"player_seasons needs {sorted(missing)}")

    w = stats
    if "season_type" in w.columns:
        w = w.filter(pl.col("season_type") == "REG")
    w = w.filter(pl.col("position").is_in(POSITIONS))

    have = [c for c in _TD_COLS if c in w.columns]
    tds = (sum((pl.col(c).cast(pl.Float64).fill_null(0.0) for c in have),
               start=pl.lit(0.0)) if have else pl.lit(0.0))
    out = w.with_columns(tds.alias("_td"),
                         _opt(w, "target_share").alias("_tgt"),
                         _opt(w, "air_yards_share").alias("_ay")).group_by(
        ["season", "player_id", "position"]).agg(
        pl.len().alias("games"),
        pl.col("fantasy_points_ppr").mean().alias("mu"),
        pl.col("fantasy_points_ppr").std(ddof=1).alias("sd"),
        pl.col("_tgt").mean().alias("tgt_share"),
        pl.col("_ay").mean().alias("ay_share"),
        pl.col("_td").mean().alias("td_pg"),
    ).filter((pl.col("games") >= min_games) & (pl.col("mu") > min_ppg)
             & pl.col("sd").is_not_null() & (pl.col("sd") > 0))

    if snaps is not None and crosswalk is not None:
        out = out.join(snap_usage(snaps, crosswalk), on=["season", "player_id"], how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("snap_pct"),
                               pl.lit(None, pl.Float64).alias("drift"))

    return out.with_columns(
        (pl.col("sd") / pl.col("mu").sqrt()).alias("k"),
        pl.col("mu").log().alias("log_mu"),
    ).sort(["season", "player_id"])


def pairs(seasons: pl.DataFrame) -> pl.DataFrame:
    """Prior-season role joined to the next season's realised spread.

    The outcome is `sd_next`; every feature carries the `_prev` suffix. Nothing from the
    outcome season is available to any candidate except `mu_next`, which both arms share.
    """
    prev = seasons.select(
        pl.col("season") + 1, "player_id", "position",
        *[pl.col(c).alias(f"{c}_prev") for c in ("k", "mu", *FEATURES)])
    nxt = seasons.select("season", "player_id",
                         pl.col("mu").alias("mu_next"), pl.col("sd").alias("sd_next"))
    return prev.join(nxt, on=["season", "player_id"], how="inner").sort(
        ["season", "player_id"])


def fit_shrinkage(train: pl.DataFrame) -> float:
    """How much of a player's own prior `k` to keep, on a grid, chosen on training rows only.

    Shrinkage happens in logs because `k` is a positive scale parameter -- averaging 0.5 and
    2.0 to 1.25 rather than 1.0 would bias every prediction upward.
    """
    if train.is_empty():
        return 0.0
    grid = np.linspace(0.0, 1.0, 21)
    best, best_mae = 0.0, np.inf
    for w in grid:
        mae = float(np.abs(_predict(train, "own_k", w=float(w))
                           - train["sd_next"].to_numpy()).mean())
        if mae < best_mae:
            best, best_mae = float(w), mae
    return best


def fit_usage(train: pl.DataFrame) -> dict[str, float]:
    """OLS of log(k_next) on prior-season role, with the positional constant as the offset.

    Fitting the *residual* from `positional` rather than log k itself means a zero
    coefficient vector reproduces the shipped model exactly, so the usage arm can only be
    adopted by earning it.
    """
    if train.height < 50:
        return {}
    x, mean = _design(train, None)
    base = train.select(_positional_k(pl.col("position")))[:, 0].to_numpy().astype(float)
    y = np.log(train["k_next_implied"].to_numpy().astype(float)) - np.log(base)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    out = {"_intercept": float(coef[0])}
    out.update({f: float(c) for f, c in zip(FEATURES, coef[1:], strict=True)})
    out.update({f"_mean_{f}": v for f, v in mean.items()})
    return out


def _design(df: pl.DataFrame, mean: dict[str, float] | None) -> tuple[np.ndarray, dict[str, float]]:
    """Feature matrix with an intercept, nulls filled from the *training* mean.

    Filling a held-out null with a held-out mean would leak the evaluation set into the
    features, so the fitted means travel with the coefficients.
    """
    cols, out = [], {}
    for f in FEATURES:
        v = df[f"{f}_prev"].cast(pl.Float64).to_numpy().astype(float)
        finite = np.isfinite(v)
        # An all-null feature column (no snap-count match in this slice) has no mean at
        # all; nanmean would warn and return nan, and nan in the design matrix silently
        # destroys every coefficient, not just this one.
        m = (float(v[finite].mean()) if mean is None and finite.any()
             else (mean or {}).get(f, 0.0))
        if not np.isfinite(m):
            m = 0.0
        out[f] = m
        cols.append(np.where(finite, v, m))
    return np.column_stack([np.ones(df.height), *cols]), out


def _predict(df: pl.DataFrame, candidate: str, *, w: float = 0.0,
             coef: dict[str, float] | None = None) -> np.ndarray:
    """Predicted `sd` for the outcome season. Every arm uses the same `mu_next`."""
    mu = df["mu_next"].to_numpy().astype(float)
    base = df.select(_positional_k(pl.col("position")))[:, 0].to_numpy().astype(float)
    if candidate == "positional":
        k = base
    elif candidate == "own_k":
        own = df["k_prev"].to_numpy().astype(float)
        own = np.where(np.isfinite(own) & (own > 0), own, base)
        k = np.exp(w * np.log(own) + (1.0 - w) * np.log(base))
    elif candidate == "usage":
        if not coef:
            k = base
        else:
            mean = {f: coef.get(f"_mean_{f}", 0.0) for f in FEATURES}
            x, _ = _design(df, mean)
            beta = np.array([coef.get("_intercept", 0.0)]
                            + [coef.get(f, 0.0) for f in FEATURES])
            # No exp(sigma^2/2) correction: MAE is minimised by the median, and the median
            # of a lognormal is exp of the median in logs. Correcting would bias it.
            k = base * np.exp(x @ beta)
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return k * np.sqrt(np.clip(mu, 0.0, None))


def walk_forward(pr: pl.DataFrame) -> pl.DataFrame:
    """One row per held-out observation, carrying each candidate's absolute error.

    Returning the errors rather than a table of means is what lets `verdict` run a *paired*
    test. The same player-season is scored by every candidate, so the comparison is not
    contaminated by which players happened to land in which season -- and the standard
    error is of the difference, which is far tighter than the error of either arm.
    """
    frames = []
    for season in sorted(pr["season"].unique().to_list()):
        train = pr.filter(pl.col("season") < season)
        test = pr.filter(pl.col("season") == season)
        if train.is_empty() or test.is_empty():
            continue
        train = train.with_columns(
            (pl.col("sd_next") / pl.col("mu_next").sqrt()).alias("k_next_implied"))
        w, coef = fit_shrinkage(train), fit_usage(train)
        actual = test["sd_next"].to_numpy().astype(float)
        frames.append(pl.DataFrame(
            {"season": [season] * test.height, "w": [w] * test.height,
             **{f"err_{c}": np.abs(_predict(test, c, w=w, coef=coef) - actual)
                for c in CANDIDATES}}))
    return pl.concat(frames) if frames else pl.DataFrame()


def summarise(errs: pl.DataFrame) -> pl.DataFrame:
    """Per-season mean absolute error, for reading. The gate uses `errs` itself."""
    if errs.is_empty():
        return errs
    return errs.group_by("season").agg(
        pl.len().alias("n"), pl.col("w").first(),
        *[pl.col(f"err_{c}").mean().alias(f"mae_{c}") for c in CANDIDATES],
    ).sort("season")


def verdict(errs: pl.DataFrame) -> tuple[str, str]:
    """The pre-registered rule, both halves of it.

    A candidate is adopted only if it beats `positional` in **every** held-out season *and*
    the paired difference clears `MIN_SE` standard errors. The first half stops a fit being
    adopted on one lucky year; the second stops one being adopted on a gain too small to
    distinguish from noise, which is exactly what the first version of this function -- which
    checked only the seasons -- would have done.
    """
    if errs.is_empty() or "err_positional" not in errs.columns:
        return "positional", "nothing measured -- no held-out season"
    per = summarise(errs)
    base = errs["err_positional"].to_numpy().astype(float)
    seasons = per.height
    lines, winner, best = [], "positional", 0.0
    for c in CANDIDATES:
        if c == "positional":
            continue
        d = base - errs[f"err_{c}"].to_numpy().astype(float)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        t = float(d.mean() / se) if se > 0 else 0.0
        wins = int((per[f"mae_{c}"].to_numpy() < per["mae_positional"].to_numpy()).sum())
        lines.append(f"  {c}: mean gain {d.mean():+.4f} MAE at {t:.1f} se, "
                     f"wins {wins}/{seasons} seasons")
        if wins == seasons and t >= MIN_SE and d.mean() > best:
            winner, best = c, float(d.mean())
    body = "\n".join(lines)
    if winner == "positional":
        return winner, ("KEEP 'positional': no candidate cleared both halves of the gate "
                        f"({MIN_SE:.0f} se, and every held-out season).\n{body}")
    return winner, (f"ADOPT '{winner}': beats the shipped positional constant by "
                    f"{best:.4f} MAE in every held-out season.\n{body}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hub.models.spread", description=__doc__)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--seasons", default="2022,2023,2024,2025")
    p.add_argument("--out")
    a = p.parse_args(list(argv) if argv is not None else None)
    if not a.fit:
        p.print_help()
        return 0

    import nflreadpy as nfl
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    stats = nfl.load_player_stats(seasons=seasons)
    try:
        snaps, xw = nfl.load_snap_counts(seasons=seasons), nfl.load_ff_playerids()
    except Exception as exc:
        print(f"snap counts unavailable ({exc}); usage arm runs without snap share")
        snaps, xw = None, None

    ps = player_seasons(stats, snaps, xw)
    pr = pairs(ps)
    print(f"{ps.height} qualifying player-seasons, {pr.height} consecutive-season pairs")
    matched = ps.filter(pl.col("snap_pct").is_not_null()).height
    print(f"snap share matched on {matched}/{ps.height} "
          f"({100.0 * matched / max(ps.height, 1):.1f}%)")

    errs = walk_forward(pr)
    if errs.is_empty():
        print("no held-out season")
        return 0
    per = summarise(errs)
    print("\nHeld-out MAE of predicted weekly sd, fitting only on earlier seasons:")
    print(per.select("season", "n", *[f"mae_{c}" for c in CANDIDATES]))
    print(f"\n{verdict(errs)[1]}")

    if a.out:
        per.write_parquet(a.out)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
