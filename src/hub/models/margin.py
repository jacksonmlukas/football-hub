"""Fit the dispersion of game margin around the closing spread.

`hub.models.market` sets `MARGIN_SD = 13.5` under a comment reading "stable across decades of
NFL results". There was no fit, no interval and no write-up — and this is the number that turns
every closing spread into a win probability, so it is the most load-bearing constant in the NFL
path. It also sits in `config.FITTED_MODULES`, hashed into the model version *as though* it had
been fitted, which is ADR-0006's distinction landing on the wrong side.

The data to measure it has been in the fetch layer the whole time: `schedules` carries
`spread_line` and `result` back to 1999. `result` is home-relative (verified: it equals
`home_score - away_score` for every completed game), and `spread_line` is too, so the residual
is `result - spread_line` and needs no sign juggling.

**The dispersion is not constant, which is the interesting part.** Per-season sd runs from 11.5
to 14.4, and the trend is downward at -0.037 a year (-2.4 se) — the market has got sharper. So
"which window" is a real modelling choice, not a detail, and this module tests three candidates
rather than assuming one.

**The gate, fixed before any log-loss was computed.** A candidate must beat 13.5 on *held-out*
log-loss, walking forward one season at a time, fitting only on strictly earlier seasons. If
none does, 13.5 stays — and an asserted number that survives a fit is no longer asserted, which
is a result worth having.

    uv run python -m hub.models.margin --fit

**The shape, measured 2026-09-11 (issue #185).** Football margins are lumpy -- 15.0% of games
over the trailing ten seasons end on exactly 3, 2.75 times what a normal puts there -- and a
smooth bell has no notion of that. This module also asks
whether that lumpiness moves the number the repo actually consumes, which is `P(margin > 0)`:
it computes the ceiling first (rule 8), reproduces the histogram from our own sample, layers
fitted bumps at the key numbers onto the unchanged spine, and walks the result forward against
the plain Gaussian on held-out log-loss. **The Gaussian stays**; the section beginning
`KEY_NUMBERS` records why, and `docs/margin-sd.md` carries the tables.

    uv run python -m hub.models.margin --shape
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import erf, sqrt

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.models.experiment import expanding_seasons

# The incumbent. Imported rather than restated so the two cannot drift apart.
from hub.models.market import MARGIN_SD
from hub.models.scoring_rules import log_loss

NOT_FITTED_BECAUSE = (
    "the recorded output of the MARGIN_SD fit, used by a test to guard the live constant -- "
    "an assertion about a prediction, not an input. The live number lives in "
    "hub.models.market, which IS registered. "
)

# What the 2026-08-24 fit found, kept so a test can guard the live constant against it -- the
# same pattern `hub.draft.calibrate.FITTED_CI95` uses for TALENT_CV. A refit updates both.
FITTED_SD = 12.741
FITTED_SE = 0.164
FITTED_N = 3018
FITTED_WINDOW = "trailing 10 seasons, as of 2025"

# Candidate windows, named before the numbers.
#
# `all` is every season on record; `trailing10` reflects the measured downward trend, on the
# view that a decade-old market is a different market. Both are compared against the incumbent,
# and the incumbent wins ties -- an equal-performing new number is not an improvement, it is
# churn in a constant that is hashed into every model version.
CANDIDATES = ("incumbent", "all", "trailing10")
TRAILING = 10

# A tie is neither a home win nor an away win, and there is no sensible probability to score it
# against. 1999-2025 has a handful; dropping them is cleaner than inventing a convention.
#
# **The repo's one answer, and `home_won` below is where it is read.** Until issue #64 this
# constant was quoted in `hub.models.eval`'s docstring and read by nothing outside this
# module: the comparison hardcoded the same test, and `hub.publish._scored` -- the path that
# feeds the public record -- took the other convention, deriving the outcome as `result > 0`
# and so scoring a tie as a home loss. That is log-loss credit for a game nobody won, handed
# to whichever model gave the home side the lower probability. Flipping this to False changed
# no behaviour and broke no test, which is the same defect the excision work exists to
# eliminate. Everything that scores an outcome now goes through `home_won`.
DROP_TIES = True


def home_won(games: pl.DataFrame) -> pl.DataFrame:
    """`games` with unscorable rows dropped and the realised outcome added as `home_won`.

    The one place a realised margin becomes the binary outcome a proper scoring rule reads,
    so that `DROP_TIES` is read rather than restated. `result` is home score minus away, so
    the outcome is its sign and needs no team columns.

    Two kinds of row cannot be scored and both go, for the same reason: log loss cannot tell
    an invented outcome from an observed one. An **unplayed** game has no result at all, and
    a **tie** has one that answers a question nobody asked -- the prediction was P(home win),
    and a game nobody won is not a home loss.

    Dropping is a real cost and is worth naming: a tied game the model priced is a game the
    record does not count. It is the smaller cost, and cheap -- issue #64 puts NFL ties at
    about one every season or two, against a ~285-game season. Scoring one as a home loss
    would put a confident wrong outcome into a calibration bin instead, and calibration is
    the page's whole claim (`docs/track-record.md` rule 3).
    """
    out = games.drop_nulls("result")
    if DROP_TIES:
        out = out.filter(pl.col("result") != 0)
    return out.with_columns((pl.col("result") > 0).cast(pl.Int64).alias("home_won"))


def residuals(schedules: pl.DataFrame) -> pl.DataFrame:
    """(season, spread_line, result, resid, home_won) per completed game with a spread."""
    keep = ("season", "spread_line", "result")
    if not set(keep) <= set(schedules.columns):
        missing = sorted(set(keep) - set(schedules.columns))
        raise ValueError(f"schedules is missing {missing}")
    out = (schedules.select(keep)
                    .drop_nulls("spread_line")
                    .with_columns((pl.col("result") - pl.col("spread_line")).alias("resid")))
    return home_won(out)


def fit(resid: pl.DataFrame) -> dict[str, float]:
    """Sample dispersion of the residual, with a standard error.

    `se(sd) ≈ sd / sqrt(2(n-1))` — the large-sample standard error of a standard deviation.
    Reported because a point estimate of a dispersion invites being compared to another point
    estimate, and the whole question here is whether 13.5 is far enough away to matter.
    """
    r = resid["resid"].to_numpy().astype(float)
    n = r.size
    if n < 2:
        return {"n": float(n), "sd": float("nan"), "se": float("nan"), "mean": float("nan")}
    sd = float(r.std(ddof=1))
    return {"n": float(n), "sd": sd, "se": sd / sqrt(2 * (n - 1)), "mean": float(r.mean())}


def home_win_prob(spread: np.ndarray, sd: float) -> np.ndarray:
    """P(home team wins) given a home-relative closing spread and a margin dispersion."""
    z = np.asarray(spread, dtype=float) / (sd * sqrt(2.0))
    return 0.5 * (1.0 + np.array([erf(v) for v in z]))


# --- the shape: mass on the key numbers (#185) ----------------------------------------------
#
# The numbers football margins land on. 3 and 7 are a field goal and a touchdown; 6, 10 and 14
# are the combinations a one-score and two-score game resolve to. Named before the histogram
# was drawn, as issue #185's amendment of 2026-09-07 named them.
KEY_NUMBERS: tuple[int, ...] = (3, 6, 7, 10, 14)

# What the 2026-09-11 measurement found, kept so a test can guard the live shape against it
# -- the pattern `FITTED_SD` uses two screens up. Every number here is an *output* of a fit
# used by a test, not an input to a prediction: the prediction is still `home_win_prob`
# with the plain Gaussian, and that is the finding.
#
# `FITTED_KEY_EXCESS[k]` is (share of games whose |margin| is exactly k) / (the share the
# Gaussian spine puts in (k - 1/2, k + 1/2]) - 1, over the spine's own window. The excess is
# real: a margin of 3 happens 2.75 times as often as the spine says. It is also nearly
# symmetric about the spread -- a favourite wins by 3 at 2.9x the spine and loses by 3 at
# 2.5x -- and every key number sits within 14 points of the centre, so the mass a bump adds
# lands on *both* sides of zero and pulls every favourite toward one half. The data pull the
# other way: 7-point-or-better favourites win 80.3% against a 77.1% price (#176). Lumpiness
# therefore prices `P(margin > 0)` worse than the smooth spine does, held out, in 20 of 27
# seasons, and the ceiling says what the calibration miss is instead: a *location* that moves
# with the spread, which no shape symmetric about the spread can reach.
FITTED_KEY_EXCESS: dict[int, float] = {3: 1.751, 6: 0.296, 7: 0.671, 10: 0.121, 14: 0.499}
FITTED_SHAPE_GAIN = -0.00094       # mean held-out log-loss, lumpy over gaussian, 27 seasons
FITTED_SHAPE_SE = 0.00032          # se of that mean across seasons; negative at -2.9 se
FITTED_SHAPE_SEASONS_BETTER = 7    # of 27, one of them 2026 at two games
FITTED_SHAPE_CEILING = 0.0056      # in-sample gain of a perfect P(win | spread), per game
FITTED_SHAPE_WINDOW = "trailing 10 seasons as of 2026-09-11 (2017-2026), spine at MARGIN_SD"


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised the way `home_win_prob` is (no scipy in the tree)."""
    z = np.asarray(x, dtype=float) / sqrt(2.0)
    return 0.5 * (1.0 + np.array([erf(v) for v in np.atleast_1d(z)]))


def _spine_cell(spread: np.ndarray, k: float, sd: float) -> np.ndarray:
    """The spine's mass in (k - 1/2, k + 1/2] for a margin centred on `spread` -- the share a
    Gaussian puts on the integer k, which is what an empirical share is compared against."""
    s = np.asarray(spread, dtype=float)
    return _phi((k + 0.5 - s) / sd) - _phi((k - 0.5 - s) / sd)


def key_number_mass(resid: pl.DataFrame, sd: float = MARGIN_SD,
                    keys: Sequence[int] = KEY_NUMBERS) -> pl.DataFrame:
    """The histogram at the key numbers, from our own sample rather than a published figure.

    One row per key number: the share of games whose |margin| is exactly `key`, the share the
    spine centred on each game's spread would put there, and `excess = empirical / spine - 1`.
    Pooled over both signs on purpose -- the bump model below is symmetric about the spread,
    and the comment on `FITTED_KEY_EXCESS` says what that symmetry costs.
    """
    m = np.abs(resid["result"].to_numpy().astype(float))
    s = resid["spread_line"].to_numpy().astype(float)
    rows = []
    for k in keys:
        emp = float(np.mean(m == k)) if m.size else float("nan")
        spine = (float(np.mean(_spine_cell(s, k, sd) + _spine_cell(s, -k, sd)))
                 if s.size else float("nan"))
        rows.append({"key": int(k), "empirical": emp, "spine": spine,
                     "excess": (emp / spine - 1.0) if spine else float("nan")})
    return pl.DataFrame(rows)


def key_excess(resid: pl.DataFrame, sd: float = MARGIN_SD,
               keys: Sequence[int] = KEY_NUMBERS) -> dict[int, float]:
    """`key_number_mass` as the mapping `lumpy_home_win_prob` takes: the fitted bump weights."""
    t = key_number_mass(resid, sd, keys)
    return {int(k): float(e) for k, e in zip(t["key"].to_list(), t["excess"].to_list(),
                                            strict=True)}


def lumpy_home_win_prob(spread: np.ndarray, sd: float,
                        excess: Mapping[int, float]) -> np.ndarray:
    """P(home wins) under the spine with a multiplicative bump of `1 + excess[k]` on the
    spine's mass at +k and at -k, renormalised. The spine is unchanged; with no bumps this is
    `home_win_prob` exactly.

    Closed form, because every term is a Gaussian cell: the win side is the spine's mass above
    zero plus the bumped cells at the positive key numbers, over the total mass after bumping
    both signs. A margin of exactly zero is a tie and is outside both `home_won` and this.
    """
    s = np.asarray(spread, dtype=float)
    win = 1.0 - _phi(-s / sd)
    total = np.ones_like(win)
    for k, b in excess.items():
        up, down = _spine_cell(s, k, sd), _spine_cell(s, -k, sd)
        win = win + b * up
        total = total + b * (up + down)
    return win / total


def ceiling(resid: pl.DataFrame, sd: float = MARGIN_SD) -> dict[str, float]:
    """Rule 8: how much *any* margin distribution could gain on the number the repo consumes.

    `P(margin > 0 | spread)` is all that `survivor` and `MarketBaseline` read, so a perfect
    margin distribution is worth exactly what a perfect `P(win | spread)` is worth and no
    more. The oracle is the favourite's realised win rate in one-point buckets of |spread|,
    scored **in sample** -- an upper bound, flattered by construction, which is what a ceiling
    should be. A shape that closes a small fraction of this is chasing the gap below it.
    """
    s = resid["spread_line"].to_numpy().astype(float)
    won = resid["home_won"].to_numpy().astype(float)
    if s.size == 0:
        return {"n": 0.0, "ll_gaussian": float("nan"), "ll_oracle": float("nan"),
                "gain": float("nan")}
    fav = np.where(s > 0, won, 1.0 - won)
    bucket = np.rint(np.abs(s))
    rate = {float(b): float(fav[bucket == b].mean()) for b in np.unique(bucket)}
    oracle = np.array([0.5 if v == 0 else rate[float(b)] if v > 0 else 1.0 - rate[float(b)]
                       for v, b in zip(s, bucket, strict=True)])
    ll_g = log_loss(home_win_prob(s, sd), won)
    ll_o = log_loss(np.clip(oracle, 1e-6, 1.0 - 1e-6), won)
    return {"n": float(s.size), "ll_gaussian": ll_g, "ll_oracle": ll_o, "gain": ll_g - ll_o}


def walk_forward_shape(resid: pl.DataFrame, *, sd: float = MARGIN_SD,
                       trailing: int = TRAILING) -> pl.DataFrame:
    """Held-out log-loss per season, plain Gaussian against the spine with fitted bumps.

    The spine is held at `sd` in both arms -- the question is the shape, not the width, and
    `walk_forward` above already settled the width. The bumps are refitted each season on the
    trailing window of strictly earlier seasons, never the one being scored. `min_past=2`
    for `expanding_seasons`'s stated reason.
    """
    rows = []
    for yr, past, now in expanding_seasons(resid, min_past=2):
        bumps = key_excess(past.filter(pl.col("season") >= yr - trailing), sd)
        spread = now["spread_line"].to_numpy().astype(float)
        won = now["home_won"].to_numpy().astype(float)
        ll_g = log_loss(home_win_prob(spread, sd), won)
        ll_l = log_loss(lumpy_home_win_prob(spread, sd, bumps), won)
        rows.append({"season": yr, "n": now.height, "ll_gaussian": ll_g, "ll_lumpy": ll_l,
                     "gain": ll_g - ll_l})
    return pl.DataFrame(rows)


def shape_verdict(wf: pl.DataFrame) -> tuple[str, str]:
    """The pre-registered rule, fixed before the walk-forward ran. Returns (shape, sentence).

    The lumpy distribution must beat the Gaussian on **mean held-out log-loss**. A tie or a
    loss keeps the Gaussian: `docs/margin-sd.md` recorded that the width gate fired on a sign
    alone and said a future gate of this shape should ask for more, so the sentence carries
    the mean, its standard error across seasons and the season count either way -- and the
    rule still selects on the mean, because changing the rule after the number is in is the
    failure this repo has caught twice.
    """
    if wf.is_empty():
        return "gaussian", "no held-out seasons; the Gaussian stands by default."
    g = wf["gain"].to_numpy().astype(float)
    mean = float(g.mean())
    se = float(g.std(ddof=1) / sqrt(g.size)) if g.size > 1 else float("nan")
    better = int((g > 0).sum())
    detail = (f"mean held-out log-loss gain {mean:+.5f} (se {se:.5f}), lumpy better in "
              f"{better}/{g.size} seasons")
    if mean > 0:
        return "lumpy", f"ADOPT the key-number shape: {detail}."
    return "gaussian", (f"KEEP the Gaussian: {detail}. Mass on the key numbers is real and "
                        f"does not price P(margin > 0) better than the spine alone.")


def calibration_by_spread(resid: pl.DataFrame, *, sd: float = MARGIN_SD,
                          trailing: int = TRAILING,
                          since: int | None = None) -> pl.DataFrame:
    """Win probability by spread bucket, both models beside the realised rate, **held out**.

    Both sides of every game go in, as `coverage.survivor_price` grades and as
    `survivor.grid_from_schedule` builds. The lumpy price for each season is fitted on the
    trailing window of earlier seasons only, so this is the calibration a pick rule would
    actually have walked into, not the in-sample one. The buckets and the survivor threshold
    are read from `hub.models.coverage`, the module that owns that grading -- imported inside
    the function because the grader imports this module the same way, and a third spelling of
    those edges is how the two tables would disagree.

    One row per bucket plus a `favourites` row at the survivor threshold. Columns: `bucket`,
    `n`, `gaussian`, `lumpy`, `actual`.
    """
    from hub.models.coverage import SPREAD_EDGES, SURVIVOR_SPREAD

    parts = []
    for yr, past, now in expanding_seasons(resid, min_past=2):
        if since is not None and yr < since:
            continue
        bumps = key_excess(past.filter(pl.col("season") >= yr - trailing), sd)
        s = now["spread_line"].to_numpy().astype(float)
        won = now["home_won"].to_numpy().astype(float)
        pg, pli = home_win_prob(s, sd), lumpy_home_win_prob(s, sd, bumps)
        parts.append(pl.DataFrame({"spread": np.concatenate([s, -s]),
                                   "gaussian": np.concatenate([pg, 1.0 - pg]),
                                   "lumpy": np.concatenate([pli, 1.0 - pli]),
                                   "won": np.concatenate([won, 1.0 - won])}))
    if not parts:
        return pl.DataFrame(schema={"bucket": pl.Utf8, "n": pl.Int64, "gaussian": pl.Float64,
                                    "lumpy": pl.Float64, "actual": pl.Float64})
    sides = pl.concat(parts)
    rows = []
    for lo, hi in pairwise(SPREAD_EDGES):
        q = sides.filter((pl.col("spread") >= lo) & (pl.col("spread") < hi))
        rows.append((f"[{lo:g}, {hi:g})", q))
    rows.append(("favourites", sides.filter(pl.col("spread") >= SURVIVOR_SPREAD)))
    return pl.DataFrame({
        "bucket": [b for b, _ in rows],
        "n": [q.height for _, q in rows],
        "gaussian": [_mean(q, "gaussian") for _, q in rows],
        "lumpy": [_mean(q, "lumpy") for _, q in rows],
        "actual": [_mean(q, "won") for _, q in rows],
    })


def survival_beside(calibration: pl.DataFrame, *, picks: int | None = None) -> dict[str, float]:
    """Season-long survival for a plan of favourites, under each model and as realised.

    The product `survivor` prints is a chain of these probabilities, one a week, so the
    honest side-by-side is the favourites row raised to the season's length: what a plan of
    `picks` such favourites survives at under the Gaussian, under the lumpy price, and at
    the realised rate. `picks` defaults to `hub.season.survivor.NFL_WEEKS`, read from the
    module that owns the season's length rather than restated; imported inside because the
    product's module is downstream of this one.

    **It is a bound, not a measurement** (#286). Raising one bucket's rate to the
    eighteenth power asserts that every week is priced at that rate and that the weeks are
    independent; a real plan's favourites are priced differently week to week and share a
    season. No interval is carried because none would be honest -- the sampling error on
    the rate is the smaller of its two errors, and the other is the assumption. What is
    carried is `n`, the games the rate rests on, and `survival_line` is the one place the
    figure is printed, so it is labelled wherever it is read.
    """
    if picks is None:
        from hub.season.survivor import NFL_WEEKS
        picks = NFL_WEEKS
    fav = calibration.filter(pl.col("bucket") == "favourites")
    if fav.is_empty():
        return {"picks": float(picks), "n": 0.0, "gaussian": float("nan"),
                "lumpy": float("nan"), "actual": float("nan")}
    return {"picks": float(picks), "n": float(fav["n"][0]),
            **{c: float(fav[c][0]) ** picks for c in ("gaussian", "lumpy", "actual")}}


def survival_line(surv: dict[str, float]) -> str:
    """`survival_beside` as the one line it is printed on, labelled as the bound it is."""
    return (f"Survival over {int(surv['picks'])} such favourites, as the independence bound "
            f"-- one rate every week, weeks independent, off {int(surv['n'])} games -- and "
            f"not a measurement: gaussian {surv['gaussian']:.4f}  lumpy {surv['lumpy']:.4f}  "
            f"realised {surv['actual']:.4f}")




def walk_forward(resid: pl.DataFrame, *, trailing: int = TRAILING) -> pl.DataFrame:
    """Held-out log-loss per season for each candidate, fitting only on earlier seasons.

    Expanding window, one season at a time, never peeking. The first season with any history
    is the first that can be scored, so the earliest season on record is used only for fitting.
    `min_past=2` because `fit` needs two residuals before it has a standard deviation.
    """
    rows = []
    for yr, past, now in expanding_seasons(resid, min_past=2):
        sds = {
            "incumbent": MARGIN_SD,
            "all": fit(past)["sd"],
            "trailing10": fit(past.filter(pl.col("season") >= yr - trailing))["sd"],
        }
        spread = now["spread_line"].to_numpy().astype(float)
        # Off the column `residuals` built, not re-derived here. This read `result > 0`, a
        # fourth restatement of the outcome convention in a module that declares it.
        won = now["home_won"].to_numpy().astype(float)
        row: dict[str, float] = {"season": yr, "n": now.height}
        for name, sd in sds.items():
            row[f"sd_{name}"] = sd
            row[f"ll_{name}"] = log_loss(home_win_prob(spread, sd), won)
        rows.append(row)
    return pl.DataFrame(rows)


def _mean(df: pl.DataFrame, col: str) -> float:
    """Column mean as a plain float. Polars types `.mean()` as a union including None."""
    v = df[col].mean()
    return float(v) if isinstance(v, (int, float)) else float("nan")


def verdict(wf: pl.DataFrame) -> tuple[str, str]:
    """The pre-registered rule. Returns (winning candidate, the sentence explaining it).

    A candidate must beat the incumbent on **mean held-out log-loss**. Ties go to the
    incumbent: replacing a constant that is hashed into every model version, for no measured
    gain, is churn rather than improvement.
    """
    if wf.is_empty():
        return "incumbent", "no held-out seasons; 13.5 stands by default."
    means = {c: _mean(wf, f"ll_{c}") for c in CANDIDATES}
    base = means["incumbent"]
    challengers = {c: m for c, m in means.items() if c != "incumbent"}
    best = min(challengers, key=lambda c: challengers[c])
    if challengers[best] < base:
        gain = base - challengers[best]
        return best, (f"ADOPT '{best}': mean held-out log-loss {challengers[best]:.5f} against "
                      f"{base:.5f} for MARGIN_SD={MARGIN_SD}, an improvement of {gain:.5f}.")
    return "incumbent", (f"KEEP {MARGIN_SD}: no candidate beat it on held-out log-loss "
                         f"(best challenger {challengers[best]:.5f} against {base:.5f}). "
                         f"An asserted number that survives a fit is no longer asserted.")


def _report_shape(resid: pl.DataFrame, *, trailing: int = TRAILING) -> None:
    """Print the shape measurement in the order rule 8 requires: the ceiling first."""
    latest = int(resid["season"].to_numpy().max()) if resid.height else 0
    window = resid.filter(pl.col("season") > latest - trailing)
    top = ceiling(window)
    print(f"\n  Ceiling, in sample over {int(top['n'])} games from the trailing {trailing} "
          f"seasons: a perfect P(win | spread) scores {top['ll_oracle']:.5f} against "
          f"{top['ll_gaussian']:.5f} for the Gaussian, a gain of {top['gain']:.5f} a game.")

    print(f"\n  Mass on the key numbers, same window, spine at MARGIN_SD={MARGIN_SD}:")
    print(f"  {'|margin|':>8} {'empirical':>10} {'spine':>8} {'excess':>8}")
    for r in key_number_mass(window).iter_rows(named=True):
        print(f"  {r['key']:>8} {r['empirical']:>10.4f} {r['spine']:>8.4f} {r['excess']:>+8.3f}")

    wf = walk_forward_shape(resid, trailing=trailing)
    print(f"\n  Walk-forward, {wf.height} held-out seasons, bumps fitted only on earlier ones, "
          f"spine unchanged.")
    print(f"  {'season':>6} {'n':>4} {'ll_gaussian':>12} {'ll_lumpy':>10} {'gain':>10}")
    for r in wf.tail(10).iter_rows(named=True):
        print(f"  {r['season']:>6} {r['n']:>4} {r['ll_gaussian']:>12.5f} "
              f"{r['ll_lumpy']:>10.5f} {r['gain']:>+10.5f}")

    cal = calibration_by_spread(resid, trailing=trailing, since=latest - trailing + 1)
    print(f"\n  Calibration by spread bucket, held out, both sides, seasons "
          f"{latest - trailing + 1}-{latest}:")
    print(f"  {'bucket':>12} {'n':>6} {'gaussian':>9} {'lumpy':>8} {'actual':>8}")
    for r in cal.iter_rows(named=True):
        print(f"  {r['bucket']:>12} {r['n']:>6} {r['gaussian']:>9.3f} {r['lumpy']:>8.3f} "
              f"{r['actual']:>8.3f}")
    print(f"\n  {survival_line(survival_beside(cal))}")

    _, sentence = shape_verdict(wf)
    print(f"\n  {sentence}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.margin",
        description="Fit the margin dispersion around the closing spread, and gate it.")
    ap.add_argument("--fit", action="store_true", help="run the walk-forward and report")
    ap.add_argument("--shape", action="store_true",
                    help="the key-number shape: ceiling, histogram, walk-forward, calibration")
    ap.add_argument("--trailing", type=int, default=TRAILING)
    ap.add_argument("--out", default=None, help="write the per-season frame to this parquet")
    a = ap.parse_args(argv)
    if not (a.fit or a.shape):
        ap.print_help()
        return 0

    import nflreadpy as nfl

    print("  loading schedules ...")
    try:
        resid = residuals(nfl.load_schedules())
    except Exception as e:
        return unavailable("hub.models.margin", "nflverse schedules", e)
    if a.shape:
        _report_shape(resid, trailing=a.trailing)
    if not a.fit:
        return 0
    whole = fit(resid)
    print(f"\n  full sample: n={int(whole['n'])}  sd={whole['sd']:.3f} +/-{whole['se']:.3f}  "
          f"mean residual {whole['mean']:+.3f}")
    print(f"  incumbent MARGIN_SD={MARGIN_SD} is "
          f"{(MARGIN_SD - whole['sd']) / whole['se']:+.1f} se from the full-sample fit")

    wf = walk_forward(resid, trailing=a.trailing)
    print(f"\n  Walk-forward, {wf.height} held-out seasons, fitting only on earlier ones.")
    print(f"  {'season':>6} {'n':>4}  " + "  ".join(f"{'ll_' + c:>13}" for c in CANDIDATES))
    for r in wf.tail(10).iter_rows(named=True):
        print(f"  {r['season']:>6} {r['n']:>4}  "
              + "  ".join(f"{r['ll_' + c]:>13.5f}" for c in CANDIDATES))
    print(f"  {'mean':>6} {'':>4}  "
          + "  ".join(f"{_mean(wf, 'll_' + c):>13.5f}" for c in CANDIDATES))

    winner, sentence = verdict(wf)
    print(f"\n  {sentence}")
    if winner != "incumbent":
        final = fit(resid if winner == "all" else
                    resid.filter(pl.col("season") >= resid["season"].max() - a.trailing))
        print(f"  Value to adopt: {final['sd']:.3f} +/-{final['se']:.3f} (n={int(final['n'])})")
    if a.out:
        wf.write_parquet(a.out)
        print(f"  wrote {wf.height} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
