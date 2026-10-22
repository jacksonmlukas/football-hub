"""A state-space team rating, fitted and left unwired (#365, S8).

**This is an exhibit, and nothing reads it.** `hub.models.ratings.forecaster()` still returns
`MarketBaseline()`; no module outside `hub.exhibits` imports this one, and the weekly forward
arm's import closure does not contain it. Every constant below is `not_an_input`, so the config
digest stamped on every published prediction does not move (ADR-0006: a fitted constant lives
with its provenance, and one no prediction reads is not part of one). The pre-registration is
`docs/gate-power.md`, "Pre-registered 2026-09-21 -- the state-space team rating (#365)" and its
"Fit-stage specification (stage 1)", committed before this file existed.

**The model.** Glickman-Stern in kind: a Kalman filter over team strengths on the game margin.

    result[g] = theta[home] - theta[away] + HOME_ADV + e,     e ~ N(0, RATING_MARGIN_SD^2)
    within a season:   each team's variance grows by WEEK_STEP_SD^2 per elapsed week
    season boundary:   theta <- (1 - SEASON_SHRINK) theta
                       P     <- (1 - SEASON_SHRINK)^2 P + SEASON_SHRINK (2 - SEASON_SHRINK) tau^2 I

The full covariance is kept (two teams that met share an error), the boundary is the AR(1) whose
stationary variance is the prior `INITIAL_STRENGTH_VAR`, and a team first seen starts at the
prior. The filter reads **results only**: never `spread_line`, which would make Arm B partly
Arm A (`docs/method.md` rule 6).

**Walk-forward (rule 2).** For held-out season `S`, `(WEEK_STEP_SD, SEASON_SHRINK, HOME_ADV,
RATING_MARGIN_SD)` is the maximiser of the filter's one-step-ahead predictive log-likelihood over
seasons strictly before `S`, with `hub.models.experiment.expanding_seasons` drawing the line. The
filter then runs through `S` with those constants and only `S` is scored; a game in week `w` is
predicted from weeks `< w`. Flipping a later season's results therefore moves no earlier fold.

**What it records.** Per season, the walk-forward log-loss of the close minus the rating's, in
nats per game, with no verdict word. `gap_trailing` repeats it with the close's sigma taken from
the strictly earlier ten seasons, because the shipped `MARGIN_SD` was fitted on 2015-2025
(eleven seasons, #523) and so has seen the seasons it scores (a disclosure beside the
pre-registered column, deciding nothing).

**The gate** is declared (`HARNESS`) and, run on this archive, prints NOT-RUNNABLE: its ceiling
is the betting market's last pre-kickoff poll and the snapshot archive holds one season (#383),
against ADR-0019's minimum of three.

    uv run python -m hub.exhibits.state_space_rating --fit
    uv run python -m hub.exhibits.state_space_rating --gate
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import polars as pl

from hub.declare import not_an_input
from hub.ledger import recipe
from hub.models.experiment import Actions, Harness, expanding_seasons
from hub.models.market import MARGIN_SD, normal_cdf

# The start of the walk-forward optimiser, the same every fold; the value in force for a
# season is the fold's own fit, reported in the table, never one of these. Every reason is a
# literal because `hub.declare` reads it off the source.
WEEK_STEP_SD = not_an_input(
    0.5, "start value of the per-fold optimiser, an unwired exhibit constant (#365)")
SEASON_SHRINK = not_an_input(
    0.3, "start value of the per-fold optimiser, an unwired exhibit constant (#365)")
HOME_ADV = not_an_input(
    2.0, "start value of the per-fold optimiser, an unwired exhibit constant (#365)")
RATING_MARGIN_SD = not_an_input(
    13.0, "start value of the per-fold optimiser, an unwired exhibit constant (#365)")
# Chosen, never fitted: a team's variance before its first game has no data to be read from.
INITIAL_STRENGTH_VAR = not_an_input(
    36.0, "the chosen weakly informative prior width, an unwired exhibit constant (#365)")

BOUNDS = not_an_input(
    ((0.01, 3.0), (0.01, 0.99), (-1.0, 6.0), (8.0, 20.0)),
    "box bounds of the per-fold optimiser, in the order of the four fitted constants")
TRAILING = not_an_input(
    10, "seasons behind the supplementary close sigma, a disclosure column that decides nothing")
COMPLETE_SEASON_GAMES = not_an_input(
    256, "regular-season games below which a season counts as still in progress")
_PROB_EPS = not_an_input(1e-12, "clip so a log-loss stays finite on a certain wrong call")
MIN_FIT_GAMES = not_an_input(2, "fewest training games the optimiser is run on at all")

# One franchise, one team: nflverse writes the old abbreviation before a move.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

WITHIN = ("game_id",)
RATING_CEILING_ARM = "the betting market's last pre-kickoff poll"
# Seasons of committed pre-kickoff snapshots the gate needs (ADR-0019's minimum k).
ARCHIVE_MIN_SEASONS = not_an_input(
    3, "ADR-0019's minimum k, quoted in the NOT-RUNNABLE print and read by nothing else")

ACTIONS = Actions(
    adopt="`ratings.forecaster()` moves to the rating behind a flag and the constants are "
          "re-spelled fitted/chosen in that commit",
    remove="the exhibit stays, re-runnable, as a calibrated second opinion and a residual",
    show="nothing moves; the rating stays unwired")

HARNESS = Harness(name="state_space_rating", arm_a="the close", arm_b="state-space rating",
                  within=WITHIN, ceiling_arm=RATING_CEILING_ARM, actions=ACTIONS,
                  unit="nats of log-loss per game", places=4, arm_roots=(__name__,))


class Params(NamedTuple):
    """The four constants fitted per fold."""
    week_step_sd: float
    season_shrink: float
    home_adv: float
    margin_sd: float


def start_params() -> Params:
    return Params(WEEK_STEP_SD, SEASON_SHRINK, HOME_ADV, RATING_MARGIN_SD)


def prepare(schedules: pl.DataFrame) -> pl.DataFrame:
    """The regular-season games that have been played, one team per franchise, in play order.

    `scored` marks the games either arm is scored on (rule 11): priced, and not a tie. The
    others still feed the filter -- a result is not a price -- but are never scored.
    """
    need = {"game_id", "season", "week", "game_type", "home_team", "away_team", "result",
            "spread_line"}
    if missing := sorted(need - set(schedules.columns)):
        raise ValueError(f"schedules is missing {missing}")
    return (schedules.filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null())
            .with_columns(pl.col("home_team").replace(FRANCHISE),
                          pl.col("away_team").replace(FRANCHISE),
                          (pl.col("spread_line").is_not_null() & (pl.col("result") != 0))
                          .alias("scored"))
            .select("game_id", "season", "week", "home_team", "away_team", "result",
                    "spread_line", "scored")
            .sort("season", "week", "game_id"))


def run_filter(games: pl.DataFrame, p: Params, *, teams: Sequence[str] | None = None,
               ) -> tuple[np.ndarray, np.ndarray]:
    """`(margin_hat, variance)` for every game, each made before its week is learned from.

    `games` is `prepare`'s output (or a slice of it, still in play order). A game's prediction
    is made from the state after every game of an earlier week and none of its own week; the
    week's games are then learned from one after another, which is exact because each team
    plays once a week.
    """
    names = sorted(set(games["home_team"]) | set(games["away_team"])) if teams is None \
        else list(teams)
    ix = {t: i for i, t in enumerate(names)}
    n = len(names)
    hi = np.array([ix[t] for t in games["home_team"]], dtype=int)
    ai = np.array([ix[t] for t in games["away_team"]], dtype=int)
    season = games["season"].to_numpy()
    week = games["week"].to_numpy()
    y = games["result"].to_numpy().astype(float)

    q2 = p.week_step_sd ** 2
    keep = 1.0 - p.season_shrink
    add = p.season_shrink * (2.0 - p.season_shrink) * INITIAL_STRENGTH_VAR
    r2 = p.margin_sd ** 2
    theta = np.zeros(n)
    cov = np.eye(n) * INITIAL_STRENGTH_VAR
    last = np.zeros(n, dtype=int)          # week of a team's last game this season, 0 = none
    mean_hat = np.empty(len(y))
    var = np.empty(len(y))
    cur = season[0] if len(y) else 0
    k = 0
    while k < len(y):
        # One block is one (season, week): every game in it is predicted from the state before
        # the block, then learned from. A Sunday game's result is not known at Thursday's
        # kickoff, so no game's prediction may see another of its own week.
        stop = k
        while stop < len(y) and season[stop] == season[k] and week[stop] == week[k]:
            stop += 1
        if season[k] != cur:
            theta *= keep
            cov *= keep * keep
            cov[np.diag_indices(n)] += add
            last[:] = 0
            cur = season[k]
        w = week[k]
        for j in range(k, stop):
            for t in (hi[j], ai[j]):
                if last[t]:
                    cov[t, t] += q2 * (w - last[t])
                    last[t] = 0
        for j in range(k, stop):
            h, a = hi[j], ai[j]
            ph = cov[:, h] - cov[:, a]
            mean_hat[j] = theta[h] - theta[a] + p.home_adv
            var[j] = ph[h] - ph[a] + r2
        for j in range(k, stop):
            h, a = hi[j], ai[j]
            ph = cov[:, h] - cov[:, a]
            gain = ph / (ph[h] - ph[a] + r2)
            theta += gain * (y[j] - (theta[h] - theta[a] + p.home_adv))
            cov -= np.outer(gain, ph)
        for j in range(k, stop):
            last[hi[j]] = last[ai[j]] = w
        k = stop
    return mean_hat, var


def neg_loglik(games: pl.DataFrame, p: Params, teams: Sequence[str] | None = None) -> float:
    """Minus the filter's one-step-ahead Gaussian log-likelihood of every margin in `games`."""
    mean_hat, var = run_filter(games, p, teams=teams)
    v = games["result"].to_numpy().astype(float) - mean_hat
    return float(0.5 * np.sum(np.log(2.0 * math.pi * var) + v * v / var))


def fit_constants(past: pl.DataFrame) -> Params:
    """The four constants maximising the predictive likelihood of `past`, from the same start
    every call (L-BFGS-B inside `BOUNDS`)."""
    from scipy.optimize import minimize

    if past.height < MIN_FIT_GAMES:
        return start_params()
    names = sorted(set(past["home_team"]) | set(past["away_team"]))

    def objective(x: np.ndarray) -> float:
        return neg_loglik(past, Params(*map(float, x)), names)

    res = minimize(objective, np.array(start_params(), dtype=float), method="L-BFGS-B",
                   bounds=BOUNDS)
    return Params(*map(float, res.x))


def _logloss(p_home: np.ndarray, home_won: np.ndarray) -> np.ndarray:
    q = np.clip(p_home, _PROB_EPS, 1.0 - _PROB_EPS)
    return -(home_won * np.log(q) + (1.0 - home_won) * np.log(1.0 - q))


def _phi(x: np.ndarray) -> np.ndarray:
    return np.array([normal_cdf(float(v)) for v in x])


def trailing_sd(past: pl.DataFrame, *, trailing: int = TRAILING) -> float:
    """The sd of `result - spread_line` over the last `trailing` seasons of `past`, scored
    games only -- the close's sigma as it could have been known."""
    scored = past.filter(pl.col("scored"))
    if scored.is_empty():
        return MARGIN_SD
    floor = int(cast(int, scored["season"].max())) - trailing + 1
    resid = scored.filter(pl.col("season") >= floor)
    r = (resid["result"] - resid["spread_line"]).to_numpy().astype(float)
    return float(np.sqrt(np.mean(r * r) - np.mean(r) ** 2)) if r.size > 1 else MARGIN_SD


def walk_forward(games: pl.DataFrame, *, arm_a_sd: float = MARGIN_SD,
                 through: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`(per_game, per_season)` over every held-out season of `games` (`prepare`'s output).

    `per_game` is the paired frame the gate reads: `season`, `game_id`, `ll_close`,
    `ll_rating`, `diff` (close minus rating, positive when the rating is lower).
    `per_season` is the table this exhibit exists to print, with no verdict in it.
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    pg: list[pl.DataFrame] = []
    rows: list[dict] = []
    for season, past, now in expanding_seasons(games):
        if through is not None and season > through:
            break
        p = fit_constants(past)
        both = pl.concat([past, now])
        mean_hat, _ = run_filter(both, p, teams=teams)
        k = past.height
        sc = now["scored"].to_numpy()
        spread = now["spread_line"].to_numpy().astype(float)[sc]
        won = (now["result"].to_numpy().astype(float) > 0).astype(float)[sc]
        ll_b = _logloss(_phi(mean_hat[k:][sc] / p.margin_sd), won)
        ll_a = _logloss(_phi(spread / arm_a_sd), won)
        sd_t = trailing_sd(past)
        ll_t = _logloss(_phi(spread / sd_t), won)
        ids = now.filter(pl.col("scored"))["game_id"]
        pg.append(pl.DataFrame({"season": [season] * len(ids), "game_id": ids,
                                "ll_close": ll_a, "ll_rating": ll_b, "diff": ll_a - ll_b}))
        rows.append({"season": season, "n": int(sc.sum()),
                     "train_seasons": past["season"].n_unique(),
                     "ll_close": float(ll_a.mean()), "ll_rating": float(ll_b.mean()),
                     "gap": float(ll_a.mean() - ll_b.mean()),
                     "gap_trailing": float(ll_t.mean() - ll_b.mean()),
                     "week_step_sd": p.week_step_sd, "season_shrink": p.season_shrink,
                     "home_adv": p.home_adv, "margin_sd": p.margin_sd})
    return (pl.concat(pg) if pg else pl.DataFrame()), pl.DataFrame(rows)


def report(table: pl.DataFrame, *, places: int = 4) -> list[str]:
    """The per-season table as printed lines: gaps in nats per game, no verdict word."""
    if table.is_empty():
        return ["  no held-out season"]
    out = [f"  {'season':>6} {'n':>4} {'train':>5} {'ll_close':>9} {'ll_rating':>9} "
           f"{'gap':>8} {'gap_trail':>9}  {'q':>5} {'shrink':>6} {'home':>5} {'sd':>6}"]
    for r in table.iter_rows(named=True):
        out.append(f"  {r['season']:>6} {r['n']:>4} {r['train_seasons']:>5} "
                   f"{r['ll_close']:>9.{places}f} {r['ll_rating']:>9.{places}f} "
                   f"{r['gap']:>+8.{places}f} {r['gap_trailing']:>+9.{places}f}  "
                   f"{r['week_step_sd']:>5.2f} {r['season_shrink']:>6.2f} "
                   f"{r['home_adv']:>5.2f} {r['margin_sd']:>6.2f}")
    out.append(f"  mean gap over {table.height} seasons: {table['gap'].mean():+.{places}f} "
               f"nats per game (gap_trail {table['gap_trailing'].mean():+.{places}f})")
    return out


def archive_seasons(state: Path | None = None) -> int:
    """Seasons holding committed pre-kickoff snapshots under `state/odds/` (#383); listed by
    directory name, never read."""
    root = state if state is not None else Path(__file__).resolve().parents[3] / "state" / "odds"
    if not root.is_dir():
        return 0
    return sum(1 for d in root.iterdir() if d.is_dir() and d.name.isdigit())


def not_runnable_line(seasons: int) -> str:
    """The expected first print, naming what ends it."""
    return (f"NOT-RUNNABLE (expected, #365 stage 2 / #383): the ceiling arm is "
            f"{RATING_CEILING_ARM}, and the pre-kickoff snapshot archive holds {seasons} "
            f"season(s) against ADR-0019's minimum of {ARCHIVE_MIN_SEASONS}; the gate "
            f"becomes decidable when the archive has depth, not on a date")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.exhibits.state_space_rating",
        description="Fit the unwired state-space rating walk-forward and print the gaps.")
    ap.add_argument("--fit", action="store_true", help="the per-season walk-forward table")
    ap.add_argument("--gate", action="store_true",
                    help="run the declared harness (no ceiling exists: NOT-RUNNABLE)")
    ap.add_argument("--through", type=int, default=None,
                    help="last held-out season (default: the last complete one)")
    a = ap.parse_args(argv)
    if not (a.fit or a.gate):
        ap.print_help()
        return 0

    from hub.cli import unavailable
    from hub.fetch import nflverse

    cols = ["game_id", "season", "week", "game_type", "home_team", "away_team", "result",
            "spread_line"]
    print("  loading schedules ...")
    try:
        games = prepare(nflverse.load("schedules", nflverse.every_season(), cols=cols))
    except Exception as e:
        return unavailable("hub.exhibits.state_space_rating", "nflverse schedules", e)
    through = a.through
    if through is None:
        counts = games.group_by("season").len().filter(pl.col("len") >= COMPLETE_SEASON_GAMES)
        latest = counts["season"].max()
        through = None if latest is None else int(cast(int, latest))
    per_game, table = walk_forward(games, through=through)
    for line in report(table):
        print(line)
    if a.gate:
        # No ceiling frame: this archive holds no measured last-poll arm for these seasons.
        run = HARNESS.run(per_game, recipe=recipe(through=through, trailing=TRAILING,
                                                  close_sd=MARGIN_SD))
        for line in run.lines:
            print(line)
        print(f"  {run.verdict[0]}: {run.verdict[1]}")
        print("  " + not_runnable_line(archive_seasons()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
