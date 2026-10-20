"""Is `TD_DISPERSION` right? Refit it on strictly earlier seasons and score the next one (#472).

Pre-registered in `docs/td-dispersion.md` (committed before this module existed); that document
is the contract and this is its code. The question: `components.TD_DISPERSION` is the variance
over mean `components.draw_counts` holds exactly for an under-dispersed touchdown count, and #307
measured the within-player-season variance over mean as nearer one than the constants say.
Does the same measurement, made only from earlier seasons, predict the next season's weekly
touchdown counts better than the constants in the file?

**The estimator** (`fit_dispersion`): per phase, over the qualifying player-seasons, the pooled
within-player-season variance (`n - 1` degrees of freedom) over the pooled mean. Pooled across
positions because the constant is keyed by phase.

**The score** (`walk_forward`): for each held-out season, the log pmf of every player-week's
count under the family `draw_counts` draws, at a leave-one-out mean with one pseudo-game (the
same for both arms, so the comparison is the dispersion and nothing else), summed over the
phases the player-season qualifies in. The paired difference is refit minus shipped.

**The gate** is `experiment.gate` through one `Harness`: the season is the cluster, the player
the within-season unit, and the ceiling is a dispersion fitted to the held-out season itself.

    uv run python -m hub.models.td_dispersion --fit        # the one ledger entry
    uv run python -m hub.models.td_dispersion --power      # the pre-run simulation, no ledger
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps
from scipy.optimize import minimize_scalar

from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.declare import not_an_input
from hub.ledger import Ledger, recipe
from hub.models import components
from hub.models.experiment import (
    Actions,
    GateRun,
    Harness,
    expanding_seasons,
)

PHASES = ("rec", "rush", "pass")
_UNITS = {"rec": "receptions", "rush": "carries", "pass": "attempts"}
_TDS = {"rec": "receiving_tds", "rush": "rushing_tds", "pass": "passing_tds"}

# What the table reads of the weekly stats: the contract's one required column, the keys, the
# season type it filters on, each phase's volume and its touchdown count.
STAT_COLS = ("fantasy_points_ppr", "season", "week", "player_id", "position", "season_type",
             "attempts", "carries", "receptions", "passing_tds", "rushing_tds", "receiving_tds")

SEASONS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

# The sample of a measurement: a player-season counts at this many games and this volume a game
# in the phase (receptions, carries, attempts). They replicate #307's sample to within a few
# player-seasons a phase. No prediction reads them; the constant this gate may move is the one
# `components` declares.
MIN_GAMES = not_an_input(
    8, "a sample threshold of a measurement, docs/td-dispersion.md section 1: a player-season "
       "below it is a handful of weeks")
VOLUME_FLOOR: dict[str, float] = not_an_input(
    {"rec": 2.0, "rush": 4.0, "pass": 10.0},
    "a sample threshold of a measurement, docs/td-dispersion.md section 1: real volume in the "
    "phase, in the phase's own unit a game")
# The pseudo-game in the leave-one-out mean (docs/td-dispersion.md section 2): keeps a player
# with no other touchdown at a positive mean, shared by both arms.
PRIOR_GAMES = not_an_input(
    1.0, "the weight of the training-rate pseudo-game in the scored mean, shared by both arms; "
         "it sizes a measurement and nothing predicts from it")
# The search range for the ceiling arm's dispersion and its tolerance.
CEILING_RANGE = not_an_input(
    (0.5, 1.5), "the range the ceiling arm's dispersion is searched over; a bound on a "
                "measurement, not an input")
CEILING_TOL = not_an_input(
    0.005, "the tolerance of the ceiling arm's search, finer than the constants' own grain")
BOOTSTRAP_RESAMPLES = not_an_input(
    1000, "player-cluster resamples behind the estimator's reported interval; the gate does "
          "not read it")
_FLOOR = not_an_input(1e-300, "the probability a count past the table's end is scored at "
                              "(the support is mean + 10 sd + 25, so it is not reached)")
# The log score's floor (docs/td-dispersion.md section 10, amendment 1): a count the law calls
# rarer than one in a thousand is scored as one in a thousand. Without it a week at the edge of
# an unreachable law (a two-touchdown week at a mean of 0.05, where the narrowest law there is
# gives it 1e-30) scores minus seventy nats and a handful of such weeks are the verdict.
SCORE_FLOOR = not_an_input(
    1e-3, "the floor a count's probability is scored at in the walk-forward log score; it sizes "
          "a measurement and nothing predicts from it")

# The constants the gate was registered against and run against (docs/td-dispersion.md): the
# comparison "refit against the shipped constants" is a statement about these three numbers.
# #472 changed `components.TD_DISPERSION`, so this is no longer the live value and the gate does
# not read the live one: re-running it reproduces the run in the Ledger, and a later refit would
# name its own incumbent.
INCUMBENT: dict[str, float] = not_an_input(
    {"pass": 0.85, "rush": 0.86, "rec": 0.83},
    "the constants the gate of docs/td-dispersion.md was registered and run against, kept so "
    "the run reproduces after components.TD_DISPERSION moved")

WITHIN: tuple[str, ...] = ("player_id",)
CEILING_ARM = "a dispersion fitted to the held-out season itself"
ACTIONS = Actions(
    adopt="TD_DISPERSION is replaced by the refit (docs/td-dispersion.md section 6).",
    remove="The refit predicts worse than the shipped constants: they stay.",
    show="TD_DISPERSION stays as shipped; the comparison could not replace it or remove it.")
HARNESS = Harness(name="td_dispersion", arm_a="refit", arm_b="shipped", within=WITHIN,
                  ceiling_arm=CEILING_ARM, actions=ACTIONS,
                  unit="nats of held-out log score per player-week", places=5,
                  arm_roots=("hub.models.td_dispersion",))


# --- the table ------------------------------------------------------------------------

def phase_table(stats: pl.DataFrame) -> pl.DataFrame:
    """One row per qualifying (phase, player-season, week): its touchdown count `x`, the
    player-season's games `n`, touchdowns `s`, and position."""
    need = set(STAT_COLS) - {"season_type"}
    if need - set(stats.columns):
        raise ValueError(f"phase_table needs {sorted(need - set(stats.columns))}")
    w = stats
    if "season_type" in w.columns:
        w = w.filter(pl.col("season_type") == "REG")
    w = w.filter(pl.col("position").is_in(DRAFTED_POSITIONS))
    key = ["season", "player_id"]
    out = []
    for phase in PHASES:
        f = (w.select(*key, "week", "position",
                      pl.col(_UNITS[phase]).cast(pl.Float64).fill_null(0.0).alias("u"),
                      pl.col(_TDS[phase]).cast(pl.Float64).fill_null(0.0).alias("x"))
             .with_columns(pl.len().over(key).alias("n"),
                           pl.col("u").sum().over(key).alias("_u"),
                           pl.col("x").sum().over(key).alias("s"))
             .filter((pl.col("n") >= MIN_GAMES)
                     & (pl.col("_u") / pl.col("n") >= VOLUME_FLOOR[phase]))
             .select(pl.lit(phase).alias("phase"), *key, "week", "position", "x",
                     pl.col("n").cast(pl.Float64), "s"))
        out.append(f)
    return pl.concat(out).sort(["phase", "season", "player_id", "week"])


def _player_seasons(table: pl.DataFrame, by: Sequence[str] = ("phase",)) -> pl.DataFrame:
    """Per (phase, player-season): games, touchdowns, sum of squares and degrees of freedom."""
    return (table.group_by(["phase", "season", "player_id", "position"], maintain_order=True)
            .agg(pl.col("n").first(), pl.col("s").first(), (pl.col("x") ** 2).sum().alias("q"))
            .with_columns((pl.col("q") - pl.col("s") ** 2 / pl.col("n")).alias("ss"),
                          (pl.col("n") - 1.0).alias("dof")))


def _ratio(ss: float, dof: float, s: float, n: float) -> float:
    return float("nan") if s <= 0 or dof <= 0 else (ss / dof) / (s / n)


# --- the estimator --------------------------------------------------------------------

def fit_dispersion(table: pl.DataFrame, seasons: Sequence[int] | None = None, *,
                   by_position: bool = False) -> dict[Any, float]:
    """The pooled within-player-season variance over mean, per phase (or per phase and
    position, a diagnostic). `(sum SS / sum (n - 1)) / (sum s / sum n)` over the qualifying
    player-seasons of `seasons` (default: every season in `table`). A phase with no touchdown is
    absent rather than NaN."""
    t = table if seasons is None else table.filter(pl.col("season").is_in(list(seasons)))
    ps = _player_seasons(t)
    keys = ["phase", "position"] if by_position else ["phase"]
    out: dict[Any, float] = {}
    for key, g in ps.group_by(keys):
        phi = _ratio(float(g["ss"].sum()), float(g["dof"].sum()), float(g["s"].sum()),
                     float(g["n"].sum()))
        if np.isfinite(phi):
            out[key if by_position else key[0]] = phi
    return out


def fit_interval(table: pl.DataFrame, phase: str, *, seed: int = 0,
                 resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """A 95% player-cluster bootstrap interval of `fit_dispersion` for one phase: a player's
    seasons and weeks are resampled together (`docs/method.md` rule 3)."""
    ps = _player_seasons(table.filter(pl.col("phase") == phase))
    per = ps.group_by("player_id").agg(pl.col("ss").sum(), pl.col("dof").sum(),
                                       pl.col("s").sum(), pl.col("n").sum())
    ss, dof, s, n = (per[c].to_numpy().astype(float) for c in ("ss", "dof", "s", "n"))
    if per.is_empty():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per.height, size=(resamples, per.height))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (ss[idx].sum(1) / dof[idx].sum(1)) / (s[idx].sum(1) / n[idx].sum(1))
    lo, hi = np.nanpercentile(draws, (2.5, 97.5))
    return float(lo), float(hi)


def training_rate(table: pl.DataFrame) -> dict[str, float]:
    """Touchdowns a game per phase over the qualifying player-seasons: the pseudo-game's value."""
    ps = _player_seasons(table)
    return {k[0]: float(g["s"].sum()) / float(g["n"].sum()) for k, g in ps.group_by("phase")}


# --- the law that is scored -----------------------------------------------------------

def _logpmf_one(mean: float, phi: float, k: np.ndarray) -> np.ndarray:
    """`log P(k)` for one mean, in the family `components.draw_counts` draws at `phi`."""
    if phi < 1.0:
        pmf = components.cmp_pmf(mean, components.cmp_nu(mean, phi))
        p = np.where(k < pmf.size, pmf[np.minimum(k, pmf.size - 1)], 0.0)
        return np.log(np.maximum(p, _FLOOR))
    if phi == 1.0:
        return sps.poisson.logpmf(k, mean)
    p = 1.0 / phi
    return sps.nbinom.logpmf(k, max(mean * p / (1.0 - p), 1e-9), p)


def count_logpmf(x: np.ndarray, mean: np.ndarray, phi: float, floor: float = _FLOOR
                 ) -> np.ndarray:
    """The log probability of each count `x` at its `mean`, in the law the sampler draws:
    Conway-Maxwell-Poisson below one, Poisson at one, negative binomial above, no lower than
    `log(floor)`."""
    x = np.asarray(x, dtype=int)
    mean = np.asarray(mean, dtype=float)
    out = np.empty(x.shape)
    uniq, inv = np.unique(mean, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    for j, m in enumerate(uniq):
        idx = order[bounds[j]:bounds[j + 1]]
        out[idx] = _logpmf_one(float(m), phi, x[idx])
    return np.maximum(out, np.log(floor))


def _top_dispersion(x: np.ndarray, m: np.ndarray, floor: float = SCORE_FLOOR) -> float:
    """The dispersion that maximises these rows' total log score: the ceiling arm."""
    res = minimize_scalar(lambda p: -float(count_logpmf(x, m, p, floor).sum()),
                          bounds=CEILING_RANGE, method="bounded",
                          options={"xatol": CEILING_TOL})
    return float(res.x)


# --- the walk-forward -----------------------------------------------------------------

def walk_forward(table: pl.DataFrame, *, incumbent: Mapping[str, float] | None = None,
                 ceiling: Mapping[str, float] | None = None,
                 floor: float = SCORE_FLOOR) -> pl.DataFrame:
    """One row per held-out player-week: `diff` (refit minus shipped, nats) and `ceiling_diff`
    (the ceiling arm minus shipped), summed over the phases the player-season qualifies in.

    `expanding_seasons` is the split: the refit and the pseudo-game's rate read seasons
    strictly before the one scored. `ceiling` is a fixed dispersion per phase for a caller that
    knows it (a planted truth); absent, it is searched on the held-out season itself.
    """
    shipped = dict(INCUMBENT if incumbent is None else incumbent)
    schema = pl.Schema({"season": pl.Int64, "player_id": pl.Utf8, "week": pl.Int64,
                        "diff": pl.Float64, "ceiling_diff": pl.Float64})
    frames: list[pl.DataFrame] = []
    for _season, past, now in expanding_seasons(table):
        fit, rate = fit_dispersion(past), training_rate(past)
        for phase in PHASES:
            rows = now.filter(pl.col("phase") == phase)
            if rows.is_empty() or phase not in rate:
                continue
            x = rows["x"].to_numpy().astype(int)
            # n - 1 other games plus PRIOR_GAMES pseudo-games; with PRIOR_GAMES == 1 this is n.
            m = ((rows["s"].to_numpy() - x + PRIOR_GAMES * rate[phase])
                 / (rows["n"].to_numpy() - 1.0 + PRIOR_GAMES))
            base = count_logpmf(x, m, shipped[phase], floor)
            new = count_logpmf(x, m, fit.get(phase, shipped[phase]), floor) - base
            top_phi = ceiling[phase] if ceiling is not None else _top_dispersion(x, m, floor)
            top = count_logpmf(x, m, top_phi, floor) - base
            frames.append(rows.select("season", "player_id", "week").with_columns(
                pl.Series("diff", new), pl.Series("ceiling_diff", top)).cast(schema))
    if not frames:
        return pl.DataFrame(schema=schema)
    return (pl.concat(frames).group_by(["season", "player_id", "week"])
            .agg(pl.col("diff").sum(), pl.col("ceiling_diff").sum())
            .sort(["season", "player_id", "week"]))


def gate_run(paired: pl.DataFrame, *, publish: bool = False, seasons: Sequence[int] = (),
             seed: int = 0, ledger: Ledger | None = None) -> GateRun:
    """The refit against the shipped constants, through the one Gate. `publish=False` is
    `Harness.decide`, the pure half; `publish=True` is `Harness.run`, which writes the Ledger."""
    if publish:
        return HARNESS.run(paired, seed=seed, ledger=ledger,
                           recipe=recipe(seasons=list(seasons)))
    return HARNESS.decide(paired, seed=seed)


# --- planted data, for the controls and the power simulation --------------------------

# Per-game touchdown mean and spread by phase of the planted player-seasons: typical of the
# phase, and only the shape of the real table; no real outcome is read.
_PLANT_MEAN = not_an_input({"rec": 0.30, "rush": 0.30, "pass": 1.45},
                           "the shape of a planted table for a simulation; no prediction reads it")
_PLANT_SD = not_an_input({"rec": 0.22, "rush": 0.30, "pass": 0.45},
                         "the shape of a planted table for a simulation; no prediction reads it")
_PLANT_GRID = not_an_input(0.02, "the grain planted means are rounded to, so the sampler's "
                                 "per-mean solve is shared across players")
_PLANT_GAMES = not_an_input((8, 17), "the range of games a planted player-season plays, as a "
                                     "real one does")


def planted_table(truth: Mapping[str, float], seasons: Sequence[int],
                  per_phase: Mapping[str, int], *, seed: int = 0) -> pl.DataFrame:
    """A `phase_table` whose counts are drawn by `components.draw_counts` at the dispersion
    `truth` names, `per_phase[phase]` player-seasons a season: a known answer to recover."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for phase in PHASES:
        mu, sd = _PLANT_MEAN[phase], _PLANT_SD[phase]
        shape, scale = (mu / sd) ** 2, sd ** 2 / mu
        for season in seasons:
            for i in range(per_phase[phase]):
                n = int(rng.integers(_PLANT_GAMES[0], _PLANT_GAMES[1] + 1))
                mean = max(_PLANT_GRID, round(float(rng.gamma(shape, scale)) / _PLANT_GRID)
                           * _PLANT_GRID)
                x = components.draw_counts(rng, mean, truth[phase], n)
                for w, c in enumerate(x):
                    rows.append({"phase": phase, "season": season, "player_id": f"{phase}{i}",
                                 "week": w + 1, "position": phase, "x": float(c),
                                 "n": float(n), "s": float(x.sum())})
    return pl.DataFrame(rows).sort(["phase", "season", "player_id", "week"])


def _replicate(args: tuple[Mapping[str, float], Mapping[str, int], Sequence[int], int]
               ) -> dict[str, float]:
    """One planted walk-forward and its gate: the unit `simulate` runs many of."""
    truth, per_phase, seasons, seed = args
    table = planted_table(truth, seasons, per_phase, seed=seed)
    paired = walk_forward(table, ceiling=truth)
    run = HARNESS.decide(paired, seed=seed, bootstrap=200)
    s = run.summary
    return {"adopt": float(run.verdict[0] == "ADOPT"),
            "not_runnable": float(run.verdict[0] == "NOT-RUNNABLE"),
            "remove": float(run.verdict[0] == "REMOVE"),
            "mean": float(s["mean"]), "mde": float(s.get("mde", float("nan"))),
            "ceiling": float(s.get("ceiling", float("nan"))),
            "se": float(s.get("se", float("nan")))}


def simulate(truth: Mapping[str, float], per_phase: Mapping[str, int], seasons: Sequence[int],
             reps: int, *, seed: int = 0, workers: int = 1,
             map_fn: Callable[..., Any] | None = None) -> dict[str, float]:
    """The rule's ADOPT / NOT-RUNNABLE / REMOVE rates, with their standard errors, and the
    mean MDE and ceiling, over `reps` planted walk-forwards at `truth`."""
    jobs = [(truth, per_phase, seasons, seed + r) for r in range(reps)]
    if map_fn is not None:
        got = list(map_fn(_replicate, jobs))
    elif workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            got = list(pool.map(_replicate, jobs))
    else:
        got = [_replicate(j) for j in jobs]
    out: dict[str, float] = {"reps": float(reps)}
    for k in ("adopt", "not_runnable", "remove"):
        p = float(np.mean([g[k] for g in got]))
        out[k] = p
        out[f"{k}_se"] = float(np.sqrt(p * (1.0 - p) / reps))
    for k in ("mean", "mde", "ceiling", "se"):
        out[k] = float(np.nanmean([g[k] for g in got]))
    return out


# --- the sampler's tail, descriptive --------------------------------------------------

@contextmanager
def _dispersion(values: Mapping[str, float]):
    """`components.TD_DISPERSION` set to `values` for a block, and put back."""
    before = dict(components.TD_DISPERSION)
    components.TD_DISPERSION.update(values)
    try:
        yield
    finally:
        components.TD_DISPERSION.clear()
        components.TD_DISPERSION.update(before)


TAIL_LINES = (20, 25, 30, 35, 40)
_TAIL_MIN_PPG = not_an_input(8.0, "the sample of docs/td-count-sampler.md section 3's tail table")
_TAIL_DRAWS = not_an_input(4000, "draws a line in the tail table, as #307's")
_LINE_COLS = (("attempts", "attempts"), ("passing_yards", "passing_yards"),
              ("passing_tds", "passing_tds"), ("passing_interceptions", "interceptions"),
              ("carries", "carries"), ("rushing_yards", "rushing_yards"),
              ("rushing_tds", "rushing_tds"), ("receptions", "receptions"),
              ("receiving_yards", "receiving_yards"), ("receiving_tds", "receiving_tds"))
_FUMBLES = ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost")
TAIL_COLS = ("fantasy_points_ppr", "season", "week", "player_id", "position", "season_type",
             *(c for c, _ in _LINE_COLS), *_FUMBLES)


def tail_calibration(stats: pl.DataFrame, seasons: Sequence[int], dispersions:
                     Mapping[str, Mapping[str, float]], *, draws: int = _TAIL_DRAWS,
                     seed: int = 0) -> dict[str, Any]:
    """Weeks at or above each fantasy-points line, observed and expected under each named
    dispersion, over the qualifying QB/RB/WR/TE player-seasons of `seasons`, each sampled on its
    own season-average line by `components.sample_weeks` (docs/td-count-sampler.md section 3).

    Descriptive, not the gate: an aggregate of counts with no paired unit, confounded with the
    other fitted spreads the refit does not touch."""
    w = stats.filter((pl.col("season_type") == "REG") & pl.col("season").is_in(list(seasons))
                     & pl.col("position").is_in(DRAFTED_POSITIONS))
    fum = sum((pl.col(c).fill_null(0.0) for c in _FUMBLES if c in w.columns), start=pl.lit(0.0))
    mean_cols = [pl.col(c).cast(pl.Float64).fill_null(0.0).mean().alias(k)
                 for c, k in _LINE_COLS if c in w.columns]
    ps = (w.with_columns(fum.alias("_fum")).group_by(["season", "player_id", "position"])
          .agg(pl.len().alias("games"), pl.col("fantasy_points_ppr").mean().alias("ppg"),
               pl.col("_fum").mean().alias("fumbles_lost"), *mean_cols)
          .filter((pl.col("games") >= MIN_GAMES) & (pl.col("ppg") >= _TAIL_MIN_PPG))
          .sort(["season", "player_id", "position"]))
    obs = dict.fromkeys(TAIL_LINES, 0)
    observed = w.join(ps.select("season", "player_id", "position"),
                      on=["season", "player_id", "position"])["fantasy_points_ppr"].to_numpy()
    for t in TAIL_LINES:
        obs[t] = int((observed >= t).sum())
    out: dict[str, Any] = {"player_seasons": ps.height, "player_weeks": int(observed.size),
                           "observed": obs}
    lines = ps.to_dicts()
    for name, disp in dispersions.items():
        exp = dict.fromkeys(TAIL_LINES, 0.0)
        with _dispersion(disp):
            for i, line in enumerate(lines):
                d = components.sample_weeks(line, line["position"], draws,
                                            np.random.default_rng(seed + i))
                for t in TAIL_LINES:
                    exp[t] += float((d >= t).mean()) * line["games"]
        out[name] = {t: round(v, 1) for t, v in exp.items()}
    return out


# --- the entry point ------------------------------------------------------------------

def _estimator_report(table: pl.DataFrame, since: int = 2022) -> list[str]:
    all_fit = fit_dispersion(table)
    recent = fit_dispersion(table, [s for s in table["season"].unique().to_list() if s >= since])
    by_pos = fit_dispersion(table, by_position=True)
    lines = ["", "The estimator on every season loaded, with a player-cluster 95% interval:",
             f"  {'phase':<6}{'all seasons':>13}{'interval':>20}{f'{since}+':>9}{'registered':>11}"]
    for p in PHASES:
        lo, hi = fit_interval(table, p)
        lines.append(f"  {p:<6}{all_fit.get(p, float('nan')):>13.3f}"
                     f"{f'[{lo:.3f}, {hi:.3f}]':>20}{recent.get(p, float('nan')):>9.3f}"
                     f"{INCUMBENT[p]:>11.2f}")
    lines += ["", "By position (a diagnostic; the constant is keyed by phase):"]
    for (p, pos), v in sorted(by_pos.items()):
        lines.append(f"  {p:<6}{pos:<4}{v:>8.3f}")
    return lines


def main(argv: Sequence[str] | None = None, *, load: Callable[..., pl.DataFrame] | None = None,
         ledger: Ledger | None = None) -> int:
    """`load` and `ledger` are the seam of #403: the reader, called as `nflverse.load` is, and
    the ledger the run is written to."""
    p = argparse.ArgumentParser(prog="hub.models.td_dispersion", description=__doc__)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--power", action="store_true")
    p.add_argument("--tails", action="store_true")
    p.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    p.add_argument("--reps", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(list(argv) if argv is not None else None)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]

    if a.power:
        per_phase = {"rec": 165, "rush": 78, "pass": 36}
        scen = {"N (truth = the incumbent)": dict(INCUMBENT),
                "P (truth = the issue's figures)": {"rec": 0.96, "rush": 0.96, "pass": 0.84}}
        held_out = seasons
        for name, truth in scen.items():
            r = simulate(truth, per_phase, held_out, a.reps, workers=a.workers)
            print(f"\n{name}: {int(r['reps'])} reps over {len(held_out)} seasons")
            for k in ("adopt", "not_runnable", "remove"):
                print(f"  {k:<13}{r[k]:.3f} (se {r[f'{k}_se']:.3f})")
            print(f"  mean gain {r['mean']:+.5f}, se {r['se']:.5f}, mde {r['mde']:.5f}, "
                  f"ceiling {r['ceiling']:+.5f}")
        return 0
    if not (a.fit or a.tails):
        p.print_help()
        return 0

    from hub.fetch import nflverse
    read = load or nflverse.load
    try:
        stats = read("player_stats", seasons,
                     cols=TAIL_COLS if a.tails else STAT_COLS)
    except Exception as e:
        return unavailable("hub.models.td_dispersion", "nflverse weekly player stats", e)

    if a.tails:
        fit = fit_dispersion(phase_table(stats))
        new = {ph: round(fit.get(ph, INCUMBENT[ph]), 2) for ph in PHASES}
        recent = [s for s in seasons if s >= 2022]
        r = tail_calibration(stats, recent, {"registered": dict(INCUMBENT),
                                             "refit": new})
        print(f"{r['player_seasons']} player-seasons, {r['player_weeks']} player-weeks, "
              f"{recent[0]}-{recent[-1]}")
        for t in TAIL_LINES:
            print(f"  >= {t}: observed {r['observed'][t]}, registered {r['registered'][t]}, "
                  f"refit {r['refit'][t]}")
        return 0

    table = phase_table(stats)
    print(f"{table.select('phase', 'season', 'player_id').unique().height} qualifying "
          f"player-season-phases over {len(seasons)} seasons")
    paired = walk_forward(table)
    if paired.is_empty():
        print("no held-out season")
        return 0
    # Everything that can fail on the data is computed before the Ledger is written, so a
    # crash cannot leave one entry behind and invite a second run.
    report = _estimator_report(table)
    raw = walk_forward(table, floor=_FLOOR, ceiling=dict(INCUMBENT))
    by_season = (raw.group_by("season").agg(pl.col("diff").mean()).sort("season"))
    diag = ["", "The unfloored mean refit-minus-shipped, nats per player-week (a diagnostic; "
            "the rule reads the floored score):"]
    diag += [f"  {int(r['season'])}: {r['diff']:+.5f}" for r in by_season.iter_rows(named=True)]
    run = gate_run(paired, publish=True, seasons=seasons, ledger=ledger)
    print("\n".join(run.lines))
    print("\n".join(report))
    print("\n".join(diag))
    print(f"\n{run.verdict[0]}: {run.verdict[1]}")
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
