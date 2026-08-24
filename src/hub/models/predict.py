"""One prediction object: what a player does in a week.

The pieces of this existed but as three implementations of one idea -- `weekly_moments`
handed back two moments, the skewed correlated draw lived inside `simulate_weeks`, and
`hub/season/lineup.py` computed its own group spread. Three copies of one idea is how they
drift apart, and drift is silent.

The split is by subject, not convenience: **this module is what a player does; `hub.draft.season`
is how a league works.** Dispersion laws, skew, talent and teammate correlation live here.
Rosters, schedules, brackets and lineups live there.

Everything here was fitted rather than assumed, and each constant carries its own provenance
below: `TALENT_CV` in `docs/talent-cv.md`, the square-root spread law in
`docs/weekly-spread.md`, the skew in `docs/component-projection.md`, teammate correlation in
`docs/correlation.md`.

**One constraint the unification had to respect.** Component-derived spread was measured
*worse* than the fitted square-root law at predicting a player's actual weekly sd -- mean
error 1.365 against 1.140, P(better) 0.0%. So this object keeps the fitted laws for its
moments and exposes components alongside them rather than deriving one from the other.
Unifying must not quietly swap a validated number for a tidier one.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# How wrong a preseason projection typically is about a player's season, as a fraction of
# his projected per-game points. This is the single most important number in the model:
# at 0 the projection is truth and drafting on it is clairvoyance; the larger it gets, the
# more a draft is a lottery and the flatter every candidate's championship equity becomes.
#
# FITTED 2026-08-23 against this league's own past drafts -- `hub.draft.calibrate`, written
# up in `docs/talent-cv.md`. 460 drafted skill players over 2023-25: 0.411, 95% CI
# [0.380, 0.434]. The previous value of 0.35 was a guess and sat 4.6 se low. Refitted the
# same day once `weekly_moments` moved to the square-root law, since the fit subtracts
# weekly sampling and therefore depends on it -- the two constants are coupled.
#
# A draft pick is market opinion recorded before week 1, so E[realized | pick, position] is
# the market's projection and cannot have been revised after the fact. Availability is
# inside the number on purpose: scoring is measured per team game, and the simulator benches
# a low-talent player the same way you bench an injured one.
#
# A single scalar is a compromise. RB fits at 0.50 and TE at 0.32, which is a real
# difference and not noise; see the doc.
TALENT_CV = 0.42

# Per position, from the same fit. Only two of these are really different from the pool:
# RB sits +2.6 se above it and TE -3.8 se below, while QB and WR are within one standard
# error and shrink back onto 0.41. That is the honest reading -- an early running back is
# more of a lottery than his projection suggests, and a tight end less of one -- and it is
# why these are shrunk estimates rather than the four raw numbers, which would treat every
# difference between 51 and 200 players as real.
TALENT_CV_BY_POS = {"QB": 0.42, "RB": 0.50, "WR": 0.42, "TE": 0.32}


def talent_cv_for(pos: np.ndarray) -> np.ndarray:
    """Per-player talent dispersion. Anything unfitted (K, DST) falls back to the pool."""
    return np.array([TALENT_CV_BY_POS.get(str(p), TALENT_CV) for p in pos], dtype=float)


# Weekly spread follows sqrt(mean), not the mean. `sd = 0.55 * mu` assumed proportional;
# fitted against 1,174 player-seasons of nflverse weekly scoring the exponent is
# 0.498 +/- 0.012, which is the Poisson value and about 42 se from 1.
#
# This is derived rather than fitted-for-its-own-sake. Weekly points are a sum of
# count-driven components -- receptions, carries, touchdowns -- whose variance grows with
# their mean, so the spread of the total grows with its square root. Touchdowns are the
# lumpy part: 54% of a quarterback's weekly variance and 21% of a receiver's, against 32%
# and 17% of their points. See docs/weekly-spread.md.
#
# Predicting a player's weekly sd this way cuts RMSE about a third versus the old constant.
# Note how little is left between positions once the law is right: most of what looked like
# a position effect in weekly CV was position differences in mean points.
WEEKLY_K = {"QB": 1.88, "RB": 2.07, "WR": 2.13, "TE": 1.99}
WEEKLY_K_POOLED = 2.04

# Weekly scoring is right-skewed, measured within player-season across 2022-25. A normal
# says 0.00, and drawing normals made the simulator believe the typical week was the
# projection -- the observed median week is about 0.90 of the mean, because the mean is
# carried by touchdown spikes. That flatters every floor-based decision.
#
# Quarterbacks are nearly symmetric because passing yardage is high volume and steady, so
# the lumpy touchdown term is a smaller share of their total. See
# docs/component-projection.md, where this falls out of sampling the components rather than
# being imposed here.
WEEKLY_SKEW = {"QB": 0.15, "RB": 0.67, "WR": 0.66, "TE": 0.72}
WEEKLY_SKEW_POOLED = 0.60
# Beyond this the gamma is indistinguishable from a normal and the shift gets numerically
# silly, so fall back rather than push it.
MIN_SKEW = 0.05


def weekly_skew_for(pos: np.ndarray) -> np.ndarray:
    """Per-player weekly skew; anything unfitted falls back to the pooled value."""
    return np.array([WEEKLY_SKEW.get(str(p), WEEKLY_SKEW_POOLED) for p in pos], dtype=float)


def skewed(mean, sd, skew, z):
    """Map standard normal draws to a distribution with this mean, spread and skew.

    Cornish-Fisher rather than a gamma, so that correlation can be applied to `z` before the
    transform: a Gaussian latent is trivially correlatable and a gamma is not. The quadratic
    term supplies the skew, and dividing by sqrt(1 + 2a^2) restores the variance the term
    adds, so mean and spread come out exactly as asked.

    Clipped at zero -- the transform has support below it and nobody scores negative points
    often enough to matter.
    """
    a = np.maximum(skew, MIN_SKEW) / 6.0
    y = (z + a * (z ** 2 - 1.0)) / np.sqrt(1.0 + 2.0 * a ** 2)
    return np.clip(mean + sd * y, 0.0, None)


def correlated_normal(rng, size, pos, nfl_team):
    """Standard normals correlated between teammates, independent otherwise.

    Only the quarterback's edges carry anything -- QB-WR +0.232, QB-TE +0.225, QB-RB +0.054,
    everything else within a few points of zero (docs/correlation.md). Applied by Cholesky
    on each NFL team's own small block, which is exact and costs nothing at 32 teams.
    """
    from hub.models.components import teammate_rho

    z = rng.standard_normal(size)
    if nfl_team is None:
        return z
    teams = np.asarray(nfl_team, dtype=object)
    for team in {t for t in teams.tolist() if t is not None}:
        idx = np.flatnonzero(teams == team)
        if idx.size < 2:
            continue
        r = np.eye(idx.size)
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                r[i, j] = r[j, i] = teammate_rho(str(pos[idx[i]]), str(pos[idx[j]]))
        if not np.any(r - np.eye(idx.size)):
            continue
        try:
            chol = np.linalg.cholesky(r)
        except np.linalg.LinAlgError:
            continue
        z[..., idx] = z[..., idx] @ chol.T
    return z


def weekly_moments(xp: pl.DataFrame, floor_sd: float = 0.0) -> pl.DataFrame:
    """Per-player weekly mean and dispersion.

    mu comes from expected points rather than realised: xFP already strips the
    week-to-week luck we are about to re-add, so using realised points would double-count
    variance and make every roster look more volatile than it is.

    `floor_sd` defaults to zero now. The old floor of 2.0 gave a player projected at
    nothing a real weekly spread, which the best-lineup rule -- a max over the roster --
    turned into free points off the end of the bench. sqrt(0) is 0, which is what an
    unprojected player should carry.
    """
    # Prefer the market's projection for the season being drafted. xFP describes the
    # season just gone, and using it as truth makes any strategy ranked on xFP look
    # prescient: it is scoring against the very data it optimised. Fall back to xFP only
    # where no projection exists.
    cols = [pl.col(c) for c in ("proj_blend", "proj_ppg", "xfp_per_game")
            if c in xp.columns]
    if not cols:
        raise ValueError(
            "weekly_moments needs one of proj_blend, proj_ppg or xfp_per_game; "
            f"got {sorted(xp.columns)}")
    mu = pl.coalesce(*cols).fill_null(0.0)
    pos_col = ("position" if "position" in xp.columns
               else ("pos" if "pos" in xp.columns else None))
    # cast + fill_null: an all-null position column comes through as dtype Null, which
    # replace_strict refuses outright rather than defaulting.
    k = (pl.col(pos_col).cast(pl.Utf8).fill_null("")
         .replace_strict(WEEKLY_K, default=WEEKLY_K_POOLED, return_dtype=pl.Float64)
         if pos_col else pl.lit(WEEKLY_K_POOLED))
    return xp.with_columns(mu.alias("mu")).with_columns(
        pl.max_horizontal(k * pl.col("mu").clip(0.0).sqrt(), pl.lit(floor_sd)).alias("sd"))




# Within-game correlation between teammates, measured on standardised weekly points,
# 2022-25. Only the quarterback edges carry anything: a quarterback and a receiving
# teammate move together (+0.23), and everything else is inside +/-0.06 of zero --
# including two receivers on the same team, at +0.014, which is not what the folklore says.
#
# The gate in docs/correlation.md is what makes this worth carrying: for a lineup holding a
# quarterback and his own pass catchers, treating them as independent gives an 80% interval
# that covers 72.9% of the time. Adding these puts it at 80.4%. For a lineup with no
# quarterback, independence is already calibrated and this changes nothing.
TEAMMATE_RHO: dict[tuple[str, str], float] = {
    ("QB", "WR"): 0.232, ("QB", "TE"): 0.225, ("QB", "RB"): 0.054,
}


def teammate_rho(a: str, b: str) -> float:
    """Correlation between two teammates by position. Zero for anything unmeasured."""
    return TEAMMATE_RHO.get((a, b), TEAMMATE_RHO.get((b, a), 0.0))


def group_sd(mu_sd_pos_team) -> float:
    """Spread of a group's combined score, counting teammate covariance.

    `mu_sd_pos_team` is an iterable of (sd, position, nfl_team). Players on different NFL
    teams contribute no covariance; a summed variance that ignores the ones who do is the
    overconfidence the L1 gate measured.
    """
    rows = list(mu_sd_pos_team)
    var = sum(float(sd) ** 2 for sd, _, _ in rows)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sd_i, pos_i, team_i = rows[i]
            sd_j, pos_j, team_j = rows[j]
            if team_i is not None and team_i == team_j:
                var += 2.0 * teammate_rho(str(pos_i), str(pos_j)) * float(sd_i) * float(sd_j)
    return float(max(var, 0.0) ** 0.5)


def moments(xp: pl.DataFrame, floor_sd: float = 0.0) -> pl.DataFrame:
    """Per-player weekly mean, spread and skew.

    mu comes from expected points rather than realised: xFP already strips the week-to-week
    luck being re-added, so using realised points would double-count variance and make every
    roster look more volatile than it is.

    `floor_sd` defaults to zero. The old floor of 2.0 gave a player projected at nothing a
    real weekly spread, which the best-lineup rule -- a max over the roster -- turned into
    free points off the end of the bench. sqrt(0) is 0, which is what an unprojected player
    should carry.
    """
    cols = [pl.col(c) for c in ("proj_blend", "proj_ppg", "xfp_per_game")
            if c in xp.columns]
    if not cols:
        raise ValueError(
            "moments needs one of proj_blend, proj_ppg or xfp_per_game; "
            f"got {sorted(xp.columns)}")
    mu = pl.coalesce(*cols).fill_null(0.0)
    pos_col = ("position" if "position" in xp.columns
               else ("pos" if "pos" in xp.columns else None))
    if pos_col is None:
        k: pl.Expr = pl.lit(WEEKLY_K_POOLED)
        sk: pl.Expr = pl.lit(WEEKLY_SKEW_POOLED)
    else:
        # cast + fill_null: an all-null position column comes through as dtype Null, which
        # replace_strict refuses outright rather than defaulting.
        base = pl.col(pos_col).cast(pl.Utf8).fill_null("")
        k = base.replace_strict(WEEKLY_K, default=WEEKLY_K_POOLED,
                                return_dtype=pl.Float64)
        sk = base.replace_strict(WEEKLY_SKEW, default=WEEKLY_SKEW_POOLED,
                                 return_dtype=pl.Float64)
    return xp.with_columns(mu.alias("mu")).with_columns(
        pl.max_horizontal(k * pl.col("mu").clip(0.0).sqrt(), pl.lit(floor_sd)).alias("sd"),
        sk.alias("skew"))


def components(pick: float, position: str, proj_ppg: float) -> dict[str, float]:
    """The component line behind a points projection.

    Here rather than only in `hub.models.volume` so a consumer wanting stats instead of
    points -- the props audit, a future volume model -- reads them off the same object that
    produced the moments.
    """
    from hub.models.volume import decompose
    return decompose(pick, position, proj_ppg)
