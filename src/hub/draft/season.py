"""Simulate a season to a champion.

`cost_of_waiting` answers "who will I regret not taking". It does not answer "which pick
most improves my chance of winning", and those come apart in ways that matter: a fourth
good WR has real VOR and almost no effect on a championship, because you can only start
three. Roster construction, positional saturation and week-to-week variance only show up
if you play the season out.

The model is deliberately shallow where shallowness is cheap and honest where it is not:

  * weekly points ~ Normal(mu, sd) truncated at zero, per player, independent
  * mu is expected points per game (xFP), which is less noisy than realised points
  * sd is that player's own week-to-week dispersion
  * lineups are set optimally in hindsight each week

The last one is the biggest simplification: real managers start the wrong guy. It inflates
everyone's scores roughly equally, so it distorts P(win) far less than it distorts points.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# QB1 / RB2 / WR3 / TE1 / FLEX1 -- confirmed against the live league, not the ESPN default.
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
FLEX_FROM = ("RB", "WR", "TE")
REG_SEASON_WEEKS = 14
PLAYOFF_TEAMS = 6

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


def _skewed(mean, sd, skew, z):
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


def _correlated_normal(rng, size, pos, nfl_team):
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


def _lineup_points(scores: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Best legal lineup, vectorised over (sims, weeks).

    scores: (sims, weeks, roster) -- one draw per player per week
    pos:    (roster,) -- position string per player
    """
    total = np.zeros(scores.shape[:2])
    leftovers = []
    for p, n in STARTERS.items():
        idx = np.flatnonzero(pos == p)
        if idx.size == 0:
            continue
        block = -np.sort(-scores[:, :, idx], axis=2)      # descending
        total += block[:, :, :n].sum(axis=2)
        if p in FLEX_FROM and block.shape[2] > n:
            leftovers.append(block[:, :, n])              # best bench player at this slot
    if leftovers:
        total += np.max(np.stack(leftovers, axis=-1), axis=-1)
    return total


def simulate_weeks(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                   pos: np.ndarray, n_sims: int, weeks: int = REG_SEASON_WEEKS,
                   rng: np.random.Generator | None = None,
                   talent_cv: float | np.ndarray | None = None,
                   nfl_team: np.ndarray | None = None) -> np.ndarray:
    """Weekly points for every team. Returns (sims, weeks, teams).

    Realised talent is drawn once per season, then weekly points are drawn around it.
    Without that first draw the projection IS the truth, and any strategy that ranks on
    the projection drafts with perfect foresight while an ADP-following opponent does
    not -- which is not an edge, it is a leak.
    """
    rng = rng or np.random.default_rng(0)
    # None means per position; a scalar is still accepted, which is what the sweeps in
    # hub.draft.leverage need in order to vary one thing at a time.
    if talent_cv is None:
        talent_cv = talent_cv_for(pos)
    true_mu = mu[None, :] * (1.0 + rng.normal(0.0, talent_cv, size=(n_sims, mu.size)))
    np.clip(true_mu, 0.0, None, out=true_mu)
    # Weekly spread follows *realised* talent, not the projection. Keying it to the
    # projection made busts impossible: a player projected at 15 whose talent went to zero
    # was drawn from N(0, 8.25) and clipped, which is a half-normal averaging 3.3 points a
    # game -- 22% of his projection, manufactured entirely by the clip. Scaling by the
    # realised ratio leaves an average player untouched and lets a collapsed one score
    # nothing, which is what a bust is.
    # Spread follows *realised* talent. Under sd = k*sqrt(mu) that means scaling by
    # sqrt(realised/projected), not by the ratio -- a player who realises a quarter of his
    # projection keeps half his weekly spread, not a quarter of it.
    ratio = np.divide(true_mu, mu[None, :], out=np.zeros_like(true_mu),
                      where=mu[None, :] > 0)
    sd_eff = (sd[None, :] * np.sqrt(ratio))[:, None, :]
    z = _correlated_normal(rng, (n_sims, weeks, mu.size), pos, nfl_team)
    draws = _skewed(true_mu[:, None, :], sd_eff,
                    weekly_skew_for(pos)[None, None, :], z)
    out = np.empty((n_sims, weeks, len(rosters)))
    for t, r in enumerate(rosters):
        out[:, :, t] = _lineup_points(draws[:, :, r], pos[r]) if r.size else 0.0
    return out


def _round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method schedule, repeated until the season is full."""
    rot = list(range(teams))
    base = []
    for _ in range(teams - 1):
        base.append([(rot[i], rot[teams - 1 - i]) for i in range(teams // 2)])
        rot = [rot[0], rot[-1], *rot[1:-1]]
    return [base[w % len(base)] for w in range(weeks)]


def champion_probability(rosters: list[np.ndarray], mu: np.ndarray, sd: np.ndarray,
                         pos: np.ndarray, n_sims: int = 400,
                         rng: np.random.Generator | None = None,
                         talent_cv: float | np.ndarray | None = None,
                         nfl_team: np.ndarray | None = None) -> np.ndarray:
    """P(each team wins the league). Returns (teams,) summing to 1.

    14-week H2H regular season, top 6 seeds, two byes, then single elimination on
    one-week matchups -- read off the live league, not assumed.
    """
    rng = rng or np.random.default_rng(0)
    teams = len(rosters)
    pts = simulate_weeks(rosters, mu, sd, pos, n_sims, REG_SEASON_WEEKS, rng, talent_cv,
                         nfl_team)

    wins = np.zeros((n_sims, teams))
    for w, pairs in enumerate(_round_robin(teams, REG_SEASON_WEEKS)):
        for a, b in pairs:
            a_won = pts[:, w, a] > pts[:, w, b]
            wins[:, a] += a_won
            wins[:, b] += ~a_won

    # Seed on wins, break ties on total points -- the standard ESPN rule.
    total = pts.sum(axis=1)
    key = wins * 10_000.0 + total
    seeds = np.argsort(-key, axis=1)[:, :PLAYOFF_TEAMS]

    champs = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        field = list(seeds[s])
        wk = REG_SEASON_WEEKS
        # Seeds 1-2 bye; 3v6 and 4v5 in round one.
        alive = [field[0], field[1]]
        for a, b in ((field[2], field[5]), (field[3], field[4])):
            alive.append(a if _week_winner(pts, s, wk, a, b) else b)
        wk += 1
        semi = [alive[0] if _week_winner(pts, s, wk, alive[0], alive[3]) else alive[3],
                alive[1] if _week_winner(pts, s, wk, alive[1], alive[2]) else alive[2]]
        wk += 1
        champs[s] = semi[0] if _week_winner(pts, s, wk, semi[0], semi[1]) else semi[1]

    return np.bincount(champs, minlength=teams) / n_sims


def _week_winner(pts: np.ndarray, sim: int, week: int, a: int, b: int) -> bool:
    """True if `a` beats `b`. Playoff weeks reuse regular-season draws cyclically."""
    w = week % pts.shape[1]
    return bool(pts[sim, w, a] > pts[sim, w, b])
