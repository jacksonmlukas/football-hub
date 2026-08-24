"""Fit TALENT_CV: how wrong the market is about a player's season.

`hub/draft/season.py` carried `TALENT_CV = 0.35` under a comment calling it "the single
most important number in the model" and saying it had not been fitted.
`docs/six-of-twelve.md` then made it load-bearing -- the season-long variance sweep behind
the corrected draft-time advice is a sweep in exactly this quantity -- so it is now fitted.

**The instrument is the league's own past drafts.** A draft pick is market opinion recorded
before week 1 and cannot be revised afterwards, unlike a stored projection. So
E[realized points | pick, position] is the market's projection, fitted per position as a
power law in pick number with season intercepts, and rescaled so the market is unbiased by
construction. The spread of realized/projected around that is TALENT_CV.

Realized scoring is measured as **points per team game (total / 17)**, not per game played.
A player who misses ten weeks really did deliver close to nothing, and the simulator's
best-lineup rule benches a low-talent player the same way you bench an injured one, so
availability belongs inside the number rather than outside it.

Two corrections, both of which move the answer:

- **Weekly sampling is not talent.** A season average over ~15 games has its own spread,
  worth about 0.15 of the 0.44 raw dispersion. Counting it as talent overstates the result.
- **Vanished players are the busts.** Retired players drop out of ESPN's player universe,
  so 2022 is missing a quarter of its draft, uniformly across rounds. Fitting on it reads
  as more predictable than the season was. The headline excludes it; `--seasons` does not.

Result on 2023-25, 460 drafted skill players inside pick 168: **0.411, 95% CI
[0.384, 0.437]**, which puts the old 0.35 about 4.6 standard errors low. Only the second
moment was fitted, and the quantiles came out right on their own -- observed p10/p90 of
0.43/1.55 against 0.44/1.56 for the normal the model assumes, with skew 0.07.

    uv run python -m hub.draft.calibrate
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np
import polars as pl

# The weekly spread the model assumes, from `hub.draft.season.weekly_moments`: sd = 0.55*mu.
# The noise correction is only as good as this, and it is itself an unfitted constant --
# see the caveat in docs/talent-cv.md before reading the per-position numbers closely.
WEEKLY_CV = 0.55
TEAM_GAMES = 17
DRAFTED_THROUGH = 168          # 14 rounds x 12 teams: the roster the simulator holds
SKILL = ("QB", "RB", "WR", "TE")

# Recorded so `test_the_current_constant_is_inside_the_fitted_interval` can guard against a
# silent revert to a guessed value.
FITTED_CI95 = (0.384, 0.440)
# Shrunk, debiased per-position values behind `season.TALENT_CV_BY_POS`.
FITTED_BY_POS = {"QB": 0.410, "RB": 0.490, "WR": 0.406, "TE": 0.319}


def _curve(sub: pl.DataFrame) -> np.ndarray:
    """E[points per team game | pick] for one position: a power law with season intercepts.

    Log-log because the pick-to-points relationship is roughly a power law and a linear fit
    would put the error in the wrong place at the top of the board, which is where the
    roster's value is.
    """
    real = sub["total"].to_numpy() / TEAM_GAMES
    x = np.log(sub["pick"].to_numpy().astype(float))
    seasons = sorted(set(sub["season"].to_list()))
    cols = [np.ones_like(x), x] + [
        (sub["season"].to_numpy() == s).astype(float) for s in seasons[1:]]
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), np.log(real + 1.0), rcond=None)
    pred = np.clip(np.exp(np.column_stack(cols) @ beta) - 1.0, 0.5, None)
    # Calibrate the level: a curve that sat high or low on average would have that bias
    # counted as dispersion.
    return pred * float(np.mean(real / pred))


def simulate_seasons(mu: np.ndarray, cv: float, games: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Season totals under the model exactly as `hub.draft.season` writes it.

    Talent is multiplicative-normal and clipped at zero; weekly points are normal around
    realised talent with a spread set by the *projection*, and also clipped. Both clips
    matter -- they truncate the left tail, which is why a fitted CV comes back a few percent
    below the nominal it was generated from.
    """
    true_mu = np.clip(mu * (1.0 + rng.normal(0.0, cv, mu.size)), 0.0, None)
    total = np.zeros(mu.size)
    for w in range(int(games.max()) if games.size else 0):
        # WEEKLY_CV applies to realised talent, matching `simulate_weeks`. Keying it to the
        # projection would put a floor under busts and this fixture would stop representing
        # the model it exists to invert.
        draw = np.clip(rng.normal(true_mu, WEEKLY_CV * true_mu), 0.0, None)
        total += np.where(games > w, draw, 0.0)
    return total


def nominal_for(target: float, mu: np.ndarray, games: np.ndarray, picks: np.ndarray,
                positions: np.ndarray, seed: int = 0) -> float:
    """The TALENT_CV to put *into* the model so the model reproduces `target` coming out.

    The fit is a few percent biased low -- the model clips talent and weekly points at zero,
    and the log-log curve is fitted on the same data it is scored against. Rather than argue
    about the size of each effect, generate at a candidate nominal, run the whole fit on it,
    and search for the nominal that returns `target`. Whatever the bias is made of, this
    inverts it.
    """
    # Everyone plays a full season here even though the real players did not. That is not
    # an oversight: `simulate_weeks` has no concept of absence -- every player is drawn
    # every week and the lineup benches whoever scores least -- so availability has to be
    # carried *by* this constant rather than alongside it. Feeding the real games
    # distribution back in would let the simulation reproduce the observed dispersion using
    # missed games the model does not have, and the constant would come out too low.
    full = np.full(mu.size, TEAM_GAMES)

    def fitted(cv):
        rng = np.random.default_rng(seed)
        total = simulate_seasons(mu, cv, full, rng)
        df = pl.DataFrame({"season": np.full(mu.size, 2024), "pick": picks,
                           "pos": positions, "total": total, "games": full})
        return fit_talent_cv(df, bootstrap=1, debias=False)["talent_cv"]

    lo, hi = 0.4 * target, min(2.2 * target + 0.05, 1.5)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if fitted(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 2e-3:
            break
    return 0.5 * (lo + hi)


def rng_boot(r: np.ndarray, nv: np.ndarray, bootstrap: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the noise-corrected dispersion."""
    rng = np.random.default_rng(seed)
    return np.array([np.sqrt(max(r[i].var(ddof=1) - nv[i].mean(), 1e-9))
                     for i in (rng.integers(0, len(r), len(r)) for _ in range(bootstrap))])


def _shrink(by_pos: dict, se_pos: dict, pooled: float) -> dict:
    """Shrink each position toward the pooled value in proportion to how noisy it is.

    Four positions with 50-200 players each does not support four independent numbers: the
    spread between them is part real and part sampling error, and using the raw estimates
    treats all of it as real. So estimate how much of the spread is genuine -- the variance
    between positions net of their own sampling variance -- and move each estimate that
    fraction of the way from the pool to its raw value.

    When positions truly differ this barely moves them; when they do not, it collapses them
    onto the pool, which is the behaviour wanted in both directions.
    """
    positions = sorted(by_pos)
    dev = np.array([by_pos[p] - pooled for p in positions])
    sev = np.array([se_pos[p] ** 2 for p in positions])
    tau2 = max(float((dev ** 2).mean() - sev.mean()), 0.0)
    return {p: float(pooled + (tau2 / (tau2 + sev[i])) * dev[i]) if tau2 > 0 else pooled
            for i, p in enumerate(positions)}


def fit_talent_cv(df: pl.DataFrame, bootstrap: int = 2000, seed: int = 0,
                  debias: bool = False) -> dict:
    """Fit TALENT_CV from draft outcomes.

    `df` needs season, pick, pos, total, games -- one row per drafted player-season.

    `debias=True` additionally reports `nominal`: the value to put *into* the model so that
    the model reproduces the dispersion measured here. They differ by about 10% because the
    model clips talent and weekly points at zero. See `nominal_for`.
    """
    df = df.filter(pl.col("pos").is_in(SKILL) & (pl.col("pick") <= DRAFTED_THROUGH))
    ratios, noise, by_pos, per_pos, shape = [], [], {}, {}, {}
    for pos in SKILL:
        sub = df.filter(pl.col("pos") == pos)
        if sub.height < 20:
            continue
        real = sub["total"].to_numpy() / TEAM_GAMES
        pred = _curve(sub)
        r = real / pred
        g = sub["games"].to_numpy().astype(float)
        # Weekly sampling contribution to Var(ratio), per player. A season total is a sum of
        # `g` weekly draws, so its spread shrinks with games played and has to come off
        # player by player rather than at some average rate.
        #
        # The scale is the *projection*, not realised output, because that is what the model
        # does: `weekly_moments` sets sd = 0.55 * mu from the projection, so weekly spread
        # does not grow for a player who turns out good. Keying it to realised output
        # instead makes the correction quadratic in the outcome, over-subtracts from the
        # right tail, and biases the fit down -- visibly so at high CV, which is where the
        # recovery test caught it (0.50 came back as 0.445).
        mean_g = float(g.mean()) or 1.0
        nv = WEEKLY_CV ** 2 * g / mean_g ** 2
        by_pos[pos] = float(np.sqrt(max(r.var(ddof=1) - nv.mean(), 1e-9)))
        per_pos[pos] = (r / r.mean(), nv)
        # projected per-game points, for the debias step: pred is per *team* game
        shape[pos] = (pred * TEAM_GAMES / mean_g, g, sub["pick"].to_numpy())
        ratios.append(r)
        noise.append(nv)

    if not ratios:
        raise ValueError("no position had enough drafted players to fit")

    se_pos = {}
    for pos, (r_p, nv_p) in per_pos.items():
        rr = rng_boot(r_p, nv_p, bootstrap, seed)
        se_pos[pos] = float(rr.std())

    r = np.concatenate(ratios)
    nv = np.concatenate(noise)
    mean_ratio = float(r.mean())
    r = r / mean_ratio
    cv = float(np.sqrt(max(r.var(ddof=1) - nv.mean(), 1e-9)))

    bs = rng_boot(r, nv, bootstrap, seed)

    out_nominal, nominal_by_pos = None, {}
    if debias:
        mu = np.concatenate([shape[p][0] for p in sorted(shape)])
        gg = np.concatenate([shape[p][1] for p in sorted(shape)])
        pk = np.concatenate([shape[p][2] for p in sorted(shape)])
        ps = np.concatenate([np.full(len(shape[p][1]), p) for p in sorted(shape)])
        out_nominal = nominal_for(cv, mu, gg, pk, ps, seed=seed)
        shrunk = _shrink(by_pos, se_pos, cv)
        # Each position is inverted through the same machinery so the correction is not
        # assumed to be a constant multiple.
        for pos in sorted(shape):
            m, gp, pkp = shape[pos]
            nominal_by_pos[pos] = nominal_for(
                shrunk[pos], m, gp, pkp, np.full(len(gp), pos), seed=seed)

    return {"talent_cv": cv, "weekly": float(np.sqrt(nv.mean())),
            "nominal": out_nominal, "nominal_by_position": nominal_by_pos,
            "by_position_shrunk": _shrink(by_pos, se_pos, cv),
            "se_by_position": se_pos,
            "raw_sd": float(r.std(ddof=1)), "mean_ratio": mean_ratio,
            "by_position": by_pos, "n": int(df.height),
            "seasons": sorted(set(df["season"].to_list())),
            "ci95": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "se": float(bs.std())}


def draft_outcomes(seasons: Sequence[int], fetch=None) -> pl.DataFrame:
    """Every drafted player and what he actually scored, per season.

    Two views per season and no per-player loop: `mDraftDetail` for the picks,
    `kona_player_info` for the outcomes.
    """
    from hub.fetch import espn
    fetch = fetch or espn.league_history
    POSN = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
    rows = []
    for season in seasons:
        picks = ((fetch(season, "mDraftDetail") or {}).get("draftDetail") or {}).get("picks") or []
        pick_of = {p["playerId"]: p["overallPickNumber"] for p in picks}
        if not pick_of:
            raise ValueError(f"no draft found for {season}")
        for entry in (fetch(season, "kona_player_info") or {}).get("players") or []:
            p = entry.get("player") or {}
            if p.get("id") not in pick_of:
                continue
            st = next((s for s in p.get("stats") or []
                       if s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 0
                       and s.get("seasonId") == season), None)
            if not st or st.get("appliedTotal") is None:
                continue
            avg = st.get("appliedAverage")
            rows.append({"season": int(season), "pick": int(pick_of[p["id"]]),
                         "pos": POSN.get(p.get("defaultPositionId") or 0, "?"),
                         "total": float(st["appliedTotal"]),
                         "games": int(round(st["appliedTotal"] / avg)) if avg else 0})
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.calibrate", description="Fit TALENT_CV from past drafts.")
    ap.add_argument("--seasons", default="2023,2024,2025",
                    help="2022 is excluded by default: a quarter of its drafted players "
                         "have retired out of ESPN's universe, and they are the busts")
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",")]

    df = draft_outcomes(seasons)
    got = fit_talent_cv(df, debias=True)
    from hub.draft.season import TALENT_CV, TALENT_CV_BY_POS

    print(f"  fitted on {got['n']} drafted skill players, seasons {got['seasons']}, "
          f"picks 1-{DRAFTED_THROUGH}")
    print(f"    raw dispersion of realized/projected : {got['raw_sd']:.3f}")
    print(f"    weekly sampling, removed             : {got['weekly']:.3f}")
    print(f"    dispersion net of it                 : {got['talent_cv']:.3f}  "
          f"95% CI [{got['ci95'][0]:.3f}, {got['ci95'][1]:.3f}]")
    print(f"    nominal, i.e. what the model needs   : {got['nominal']:.3f}   "
          f"(in use: {TALENT_CV})")
    print(f"\n  {'pos':>4} {'raw':>7} {'se':>6} {'shrunk':>8} {'nominal':>8} "
          f"{'in use':>7} {'vs pool':>9}")
    for pos in ("QB", "RB", "WR", "TE"):
        d = (got["by_position"][pos] - got["talent_cv"]) / got["se_by_position"][pos]
        print(f"  {pos:>4} {got['by_position'][pos]:>7.3f} {got['se_by_position'][pos]:>6.3f} "
              f"{got['by_position_shrunk'][pos]:>8.3f} "
              f"{got['nominal_by_position'][pos]:>8.3f} "
              f"{TALENT_CV_BY_POS.get(pos, TALENT_CV):>7.2f} {d:>+8.1f}se")
    # Which positions differ is a property of the data, not a sentence to hardcode: this
    # has to stay true when it is re-run next August with another season added.
    apart = [f"{p} ({(got['by_position'][p] - got['talent_cv']) / got['se_by_position'][p]:+.1f} se)"
             for p in ("QB", "RB", "WR", "TE")
             if abs(got["by_position"][p] - got["talent_cv"]) > 2 * got["se_by_position"][p]]
    print("\n  shrunk toward the pool in proportion to noise -- 51 tight ends do not")
    print("  support an independent number. Beyond 2 se from the pool: "
          + (", ".join(apart) if apart else "none"))
    print("  see docs/talent-cv.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
