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
FITTED_CI95 = (0.384, 0.437)


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


def fit_talent_cv(df: pl.DataFrame, bootstrap: int = 2000, seed: int = 0) -> dict:
    """Fit TALENT_CV from draft outcomes.

    `df` needs season, pick, pos, total, games -- one row per drafted player-season.
    """
    df = df.filter(pl.col("pos").is_in(SKILL) & (pl.col("pick") <= DRAFTED_THROUGH))
    ratios, noise, by_pos = [], [], {}
    for pos in SKILL:
        sub = df.filter(pl.col("pos") == pos)
        if sub.height < 20:
            continue
        real = sub["total"].to_numpy() / TEAM_GAMES
        pred = _curve(sub)
        r = real / pred
        g = sub["games"].to_numpy().astype(float)
        # Weekly sampling contribution to Var(ratio), per player: a season total is a sum
        # of `g` weekly draws, so its spread shrinks with games played and must be removed
        # player by player rather than on average.
        ppg = np.where(g > 0, sub["total"].to_numpy() / np.maximum(g, 1), 0.0)
        nv = (WEEKLY_CV ** 2 * g / TEAM_GAMES ** 2) * (ppg / pred) ** 2
        by_pos[pos] = float(np.sqrt(max(r.var(ddof=1) - nv.mean(), 1e-9)))
        ratios.append(r)
        noise.append(nv)

    if not ratios:
        raise ValueError("no position had enough drafted players to fit")

    r = np.concatenate(ratios)
    nv = np.concatenate(noise)
    mean_ratio = float(r.mean())
    r = r / mean_ratio
    cv = float(np.sqrt(max(r.var(ddof=1) - nv.mean(), 1e-9)))

    rng = np.random.default_rng(seed)
    bs = np.array([np.sqrt(max(r[i].var(ddof=1) - nv[i].mean(), 1e-9))
                   for i in (rng.integers(0, len(r), len(r)) for _ in range(bootstrap))])

    return {"talent_cv": cv, "weekly": float(np.sqrt(nv.mean())),
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
    got = fit_talent_cv(df)
    from hub.draft.season import TALENT_CV

    print(f"  fitted on {got['n']} drafted skill players, seasons {got['seasons']}, "
          f"picks 1-{DRAFTED_THROUGH}")
    print(f"    raw dispersion of realized/projected : {got['raw_sd']:.3f}")
    print(f"    weekly sampling, removed             : {got['weekly']:.3f}")
    print(f"    TALENT_CV                            : {got['talent_cv']:.3f}  "
          f"95% CI [{got['ci95'][0]:.3f}, {got['ci95'][1]:.3f}]")
    print(f"    in use                               : {TALENT_CV}"
          f"  ({(TALENT_CV - got['talent_cv']) / got['se']:+.1f} se)")
    print("  by position:")
    for pos, v in got["by_position"].items():
        print(f"    {pos:<3} {v:.3f}")
    print("  a single scalar is a compromise -- RB is materially the least predictable "
          "position and TE the most; see docs/talent-cv.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
