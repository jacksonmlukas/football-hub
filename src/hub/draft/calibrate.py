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

Result on 2023-25, 460 drafted player-seasons over 235 players inside pick 168: **0.408,
95% CI [0.370, 0.453]**, which still puts the old 0.35 well outside the interval. Only the
second moment was fitted, and the quantiles came out right on their own -- observed p10/p90
of 0.43/1.55 against 0.44/1.56 for the normal the model assumes, with skew 0.07.

**The interval is over the player, and the curve is refitted inside it** (issue #172). It
used to be over the player-*season*, around a curve fitted once outside the resample, with
the noise correction indexed by the same draw as the residuals -- and it read
[0.380, 0.434], 54% narrower on the same point estimate. Restated in docs/talent-cv.md.

    uv run python -m hub.draft.calibrate
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS

# The weekly spread the model assumes, from `hub.draft.season.weekly_moments`:
# sd = k * sqrt(mu), fitted in docs/weekly-spread.md. This used to be an unfitted 0.55*mu,
# which the per-game-played variant of this fit flagged by returning an implausible 0.041
# for quarterbacks -- 0.55 over-subtracts badly for the steadiest position.
from hub.draft.season import WEEKLY_K, WEEKLY_K_POOLED

NOT_FITTED_BECAUSE = (
    "the recorded output of a fit, used by tests to guard the live constants -- an assertion "
    "about predictions, not an input "
)

TEAM_GAMES = 17
DRAFTED_THROUGH = 168          # 14 rounds x 12 teams: the roster the simulator holds
# Fewest players a position needs before it gets a curve of its own. Named rather than typed
# twice: the point estimate and every bootstrap refit have to drop a thin position on the
# same rule, or the interval is around a different estimator than the one it is reported for.
MIN_PER_POSITION = 20

# Recorded so `test_the_current_constant_is_inside_the_fitted_interval` can guard against a
# silent revert to a guessed value.
#
# RE-RUN 2026-09-07 under the corrected bootstrap, issue #172: [0.380, 0.434] -> this, 54%
# wider on an unchanged point estimate of 0.408. The old interval treated the fitted curve as
# known, and refitting it inside each resample is almost the whole of the movement. Restated
# in docs/talent-cv.md rather than edited over the top of the old figure.
FITTED_CI95 = (0.370, 0.453)
# Shrunk, debiased per-position values behind `season.TALENT_CV_BY_POS`.
#
# These moved in the same re-run, and not because any raw estimate did -- all four are
# unchanged. `_shrink` reads `se_by_position` to decide how much of the spread between
# positions is real, so correcting the standard errors necessarily moves the shrunk values:
# RB 0.501 -> 0.482 and TE 0.321 -> 0.332, with QB and WR inside a thousandth. Leaving these
# pinned to a standard error the estimator no longer produces is the "estimator repaired, its
# estimate left pinned" failure docs/method.md rule 13 names.
FITTED_BY_POS = {"QB": 0.419, "RB": 0.482, "WR": 0.419, "TE": 0.332}


def _curve_of(real: np.ndarray, logpick: np.ndarray, season: np.ndarray) -> np.ndarray:
    """E[points per team game | pick] for one position: a power law with season intercepts.

    Log-log because the pick-to-points relationship is roughly a power law and a linear fit
    would put the error in the wrong place at the top of the board, which is where the
    roster's value is.

    Takes arrays rather than a frame because the bootstrap refits it thousands of times on
    resampled rows, and a resample is an index gather. Season intercepts come from the
    seasons *present* in the rows handed in, so a draw that misses a season fits the design
    it actually has rather than a column of zeros.
    """
    seasons = np.unique(season)
    cols = [np.ones_like(logpick), logpick] + [
        (season == s).astype(float) for s in seasons[1:]]
    x = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(x, np.log(real + 1.0), rcond=None)
    pred = np.clip(np.exp(x @ beta) - 1.0, 0.5, None)
    # Calibrate the level: a curve that sat high or low on average would have that bias
    # counted as dispersion.
    return pred * float(np.mean(real / pred))


def _curve(sub: pl.DataFrame) -> np.ndarray:
    """`_curve_of` for one position's rows as a frame."""
    return _curve_of(sub["total"].to_numpy() / TEAM_GAMES,
                     np.log(sub["pick"].to_numpy().astype(float)),
                     sub["season"].to_numpy())


def simulate_seasons(mu: np.ndarray, cv: float, games: np.ndarray,
                     rng: np.random.Generator, k: float = WEEKLY_K_POOLED) -> np.ndarray:
    """Season totals under the model exactly as `hub.draft.season` writes it.

    Talent is multiplicative-normal and clipped at zero; weekly points are normal around
    realised talent with a spread set by the *projection*, and also clipped. Both clips
    matter -- they truncate the left tail, which is why a fitted CV comes back a few percent
    below the nominal it was generated from.
    """
    true_mu = np.clip(mu * (1.0 + rng.normal(0.0, cv, mu.size)), 0.0, None)
    total = np.zeros(mu.size)
    for w in range(int(games.max()) if games.size else 0):
        # sd = k * sqrt(realised talent), matching `simulate_weeks`. Keying spread to the
        # projection instead would put a floor under busts and this fixture would stop
        # representing the model it exists to invert.
        draw = np.clip(rng.normal(true_mu, k * np.sqrt(true_mu)), 0.0, None)
        total += np.where(games > w, draw, 0.0)
    return total


def _k_of(positions: np.ndarray) -> float:
    """Weekly coefficient for a set of players, pooled when they are mixed."""
    uniq = {str(p) for p in positions}
    return WEEKLY_K[uniq.pop()] if len(uniq) == 1 and next(iter(uniq), None) in WEEKLY_K \
        else WEEKLY_K_POOLED


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
        total = simulate_seasons(mu, cv, full, rng, k=_k_of(positions))
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


@dataclass(frozen=True)
class Block:
    """One position's rows, as arrays, so a resample is an index gather.

    Arrays rather than a frame because the bootstrap refits everything below thousands of
    times, and a `DataFrame.filter` inside that loop cost more than the least squares it was
    feeding. `unit` is the resample unit each row belongs to -- see `cluster_rows` -- carried
    per row so a pooled draw can be split back across positions without a join.
    """
    pos: str
    real: np.ndarray        # points per team game
    pick: np.ndarray
    season: np.ndarray
    total: np.ndarray
    games: np.ndarray

    @classmethod
    def of(cls, sub: pl.DataFrame, pos: str) -> Block:
        return cls(pos=pos,
                   real=sub["total"].to_numpy() / TEAM_GAMES,
                   pick=sub["pick"].to_numpy().astype(float),
                   season=sub["season"].to_numpy(),
                   total=sub["total"].to_numpy().astype(float),
                   games=sub["games"].to_numpy().astype(float))

    def curve(self, idx: np.ndarray | None = None) -> np.ndarray:
        """E[points per team game | pick] over these rows, or the ones `idx` names."""
        return _curve_of(self.real if idx is None else self.real[idx],
                         np.log(self.pick if idx is None else self.pick[idx]),
                         self.season if idx is None else self.season[idx])


def ratio_and_noise(block: Block, idx: np.ndarray | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One position's realized/projected ratios, and each row's weekly-sampling share.

    Both are derived from the rows named by `idx`, **including the curve** -- which is the
    whole reason this takes an index. The bootstrap has to be able to re-run every step on a
    resample, and a step that reaches outside the rows it was given cannot be.

    The ratios come out with mean exactly 1 by construction: `_curve_of` rescales its
    prediction by `mean(real / pred)`, so dividing through by it again is the identity. That
    is why nothing here normalises, and why `mean_ratio` is a check on the level rather than
    a step in the fit.

    Weekly sampling contribution to Var(ratio), per player. A season total is a sum of `g`
    weekly draws, so its spread shrinks with games played and has to come off player by
    player rather than at some average rate. Under sd = k*sqrt(talent), a season total over g
    games has variance g * k^2 * talent, so as a share of the squared prediction the
    correction is linear in realised scoring rather than quadratic in it. That linearity is
    why the residual bias this leaves is small enough for `nominal_for` to mop up.
    """
    real = block.real if idx is None else block.real[idx]
    total = block.total if idx is None else block.total[idx]
    g = block.games if idx is None else block.games[idx]
    pred = block.curve(idx)
    kk = WEEKLY_K.get(block.pos, WEEKLY_K_POOLED)
    ppg = np.where(g > 0, total / np.maximum(g, 1), 0.0)
    nv = kk ** 2 * g * ppg / (TEAM_GAMES ** 2 * np.maximum(pred, 1e-6) ** 2)
    return real / pred, nv


def _dispersion(r: np.ndarray, nv: np.ndarray) -> float:
    """Dispersion of the ratios net of the weekly sampling inside them."""
    return float(np.sqrt(max(r.var(ddof=1) - nv.mean(), 1e-9)))


def cluster_rows(df: pl.DataFrame) -> np.ndarray:
    """The resample unit of each row: the **player**, as a 0-based code per row.

    The unit is the player and not the player-season, because a player drafted in three of
    these seasons is one draw from the population of drafted players and not three. His three
    outcomes are correlated -- the same talent produced all of them -- and resampling rows
    treats them as independent, which is the same understatement `hub.models.experiment`
    clusters on the season to avoid.

    A frame with no `player` column gets one unit per row, which is the truth for a frame
    holding one season per player and is what every synthetic fixture here is. It is not a
    silent fallback: `fit_talent_cv` reports `clusters` beside `n`, so a frame that lost its
    identifier reads as 460 units over 460 rows and a frame that kept it does not.

    **Coded by first appearance, not by how the identifiers happen to sort.** `np.unique`
    numbers them lexicographically, and the bootstrap indexes into that numbering -- so
    renaming the players permutes which of them each draw takes, and the percentiles move
    while the mean does not. That is the defect [weekly-blend-gate.md] records against
    `cluster_bootstrap`, one module over, and it is worth not repeating: it also made a frame
    with one season per player disagree with the same frame with the column dropped, which is
    two answers to a question that has one.
    """
    if "player" not in df.columns:
        return np.arange(df.height)
    ids = df["player"].cast(pl.Utf8).fill_null("").to_numpy()
    if not ids.size:
        return np.zeros(0, dtype=int)
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=int)
    rank[np.argsort(first)] = np.arange(first.size)
    return rank[inverse]


def _units_to_rows(unit: np.ndarray) -> list[np.ndarray]:
    """Row indices per unit, grouped by a sort rather than a scan per unit."""
    if not unit.size:
        return []
    order = np.argsort(unit, kind="stable")
    return np.split(order, np.cumsum(np.bincount(unit))[:-1])


def cluster_boot(blocks: Sequence[Block], unit: np.ndarray, bootstrap: int,
                 seed: int) -> np.ndarray:
    """Bootstrap distribution of the noise-corrected dispersion. Three corrections, #172.

    **The curve is refitted inside the resample.** It used to be fitted once, outside, and
    the bootstrap resampled the residuals that fit produced -- so the interval carried the
    spread of the residuals around a curve it treated as known, and omitted the uncertainty
    of the curve itself. The curve is estimated from the same 460 players; it is not known.

    **The unit is the player**, via `cluster_rows`, so all of one player's seasons move
    together and a player who was drafted three times is one draw rather than three.

    **The residual and the noise term come from independent draws.** The correction
    subtracted is an estimate in its own right, and indexing it by the same draw as the
    residuals ties them: a draw that happens to take high-scoring players raises `var(r)` and
    `mean(nv)` together, so their difference is steadied by the pairing rather than by the
    data. Two draws let the two estimates vary as independently as they do.

    All three push the same way, and they should: each is a source of variation the old
    interval left out.

    `blocks` are the positions to pool over and `unit` is the resample unit of every row
    across all of them, in `blocks` order -- so one draw of players is split back across
    positions by a mask rather than a join.
    """
    rng = np.random.default_rng(seed)
    rows_of = _units_to_rows(unit)
    n_units = len(rows_of)
    # Which block each global row belongs to, and where inside it.
    block_of = np.concatenate([np.full(b.real.size, i) for i, b in enumerate(blocks)]) \
        if blocks else np.zeros(0, dtype=int)
    local_of = np.concatenate([np.arange(b.real.size) for b in blocks]) \
        if blocks else np.zeros(0, dtype=int)

    def draw() -> tuple[np.ndarray, np.ndarray]:
        if not n_units:
            return np.array([0.0, 0.0]), np.array([0.0])
        rows = np.concatenate([rows_of[k]
                               for k in rng.integers(0, n_units, n_units)])
        codes = block_of[rows]
        rs, nvs = [], []
        for i, block in enumerate(blocks):
            # A draw can leave a position too thin to fit a curve on; it is dropped for that
            # draw on the same rule the point estimate drops it, or the interval would be
            # around a different estimator than the one it is reported for.
            sel = local_of[rows[codes == i]]
            if sel.size < MIN_PER_POSITION:
                continue
            r, nv = ratio_and_noise(block, sel)
            rs.append(r)
            nvs.append(nv)
        if not rs:
            return np.array([0.0, 0.0]), np.array([0.0])
        return np.concatenate(rs), np.concatenate(nvs)

    out = np.empty(bootstrap)
    for b in range(bootstrap):
        r, _ = draw()
        _, nv = draw()
        out[b] = _dispersion(r, nv)
    return out


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
    df = df.filter(pl.col("pos").is_in(DRAFTED_POSITIONS) & (pl.col("pick") <= DRAFTED_THROUGH))
    unit_all = cluster_rows(df)
    pos_all = df["pos"].to_numpy()
    ratios, noise, by_pos, shape = [], [], {}, {}
    blocks: list[Block] = []
    units: list[np.ndarray] = []
    for pos in DRAFTED_POSITIONS:
        keep = np.flatnonzero(pos_all == pos)
        if keep.size < MIN_PER_POSITION:
            continue
        block = Block.of(df[keep], pos)
        r, nv = ratio_and_noise(block)
        mean_g = float(block.games.mean()) or 1.0
        by_pos[pos] = _dispersion(r, nv)
        blocks.append(block)
        units.append(unit_all[keep])
        # projected per-game points, for the debias step: the curve is per *team* game
        shape[pos] = (block.curve() * TEAM_GAMES / mean_g, block.games, block.pick)
        ratios.append(r)
        noise.append(nv)

    if not ratios:
        raise ValueError("no position had enough drafted players to fit")

    # Each position's own interval, resampled within that position: a quarterback's seasons
    # say nothing about the spread among tight ends, so the pooled draw is the wrong unit
    # here. The unit codes are re-based per position so `bincount` stays dense.
    se_pos = {b.pos: float(cluster_boot([b], np.unique(u, return_inverse=True)[1],
                                        bootstrap, seed).std())
              for b, u in zip(blocks, units, strict=True)}

    r = np.concatenate(ratios)
    nv = np.concatenate(noise)
    mean_ratio = float(r.mean())
    cv = _dispersion(r, nv)

    pooled_unit = np.unique(np.concatenate(units), return_inverse=True)[1]
    bs = cluster_boot(blocks, pooled_unit, bootstrap, seed)

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
            # The unit the interval was resampled over, said out loud beside the row count.
            # `clusters == n` means one season per player; anything less means a player is
            # in here more than once and was drawn once.
            "clusters": int(np.unique(unit_all).size),
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
                         # ESPN's player id, and the reason it is carried: the bootstrap
                         # resamples the *player*, so a back drafted in all three seasons is
                         # one draw and not three. Without it every row is its own unit and
                         # the interval reads too narrow.
                         "player": str(p["id"]),
                         "pos": POSN.get(p.get("defaultPositionId") or 0, "?"),
                         "total": float(st["appliedTotal"]),
                         "games": round(st["appliedTotal"] / avg) if avg else 0})
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.calibrate", description="Fit TALENT_CV from past drafts.")
    ap.add_argument("--seasons", default="2023,2024,2025",
                    help="2022 is excluded by default: a quarter of its drafted players "
                         "have retired out of ESPN's universe, and they are the busts")
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",")]

    try:
        df = draft_outcomes(seasons)
    except Exception as e:
        return unavailable("hub.draft.calibrate", "your league's past drafts", e)
    got = fit_talent_cv(df, debias=True)
    from hub.draft.season import TALENT_CV, TALENT_CV_BY_POS

    print(f"  fitted on {got['n']} drafted player-seasons over {got['clusters']} players, "
          f"seasons {got['seasons']}, picks 1-{DRAFTED_THROUGH}")
    print("    resampled over the player, and the curve refitted inside each draw")
    print(f"    raw dispersion of realized/projected : {got['raw_sd']:.3f}")
    print(f"    weekly sampling, removed             : {got['weekly']:.3f}")
    print(f"    dispersion net of it                 : {got['talent_cv']:.3f}  "
          f"95% CI [{got['ci95'][0]:.3f}, {got['ci95'][1]:.3f}]")
    print(f"    nominal, i.e. what the model needs   : {got['nominal']:.3f}   "
          f"(in use: {TALENT_CV})")
    print(f"\n  {'pos':>4} {'raw':>7} {'se':>6} {'shrunk':>8} {'nominal':>8} "
          f"{'in use':>7} {'vs pool':>9}")
    for pos in DRAFTED_POSITIONS:
        d = (got["by_position"][pos] - got["talent_cv"]) / got["se_by_position"][pos]
        print(f"  {pos:>4} {got['by_position'][pos]:>7.3f} {got['se_by_position'][pos]:>6.3f} "
              f"{got['by_position_shrunk'][pos]:>8.3f} "
              f"{got['nominal_by_position'][pos]:>8.3f} "
              f"{TALENT_CV_BY_POS.get(pos, TALENT_CV):>7.2f} {d:>+8.1f}se")
    # Which positions differ is a property of the data, not a sentence to hardcode: this
    # has to stay true when it is re-run next August with another season added.
    def _se_away(p: str) -> float:
        return (got["by_position"][p] - got["talent_cv"]) / got["se_by_position"][p]

    apart = [f"{p} ({_se_away(p):+.1f} se)"
             for p in DRAFTED_POSITIONS
             if abs(got["by_position"][p] - got["talent_cv"]) > 2 * got["se_by_position"][p]]
    print("\n  shrunk toward the pool in proportion to noise -- 51 tight ends do not")
    print("  support an independent number. Beyond 2 se from the pool: "
          + (", ".join(apart) if apart else "none"))
    print("  see docs/talent-cv.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
