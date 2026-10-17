"""The roster panel's weekly distributions, swaps with their odds, and unprojected players (#479).

**Read, not modelled.** Every number here comes out of two functions that already exist and
that the rest of the repo plays on: `hub.models.predict.skewed` (the right-skewed, zero-clipped
weekly law that `mu`, `sd` and `skew` describe) and `hub.models.predict.correlated_normal`
(the teammate correlation, `TEAMMATE_RHO`, applied to the latent normals before the skew).
Nothing here adds a constant to the model; the two below are a sample size and a seed, which
change how precisely a published odds is estimated and not what it estimates.

**Why not a normal on `mu` and `sd`.** The weekly law is skewed and has a point mass at zero
(`gate_clipped_share` was 27.8% in the published coverage), so a normal puts the median in
the wrong place and the lower tail below zero. And a swap's odds depend on whether the two
players are teammates: a quarterback and his own receiver move together (QB-WR +0.232), so
the chance the one outscores the other is not the independent answer.

**Quantiles are exact, odds are sampled.** A player's own marginal is a monotone transform of a
standard normal, so his quantile at p is that transform at the normal's p-quantile -- the same
computation `hub.models.coverage` publishes its intervals with, no sampling error. A swap's
odds are a joint property of two correlated draws and have no such closed form, so they are the
share of correlated draws in which the incoming player outscores the outgoing one.

**Assumptions, stated.** The marginal for a player is his one-game law *given that he plays*:
it does not fold in the chance he does not (that is `available`), and it carries no
uncertainty about `mu` itself. Players on different NFL teams are independent; teammates are
correlated by `TEAMMATE_RHO` (only the quarterback's edges are non-zero). A tie, which the
zero clip makes possible, counts as half -- so a swap between identical distributions is 0.5,
not a hair under it.

**`p_in_beats_out` is shown, never ranked on** (ADR-0016, as the Weekly projection). The call
is made by `gain`; this is the page's answer to "how sure", beside it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from hub.declare import not_an_input
from hub.models import predict

# 20 quantiles at (i + 0.5) / 20 -- the 20-dot plot's positions.
N_QUANTILES = not_an_input(20, "how many dots the panel draws; the marginal is exact at any p")
# Joint draws behind one odds. At 20,000 the standard error of a probability is at most 0.0035.
N_DRAWS = not_an_input(20_000, "the sample size of a published odds; sets its precision, not "
                               "the law it estimates")
SEED = not_an_input(479, "a fixed seed, so the same sync publishes the same odds")

# Positions the board does not project (`hub.season.roster`): said, not left silent.
OUT_OF_SCOPE = frozenset({"K", "D/ST", "DST", "DEF"})


def _skew(df: pl.DataFrame) -> np.ndarray:
    """The row's `skew` where the parquet carries it, else the position's fitted value."""
    from_pos = predict.weekly_skew_for(df["pos"].to_numpy())
    if "skew" not in df.columns:
        return from_pos
    col = df["skew"].cast(pl.Float64).to_numpy()
    return np.where(np.isfinite(col), col, from_pos)


def _playing(df: pl.DataFrame) -> pl.Series:
    """Rows with a distribution: projected, a finite mean and spread, and a game this week."""
    ok = pl.col("projected") & pl.col("mu").is_not_null() & pl.col("sd").is_not_null()
    if "on_bye" in df.columns:
        ok = ok & ~pl.col("on_bye")
    return df.select(ok.fill_null(False).alias("ok"))["ok"]


def quantile_levels(n: int = N_QUANTILES) -> list[float]:
    return [(i + 0.5) / n for i in range(n)]


def quantiles(df: pl.DataFrame) -> dict[str, list[float]]:
    """Player -> the 20 weekly quantiles, for every row that has a distribution."""
    ok = _playing(df).to_numpy()
    if not ok.any():
        return {}
    mu = df["mu"].cast(pl.Float64).fill_null(0.0).to_numpy()[ok]
    sd = df["sd"].cast(pl.Float64).fill_null(0.0).to_numpy()[ok]
    sk = _skew(df)[ok]
    inv = NormalDist().inv_cdf
    z = np.array([inv(p) for p in quantile_levels()])
    q = predict.skewed(mu[:, None], sd[:, None], sk[:, None], z[None, :])
    names = df["player"].to_numpy()[ok]
    return {str(n): [round(float(v), 2) for v in row] for n, row in zip(names, q, strict=True)}


def quantile_reason(row: Mapping[str, Any]) -> str | None:
    """Why a row has no quantiles, or None when it has them."""
    if not row.get("projected") or row.get("mu") is None or row.get("sd") is None:
        return unprojected_reason(row)
    if row.get("on_bye"):
        return "bye: no game this week"
    return None


def unprojected_reason(row: Mapping[str, Any]) -> str:
    if str(row.get("pos")) in OUT_OF_SCOPE:
        return (f"{row.get('pos')}: outside the board's scope (QB, RB, WR, TE), "
                "so there is no projection to total")
    return "the board carries no projection for this player"


def unprojected(df: pl.DataFrame) -> tuple[list[str], dict[str, str]]:
    """The rostered names with no projection, and why each has none."""
    rows = [r for r in df.iter_rows(named=True)
            if not r.get("projected") or r.get("mu") is None]
    names = [r["player"] for r in rows]
    return names, {r["player"]: unprojected_reason(r) for r in rows}


def pair_swaps(sit: Sequence[str], start: Sequence[str],
               pos_of: Mapping[str, str]) -> list[tuple[str | None, str | None]]:
    """`sit` and `start` are two unordered lists; a swap is one out and one in.

    Pair by position first, then the leftovers in order (a flex slot swaps across positions),
    and a side with nothing to pair against is None. The page's `pairSwaps`, moved here.
    """
    ins = list(start)
    rest: list[str] = []
    pairs: list[tuple[str | None, str | None]] = []
    for out in sit:
        pos = pos_of.get(out, "")
        i = next((k for k, n in enumerate(ins) if pos_of.get(n, "") == pos), -1) if pos else -1
        if i >= 0:
            pairs.append((out, ins.pop(i)))
        else:
            rest.append(out)
    while rest or ins:
        pairs.append((rest.pop(0) if rest else None, ins.pop(0) if ins else None))
    return pairs


def _joint_draws(df: pl.DataFrame, n: int, seed: int) -> tuple[dict[str, int], np.ndarray]:
    """Correlated skewed weekly draws for every player with a distribution: (index, n x m)."""
    ok = _playing(df).to_numpy()
    sub = df.filter(pl.Series(ok))
    mu = sub["mu"].cast(pl.Float64).to_numpy()
    sd = sub["sd"].cast(pl.Float64).to_numpy()
    rng = np.random.default_rng(seed)
    z = predict.correlated_normal(rng, (n, sub.height), sub["pos"].to_numpy(),
                                  sub["nfl_team"].to_numpy())
    x = predict.skewed(mu[None, :], sd[None, :], _skew(df)[ok][None, :], z)
    return {str(p): i for i, p in enumerate(sub["player"].to_list())}, x


def swaps(df: pl.DataFrame, sit: Sequence[str], start: Sequence[str],
          withheld: Sequence[str], *, n: int = N_DRAWS,
          seed: int = SEED) -> tuple[list[dict[str, Any]], str | None]:
    """The recommended swaps: `out`, `in`, `gain`, `forced`, `p_in_beats_out`.

    `gain` is the in-player's `mu` less the out-player's. `forced` is the out-player being
    withheld as unavailable, which is no choice and so carries no odds (null). Odds are also
    null, with the reason returned beside the list, when a side is missing or has no
    distribution, or the correlated draw would not run.
    """
    pos_of = dict(zip(df["player"].to_list(), df["pos"].to_list(), strict=True))
    mu_of = dict(zip(df["player"].to_list(), df["mu"].to_list(), strict=True))
    barred = set(withheld)
    pairs = pair_swaps(sit, start, pos_of)
    reason: str | None = None
    index: dict[str, int] = {}
    x = np.empty((0, 0))
    if any(o is not None and i is not None and o not in barred for o, i in pairs):
        try:
            index, x = _joint_draws(df, n, seed)
        except predict.CorrelationVoid as e:      # graceful: the list stands, the odds do not
            reason = f"odds unavailable: {e}"
    out_rows: list[dict[str, Any]] = []
    for o, i in pairs:
        forced = o is not None and o in barred
        mo, mi = mu_of.get(o) if o else None, mu_of.get(i) if i else None
        p = None
        if not forced and o in index and i in index:
            a, b = x[:, index[i]], x[:, index[o]]
            p = round(float((a > b).mean() + 0.5 * (a == b).mean()), 4)
        out_rows.append({"out": o, "in": i,
                         "gain": None if mo is None or mi is None else mi - mo,
                         "forced": forced, "p_in_beats_out": p})
    return out_rows, reason
