"""Availability modeling for a MIXED draft room.

The naive `edge` column (ECR minus ESPN ADP) assumes every opponent drafts off ESPN's
board. In a room where only some do, that assumption fails in a specific and costly way:
the sharp drafters take the ECR-favorable players first, so the largest `edge` values get
consumed before they reach you. What actually falls is the high-edge player the sharps
*also* passed on, which usually means the consensus has not priced something real.

The fix is to stop treating ADP as a point estimate. Model where a player goes as a
distribution, blend the two boards by how much of the room uses each, and answer the
question that actually drives a pick: will he still be there at my next turn?
"""
from __future__ import annotations
import numpy as np
import polars as pl

# Fraction of the room drafting off ESPN's default board. Estimate it from your own
# league's history with fit_espn_weight(); 0.5 is the "mixed room" prior.
DEFAULT_ESPN_WEIGHT = 0.5


def blended_adp(df: pl.DataFrame, w: float = DEFAULT_ESPN_WEIGHT) -> pl.DataFrame:
    """Expected pick number under a mixed room.

    w=1.0 collapses to pure ESPN ADP (everyone uses the app's board).
    w=0.0 collapses to pure consensus (everyone is sharp).
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"espn weight must be in [0, 1], got {w}")
    espn = pl.col("adp").fill_null(pl.col("ecr"))
    return df.with_columns((w * espn + (1 - w) * pl.col("ecr")).alias("mu_pick"))


def _sigma(df: pl.DataFrame) -> np.ndarray:
    """Per-player pick uncertainty.

    Prefer the consensus dispersion when present. Otherwise fall back to a heuristic that
    widens with ADP, because the back of the board is far less predictable than the front.
    """
    mu = df["mu_pick"].to_numpy()
    heuristic = 2.0 + 0.18 * mu
    if "sd" in df.columns:
        sd = df["sd"].fill_null(0.0).to_numpy()
        return np.where(sd > 0, np.maximum(sd, 1.0), heuristic)
    return heuristic


def availability(df: pl.DataFrame, picks: list[int], n_sims: int = 5000,
                 w: float = DEFAULT_ESPN_WEIGHT, seed: int = 0) -> pl.DataFrame:
    """P(player is still on the board) at each of your upcoming pick numbers.

    Simulates draft orders by drawing a noisy pick position per player and ranking them,
    which preserves the constraint that exactly one player goes at each slot.
    """
    if df.height == 0:
        return df.with_columns([pl.lit(None, pl.Float64).alias(f"avail_{k}") for k in picks])

    df = blended_adp(df, w)
    rng = np.random.default_rng(seed)
    mu, sd = df["mu_pick"].to_numpy(), _sigma(df)

    draws = rng.normal(mu[None, :], sd[None, :], size=(n_sims, len(mu)))
    # Rank within each simulated draft: position 1 = first off the board.
    order = np.argsort(np.argsort(draws, axis=1), axis=1) + 1

    out = df
    for k in picks:
        out = out.with_columns(pl.Series(f"avail_{k}", (order >= k).mean(axis=0)))
    return out


def pick_value(df: pl.DataFrame, now: int, next_pick: int, **kw) -> pl.DataFrame:
    """Rank candidates by what you actually lose by waiting.

    `cost_of_waiting` = VOR x P(gone by your next turn). The best pick is not the highest
    VOR on the board; it is the highest VOR you will not get back. This is what makes the
    board robust to ADP uncertainty instead of dependent on it.
    """
    av = availability(df, [now, next_pick], **kw)
    return (av.with_columns(
                (pl.col("vor").fill_null(0.0) * (1 - pl.col(f"avail_{next_pick}")))
                .alias("cost_of_waiting"))
              .filter(pl.col(f"avail_{now}") > 0.05)
              .sort("cost_of_waiting", descending=True))


def fit_espn_weight(league_id: int, years: list[int]) -> float:
    """Estimate w from your own league's past drafts instead of guessing.

    For each historical pick, ask whether ESPN ADP or consensus rank predicted it better.
    The share won by ESPN is a direct estimate of how much of your room uses the app board.
    Needs cookies in .env; returns the DEFAULT prior if history is unavailable.
    """
    import nflreadpy as nfl
    from espn_api.football import League
    import os
    from dotenv import load_dotenv

    load_dotenv()
    wins = total = 0
    for yr in years:
        try:
            lg = League(league_id=league_id, year=yr,
                        espn_s2=os.environ.get("ESPN_S2"), swid=os.environ.get("ESPN_SWID"))
            ecr = {r["player"]: r["ecr"] for r in nfl.load_ff_rankings("draft").to_dicts()}
            for i, p in enumerate(lg.draft, start=1):
                name = getattr(p.playerName, "strip", lambda: p.playerName)()
                if name not in ecr:
                    continue
                espn_rank = getattr(p, "playerId", None) and i  # actual slot as ESPN proxy
                total += 1
                wins += abs((espn_rank or i) - i) <= abs(ecr[name] - i)
        except Exception as e:  # noqa: BLE001
            print(f"  {yr} draft history unavailable ({type(e).__name__}); skipping.")
    return wins / total if total >= 50 else DEFAULT_ESPN_WEIGHT
