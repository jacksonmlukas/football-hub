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

# Sigma floor. Even the first overall pick is not perfectly predictable, and a zero or
# negative sigma makes the availability simulation degenerate.
MIN_SIGMA = 1.0


def blended_adp(df: pl.DataFrame, w: float = DEFAULT_ESPN_WEIGHT) -> pl.DataFrame:
    """Expected pick number under a mixed room.

    w=1.0 collapses to pure ESPN ADP (everyone uses the app's board).
    w=0.0 collapses to pure consensus (everyone is sharp).
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"espn weight must be in [0, 1], got {w}")
    # A *missing* adp column, not merely a null one. ECR-only mode drops it entirely rather
    # than nulling it, and a historical board never has it at all -- ESPN publishes ADP for
    # the current season only. `fill_null` needs the column to exist, so without this the
    # board that degrades most gracefully everywhere else raises here.
    #
    # Note this makes `w` a no-op on such a board: espn falls back to ecr, so
    # `w*ecr + (1-w)*ecr` is ecr for every weight. That is the correct reading of "half the
    # room drafts off ESPN" when there is no ESPN board to draft off.
    espn = (pl.col("adp").fill_null(pl.col("ecr")) if "adp" in df.columns
            else pl.col("ecr"))
    return df.with_columns((w * espn + (1 - w) * pl.col("ecr")).alias("mu_pick"))


# How far a real pick strays from consensus, fitted on 734 picks across this league's
# 2022-25 drafts by `fit_pick_noise`. It replaces a heuristic of 2.0 + 0.18*mu that had
# never been checked against a draft, and the two disagree where it matters: at ADP 100 the
# heuristic says sigma is 20 and the fit says 26. Being over-confident about who survives
# inflates cost_of_waiting and pushes the board toward taking a player now who would in fact
# have lasted.
PICK_NOISE_INTERCEPT = 1.00
PICK_NOISE_SLOPE = 0.253


def pick_noise(mu):
    """Standard deviation of where a player actually goes, given consensus `mu`."""
    return PICK_NOISE_INTERCEPT + PICK_NOISE_SLOPE * mu


def _sigma(df: pl.DataFrame) -> np.ndarray:
    """Per-player pick uncertainty.

    Prefer the consensus dispersion when present. Otherwise fall back to a heuristic that
    widens with ADP, because the back of the board is far less predictable than the front.
    """
    mu = df["mu_pick"].to_numpy()
    heuristic = pick_noise(mu)
    # `ecr_sd`, not `sd`. This is the spread of the *consensus rank*, in picks. It was
    # called `sd` on the board, which is also what a player's weekly points spread is
    # called; reading the wrong one here would have produced a confident, plausible and
    # entirely wrong availability curve, since both are small positive floats.
    if "ecr_sd" in df.columns:
        sd = df["ecr_sd"].fill_null(0.0).to_numpy()
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


def historical_picks(league_id: int, years: list[int]) -> pl.DataFrame:
    """Actual pick number and contemporaneous ECR for every pick in past drafts.

    ECR is taken from the last scrape before that season opened, so it is what the room
    could actually have seen -- not hindsight rankings.
    """
    import os

    import nflreadpy as nfl
    from dotenv import load_dotenv
    from espn_api.football import League

    from hub.draft.state import _norm

    load_dotenv()
    allr = nfl.load_ff_rankings("all")
    rows = []
    for yr in years:
        try:
            lg = League(league_id=league_id, year=yr,
                        espn_s2=os.environ.get("ESPN_S2") or None,
                        swid=os.environ.get("ESPN_SWID") or None)
            snap = (allr.filter((pl.col("page_type") == "redraft-overall")
                                # scrape_date is an ISO string, which sorts correctly
                                # as text; casting it just to compare would be waste.
                                & (pl.col("scrape_date") < f"{yr}-09-01")
                                & (pl.col("ecr").is_not_null()))
                        .sort("scrape_date", descending=True)
                        .unique(subset=["player"], keep="first"))
            ecr = {_norm(r["player"]): r["ecr"] for r in snap.iter_rows(named=True)}
            for i, pick in enumerate(lg.draft or [], start=1):
                e = ecr.get(_norm(pick.playerName))
                if e is not None:
                    rows.append({"year": yr, "pick": float(i), "ecr": float(e)})
        except Exception as exc:
            print(f"  {yr} draft history unavailable ({type(exc).__name__}); skipping.")
    return pl.DataFrame(rows, schema={"year": pl.Int64, "pick": pl.Float64,
                                      "ecr": pl.Float64})


def fit_pick_noise(league_id: int, years: list[int],
                   default: tuple[float, float] = (2.0, 0.18)) -> tuple[float, float]:
    """Fit sigma(mu) = a + b*mu from how far your room's actual picks stray from ECR.

    This is the half of the availability model that IS identifiable from league history.
    The heuristic it replaces (2.0 + 0.18*mu) was never checked against a real draft, and
    sigma drives every availability probability, so a wrong slope quietly mis-prices the
    whole board.
    """
    df = historical_picks(league_id, years)
    if df.height < 50:
        print(f"  only {df.height} matched historical picks; keeping the {default} prior.")
        return default
    mu = df["ecr"].to_numpy()
    # Absolute deviation of a normal is sigma*sqrt(2/pi); rescale to a real sigma.
    sigma_hat = np.abs(df["pick"].to_numpy() - mu) * np.sqrt(np.pi / 2.0)
    b, a = (float(x) for x in np.polyfit(mu, sigma_hat, 1))

    # An unconstrained line through this data wants a negative intercept -- early picks
    # are near-deterministic, so the fit pays for its slope by going below zero at the
    # top of the board. Sigma cannot be negative, so pin the intercept at a floor and
    # refit the slope through it rather than throwing an informative slope away.
    if a < MIN_SIGMA:
        a = MIN_SIGMA
        b = float(np.dot(mu, np.maximum(sigma_hat - a, 0.0)) / np.dot(mu, mu))
    if not (0.0 < a < 50.0 and 0.0 <= b < 2.0):
        print(f"  fitted noise (a={a:.2f}, b={b:.3f}) is out of range; keeping {default}.")
        return default
    print(f"  fitted pick noise from {df.height} picks: sigma = {a:.2f} + {b:.3f} * mu")
    return a, b


def fit_espn_weight(league_id: int, years: list[int]) -> float:
    """Not identifiable from available data. Returns the prior, loudly.

    The intent was to ask, for each historical pick, whether ESPN ADP or consensus rank
    predicted it better, and read w off the share ESPN won. That requires ESPN's ADP *as
    it stood before those drafts*, and ESPN does not keep it: querying a past season
    returns the saturation sentinel for every player (verified Aug 2026 -- Chase, Bijan
    and Jefferson all come back 170.0 for 2025). With one side of the comparison
    unavailable, w cannot be estimated this way.

    The previous implementation did not fail, it returned ~1.0 every time -- a claim that
    the entire room drafts off ESPN's board, which is the opposite of this league's
    premise and would have driven every availability number. Returning the documented
    prior and saying so is strictly better than a confident wrong answer.

    If you want w fitted, the missing ingredient is a stored snapshot of ESPN ADP taken
    *before* each draft. Since 2026-08-25 `hub.draft.adp_history` writes one on every
    successful board build, so this becomes a real fit next season rather than this
    docstring again. Meanwhile fit_pick_noise() calibrates the other half of the model from
    real history.
    """
    print("  fit_espn_weight: historical ESPN ADP is unavailable (ESPN returns the "
          f"undrafted sentinel for past seasons); using the {DEFAULT_ESPN_WEIGHT} prior.")
    return DEFAULT_ESPN_WEIGHT
