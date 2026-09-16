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

import re
from collections.abc import Sequence
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from hub.declare import chosen, fitted, not_an_input
from hub.league import preseason_start
from hub.names import player_key

if TYPE_CHECKING:                  # `board` imports this module, so runtime would cycle
    from hub.draft.board import BuildReport

# Fraction of the room drafting off ESPN's default board. Estimate it from your own
# league's history with fit_espn_weight(); 0.5 is the "mixed room" prior.
DEFAULT_ESPN_WEIGHT = chosen(0.5)

# Sigma floor. Even the first overall pick is not perfectly predictable, and a zero or
# negative sigma makes the availability simulation degenerate.
MIN_SIGMA = chosen(1.0)


def blended_adp(df: pl.DataFrame, w: float = DEFAULT_ESPN_WEIGHT, *,
                report: BuildReport | None = None) -> pl.DataFrame:
    """Expected pick number under a mixed room.

    w=1.0 collapses to pure ESPN ADP (everyone uses the app's board).
    w=0.0 collapses to pure consensus (everyone is sharp).

    **Whether there is a draft market to blend in is the report's answer.** ECR-only mode
    drops `adp` entirely rather than nulling it, and a historical board never has it at all
    -- ESPN publishes ADP for the current season only -- so this function has to know which
    of those it is holding. It used to ask `df.columns`, which is one of the eleven private
    answers issue #199 collected; `report.adp` is the recorded one, and `board.report_for`
    derives it when a caller has only the frame.

    Note that a board with no draft market makes `w` a no-op: espn falls back to ecr, so
    `w*ecr + (1-w)*ecr` is ecr for every weight. That is the correct reading of "half the
    room drafts off ESPN" when there is no ESPN board to draft off.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"espn weight must be in [0, 1], got {w}")
    from hub.draft.board import report_for
    espn = (pl.col("adp").fill_null(pl.col("ecr")) if report_for(df, report).adp
            else pl.col("ecr"))
    return df.with_columns((w * espn + (1 - w) * pl.col("ecr")).alias("mu_pick"))


# How far a real pick strays from consensus: sigma(pick) = a + b * pick, fitted by
# `fit_pick_noise` on **672 picks inside the 204-pick draftable pool** of this league's four
# drafts, 2022-25. That is the population `_sigma` evaluates -- an expected pick number -- and
# it is strictly smaller than the 732 matched picks those four drafts supply, because a
# consensus rank past the last pick anyone ever made is not a pick number at all.
#
# ------------------------------------------------------------------------------------------
# **Restated 2026-09-07 (#150), from a re-run of the repaired fitter.** The superseded figures,
# kept as they stood: **sigma = 1.00 + 0.253 * mu**, described here as "fitted on 734 picks
# across this league's 2022-25 drafts", replacing "a heuristic of 2.0 + 0.18*mu ... at ADP 100
# the heuristic says sigma is 20 and the fit says 26".
#
# **The cause.** `fit_pick_noise` carried two defects, both inflating the slope, and `02488c0`
# fixed both: it fitted an `ecr` rank over a 300-plus-player consensus list while `_sigma`
# applies the result to `mu_pick`, an expected pick number; and it refitted through
# `max(sigma_hat - a, 0)`, which zeroes every residual under the pinned intercept and so fits
# the slope to the upper envelope of the data. Nothing re-ran the fit after that repair, and
# `test_the_noise_law_is_the_fitted_one` asserted 1.00/0.253 -- so a repaired estimator and the
# estimate it was built to replace sat here disagreeing, with the assertion holding them apart.
# `docs/method.md` rule 13 names this case as one of its three.
#
# **1.00 was not a measurement.** It is exactly `MIN_SIGMA`. `_constrained` pins the intercept
# there whenever the unconstrained line wants to go negative, so that figure was the floor
# speaking rather than the data. The re-run's intercept clears the floor, so the constraint no
# longer binds and the number below is identified.
#
# **The re-run.** `fit_pick_noise(league, [2022, 2023, 2024, 2025])` on 2026-09-07: 732 of 792
# picks matched a rank scraped inside their own preseason, 672 of those inside a 204-pick pool,
# over four drafts. It returns a = 1.3127, b = 0.16860, slope 95% CI [0.159, 0.179] clustered
# on the draft -- rounded below to the precision the fitter prints.
#
# **The direction reverses, and that is the finding.** What stood here argued the fit *widened*
# a prior that was over-confident about who survives. The repaired fit is narrower than that
# prior at every pick in the pool: at pick 100 sigma is 18.2, against the prior's 20.0 and the
# superseded fit's 26.3. So availability falls, `cost_of_waiting` rises, and the correction
# pushes the board toward scarcity rather than away from it -- the opposite of what it said.
# A board built either side of it is in `docs/pick-noise.md`, with the players who moved.
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# **Restated 2026-09-11 (#155).** The 1.31 / 0.169 above stand as the record of the fit over
# the whole 204-pick pool. They are superseded because the sample is conditioned on being
# *drafted*, and at the back of the pool that conditioning is one-sided: a player ranked 190
# who went undrafted has no pick number and is absent, so the players who remain at that rank
# are exactly the ones who beat it. Measured on the 672 picks, as the mean **signed** deviation
# `pick - ecr` by rank bin -- which is zero under no censoring, and turns negative where only
# early-goers survive the pool boundary:
#
#     ecr    1-145   within +/-3 at every bin      (n = 545)
#     ecr  145-169   -5.1                          (n =  71)
#     ecr  170-192  -22.8   = the absolute mean    (n =  37)
#     ecr  193-204  -36.2   = the absolute mean    (n =  19)
#
# In the last two bins every observed player went earlier than his rank -- the signed mean
# *equals* the absolute mean, which is what a sample containing only one tail looks like.
# Those rows were fitting the slope to a one-sided residual and inflating it.
#
# So the fit is restricted to `ecr <= PICK_NOISE_FIT_CEILING`, the last bin before the
# signature appears, and stated rather than silently applied. Refit on the same 732 matched
# picks, clustered on the draft, with the ceiling swept so the choice is visible:
#
#     ceiling   n     a      slope   95% CI            sigma @ pick 100
#       204    672   1.31   0.169   [0.159, 0.179]     18.2   (superseded)
#       168    612   2.51   0.150   [0.143, 0.156]     17.5   <- this
#       145    545   2.83   0.143   [0.116, 0.170]     17.1
#       120    460   2.33   0.156   [0.138, 0.174]     17.9
#
# **Withdrawn 2026-09-13 (#287): the claim that the 204 and 168 fits are "resolvably different".**
# What stood here read non-overlap of those two intervals as the censored tail's effect
# being resolvable at two standard errors. It is not evidence of a difference at any level:
# the two fits are nested subsets of the same four drafts sharing 612 of 672 rows (91%), so
# their bootstrap resamples move together and the intervals are not two independent
# measurements of one quantity. What the interval *can* say is stated by `noise_from_picks`
# on every fit: four clusters give 4^4 = 256 ordered resamples (35 distinct multisets), so
# the 2,000 draws revisit at most 256 of them, and the band is what four observations of
# this league's drafting support -- no more. The ceiling of 168 is therefore stated as an
# **assumption** -- the last bin before the one-sided signature in the signed-deviation table
# -- with its sensitivity beside it rather than a finding the interval established.
#
# The sweep #287 asked for, {120, 144, 168, 192, 216}, runs as
# `uv run python scripts/fit_pick_noise.py --ceilings 120,144,168,192,216` and needs an ESPN
# session (the drafts live nowhere else, `historical_picks`). The 2026-09-13 lane had no
# session, so the cells it could not fill are marked; the 120 and 168 rows are the 2026-09-11
# sweep's, 216 collapses to the 204-pick pool (`min(pool, ceiling)`) and is that row:
#
#     ceiling   n     a      slope   95% CI            sigma @ pick 100
#       120    460   2.33   0.156   [0.138, 0.174]     17.9
#       144    not established (2026-09-11 ran 145: 545, 2.83, 0.143, [0.116, 0.170], 17.1)
#       168    612   2.51   0.150   [0.143, 0.156]     17.5   <- shipped
#       192    not established
#       216    672   1.31   0.169   [0.159, 0.179]     18.2   (= the 204-pick pool)
#
# Across the cells that exist the slope spans 0.143-0.169 and sigma at pick 100 spans
# 17.1-18.2, about one pick; whether 192 sits on a cliff between 168 and the pool is the
# cell the sweep is for. `docs/pick-noise.md` carries the same table.
#
# **The axis, stated correctly this time.** The fit reads `ecr`. `_sigma` applies it to
# `mu_pick`, and for the historical drafts the two are the *same number*: no draft market
# exists for a past preseason (ADR-0010), so `blended_adp`'s `w * adp + (1 - w) * ecr` falls
# back to `ecr` on both sides and the blend is the identity. The fitted axis and the applied
# axis coincide on the population the fit uses. They differ on the *live* board, where ADP
# exists -- and that is a limitation this fit carries, not one it can remove, because no
# historical ADP exists to fit on. The docstring that said "fitted on a pick number" was
# describing the intent and not the code.
#
# Modelling the censoring instead (a Tobit on the pick axis) was declined: it needs a
# counterfactual pick number for players never picked, which the data does not contain, so
# the model would be identified by its functional form rather than by evidence -- a fitted
# constant with no provenance, which ADR-0006 exists to prevent.
# ------------------------------------------------------------------------------------------
# Fitted on this league's drafts of the seasons the draft backtest replays --
# `backtest.LIMITATIONS`, last entry (#279).
PICK_NOISE_INTERCEPT = fitted(2.51)
PICK_NOISE_SLOPE = fitted(0.150)

# Beside the point estimate, never behind it. Bootstrapped over **drafts**, n = 4, not over
# picks: one manager reaching in round two moves every later pick in that room, so a
# pick-level interval would be several times too tight -- the same error `docs/gate-power.md`
# is about one layer up. `noise_from_picks` prints this pair on every fit.
PICK_NOISE_SLOPE_CI = fitted((0.143, 0.156))

# The rank past which the drafted sample is one-sided -- see the table above. A fitted
# constant in the ADR-0006 sense: it was chosen from the data, it decides the slope, and it
# is declared `chosen` so it moves the digest when it moves. Since #287 it is stated as an
# assumption with the ceiling sweep beside it, not as a cut the interval established.
PICK_NOISE_FIT_CEILING = chosen(168)

# The sweep #287 asked for. `sweep_ceilings` runs it; `scripts/fit_pick_noise.py` prints it.
PICK_NOISE_SWEEP_CEILINGS: tuple[int, ...] = not_an_input(
    (120, 144, 168, 192, 216),
    "the axis the ceiling sweep runs over; no prediction is computed at any ceiling but "
    "the shipped one, so the list moves nothing a board serves")


def pick_noise(mu):
    """Standard deviation of where a player actually goes, given consensus `mu`."""
    return PICK_NOISE_INTERCEPT + PICK_NOISE_SLOPE * mu


def _sigma(df: pl.DataFrame) -> np.ndarray:
    """Per-player pick uncertainty.

    Prefer the consensus dispersion when present. Otherwise fall back to a heuristic that
    widens with ADP, because the back of the board is far less predictable than the front.
    """
    mu = df["mu_pick"].to_numpy()
    base = pick_noise(mu)
    # `ecr_sd`, not `sd`. This is the spread of the *consensus rank*, in picks. It was
    # called `sd` on the board, which is also what a player's weekly points spread is
    # called; reading the wrong one here would have produced a confident, plausible and
    # entirely wrong availability curve, since both are small positive floats.
    #
    # It **widens** the base rather than replacing it (#41). Replacing meant a player the
    # experts happen to agree about was priced as more predictable than the base model says
    # anyone at his rank is -- so expert consensus, which is not evidence about how this room
    # drafts, could make a pick look safer than any measurement supports. Widening is the
    # weaker and truer statement: disagreement can only add uncertainty.
    #
    # `maximum`, not quadrature. The base is fitted from where picks actually landed relative
    # to consensus, so it already carries whatever disagreement contributed to that scatter;
    # adding the two in variance would count it twice.
    if "ecr_sd" in df.columns:
        sd = df["ecr_sd"].fill_null(0.0).to_numpy()
        return np.maximum(base, sd)
    return base


def availability(df: pl.DataFrame, picks: list[int], n_sims: int = 5000,
                 w: float = DEFAULT_ESPN_WEIGHT, seed: int = 0, *,
                 report: BuildReport | None = None) -> pl.DataFrame:
    """P(player is still on the board) at each of your upcoming pick numbers.

    Simulates draft orders by drawing a noisy pick number per player and ranking them,
    which preserves the constraint that exactly one player goes at each slot.

    `report` is carried rather than used: `blended_adp` below is what needs it, and a
    caller holding one should not have to reach past this function to hand it over.
    """
    if df.height == 0:
        return df.with_columns([pl.lit(None, pl.Float64).alias(f"avail_{k}") for k in picks])

    df = blended_adp(df, w, report=report)
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

    from dotenv import load_dotenv
    from espn_api.football import League

    from hub.fetch.nflverse import RANKINGS_COLS, load_rankings

    load_dotenv()
    # Routed (#36), and keyed on the contract's own column set so this and the board's dated
    # read share one cache entry rather than two half-filled copies of a 1.8M-row table.
    allr = load_rankings("all", cols=RANKINGS_COLS)
    rows = []
    for yr in years:
        try:
            lg = League(league_id=league_id, year=yr,
                        espn_s2=os.environ.get("ESPN_S2") or None,
                        swid=os.environ.get("ESPN_SWID") or None)
            # Bounded below by the season's own preseason, like `board.consensus` (#38).
            # Without it a pick whose only rank came from an earlier preseason was matched
            # and fitted as though the room had priced him that year -- so the fit learned
            # from ranks nobody in that draft could see.
            names = [p.playerName for p in (lg.draft or [])]
            fitted, said = picks_against_preseason(allr, yr, names)
            rows.extend(fitted)
            print(said)
        except Exception as exc:
            print(f"  {yr} draft history unavailable ({type(exc).__name__}); skipping.")
    return pl.DataFrame(rows, schema={"year": pl.Int64, "pick": pl.Float64,
                                      "ecr": pl.Float64})


def picks_against_preseason(allr: pl.DataFrame, year: int,
                            names: Sequence[str]) -> tuple[list[dict], str]:
    """Draft picks paired with the ECR their own preseason published, and what to say about it.

    **Bounded below as well as above (#38).** Taking the latest scrape per player before the
    draft and nothing more matched a pick whose only rank came from an earlier preseason, and
    fitted it as though the room had priced him that year -- so the fit learned from ranks
    nobody in that draft could see. `hub.draft.board.consensus` is bounded the same way and by
    the same rule.

    The count is returned rather than printed for the reason `stamped_for_publication` is:
    reaching this through `historical_picks` needs an ESPN session, and a rule that can only
    be exercised behind a credential is a rule with no test.
    """
    opens = preseason_start(f"{year}-09-01")
    snap = (allr.filter((pl.col("page_type") == "redraft-overall")
                        # scrape_date is an ISO string, which sorts correctly as text;
                        # casting it just to compare would be waste.
                        & (pl.col("scrape_date") < f"{year}-09-01")
                        & (pl.col("scrape_date") >= opens)
                        & (pl.col("ecr").is_not_null()))
                .sort("scrape_date", descending=True)
                .unique(subset=["player"], keep="first"))
    ecr = {player_key(r["player"]): r["ecr"] for r in snap.iter_rows(named=True)}
    rows = [{"year": year, "pick": float(i), "ecr": float(ecr[player_key(n)])}
            for i, n in enumerate(names, start=1) if player_key(n) in ecr]
    # Said, because the bound lowers it: a pick whose only rank predates this season's
    # preseason is unmatched now rather than fitted on a stale ECR, and a fit that silently
    # learns from fewer picks is a fit nobody can check.
    said = (f"  {year}: {len(rows)}/{len(names)} picks matched a rank scraped since {opens}")
    return rows, said


def fit_pick_noise(league_id: int, years: list[int],
                   default: tuple[float, float] = (2.0, 0.18)) -> tuple[float, float]:
    """Fit sigma(mu) = a + b*mu from how far your room's actual picks stray from ECR.

    This is the half of the availability model that IS identifiable from league history.
    The heuristic it replaces (2.0 + 0.18*mu) was never checked against a real draft, and
    sigma drives every availability probability, so a wrong slope quietly mis-prices the
    whole board.
    """
    df = historical_picks(league_id, years)
    fitted, said = noise_from_picks(df, default)
    print(said)
    return fitted


def _constrained(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`sigma = a + b*x`, least squares, with `a` pinned at `MIN_SIGMA` if it would go below.

    An unconstrained line through this data wants a negative intercept: early picks are
    near-deterministic, so the fit pays for its slope by going below zero at the top of the
    board. Sigma cannot be negative.

    **Pinned and refitted over every observation, not over the ones above the floor.** The
    previous version refitted through `max(sigma_hat - a, 0)`, which zeroes every residual
    below the pinned intercept instead of letting it pull the slope down -- so the slope was
    fitted to the upper envelope of the data and came out too steep (#40). The residual for a
    point below the floor is negative and belongs in the sum.
    """
    b, a = (float(v) for v in np.polyfit(x, y, 1))
    if a < MIN_SIGMA:
        a = MIN_SIGMA
        b = float(np.dot(x, y - a) / np.dot(x, x))
    return a, b


def _distinct_resamples(k: int) -> tuple[int, int]:
    """Ordered and distinct cluster resamples of `k` clusters with replacement: `k^k`, and
    the multisets `C(2k-1, k)`. What a bootstrap over four drafts can visit."""
    return k ** k, comb(2 * k - 1, k)


def noise_from_picks(df: pl.DataFrame, default: tuple[float, float] = (2.0, 0.18),
                     draws: int = 2000, seed: int = 0, *,
                     ceiling: float | None = None) -> tuple[tuple[float, float], str]:
    """`sigma(pick)` fitted from where this room's picks actually landed, and what to say.

    **Fitted on `ecr`, which is `mu_pick` for every draft this fits on.** `_sigma` applies the
    result to `mu_pick`, an expected pick number blending ADP and consensus -- and for a past
    preseason no ADP exists, so the blend is the identity and the two axes coincide on the
    fitting population. On the *live* board they differ, and that is a limitation the fit
    carries rather than one it can remove (#155). An earlier version of this docstring said
    "fitted on a pick number", which described the intent and not the code.

    **Restricted to `ecr <= PICK_NOISE_FIT_CEILING`**, not to the whole draftable pool. The
    sample is conditioned on being drafted, and past that rank it is one-sided -- only the
    players who beat their rank are observed -- which was fitting the slope to a single tail.
    The table above the constants shows where the signature appears and what it cost.

    The pool restriction #40 introduced stands beneath this one: a consensus rank past the
    last pick anyone ever made is not a pick number at all.

    The interval is bootstrapped over **drafts**, not picks. Four drafts supply every
    observation, and picks inside one draft are anything but independent -- one manager
    reaching in round two moves every later pick in that room. Resampling picks would report
    an interval several times too tight, which is the same error `docs/gate-power.md` is about
    one layer up.

    **And the sentence says what four clusters can support (#287).** Four drafts resampled
    with replacement give 4^4 = 256 ordered resamples and 35 distinct multisets, so the
    2,000 draws revisit at most 256 of them and the percentiles are those of a small discrete
    set. It also says what the interval cannot do: two fits on nested subsets of the same
    drafts -- the whole pool and the part under the ceiling -- share their resamples, so
    non-overlap of their intervals is not evidence that they differ. That reading was the
    claim #287 withdrew.

    `ceiling` overrides `PICK_NOISE_FIT_CEILING` for the sweep and nothing else; the shipped
    fit is the default.

    Returned rather than printed, and taking a frame rather than a league id, because reaching
    this through `fit_pick_noise` needs an ESPN session.
    """
    if df.height < 50:
        return default, (f"  only {df.height} matched historical picks; keeping the "
                         f"{default} prior.")
    pool = float(df["pick"].to_numpy().max())
    # Two boundaries, and they are different claims. The pool is where picks stop existing;
    # the ceiling is where the drafted sample stops being two-sided. The second is inside the
    # first, and it is the one the fit needs.
    ceiling = min(pool, float(PICK_NOISE_FIT_CEILING if ceiling is None else ceiling))
    inside = df.filter(pl.col("ecr") <= ceiling)
    if inside.height < 50:
        return default, (f"  only {inside.height} picks inside the fit ceiling of "
                         f"{ceiling:.0f} (pool {pool:.0f}); keeping the {default} prior.")

    x = inside["ecr"].to_numpy().astype(float)
    # Absolute deviation of a normal is sigma*sqrt(2/pi); rescale to a real sigma.
    y = np.abs(inside["pick"].to_numpy().astype(float) - x) * np.sqrt(np.pi / 2.0)
    a, b = _constrained(x, y)
    if not (0.0 < a < 50.0 and 0.0 <= b < 2.0):
        return default, (f"  fitted noise (a={a:.2f}, b={b:.3f}) is out of range; keeping "
                         f"{default}.")

    drafts = inside["year"].unique().to_list()
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(draws):
        pick = rng.choice(drafts, size=len(drafts), replace=True)
        rows = pl.concat([inside.filter(pl.col("year") == yr) for yr in pick])
        xs = rows["ecr"].to_numpy().astype(float)
        ys = np.abs(rows["pick"].to_numpy().astype(float) - xs) * np.sqrt(np.pi / 2.0)
        slopes.append(_constrained(xs, ys)[1])
    lo, hi = (float(v) for v in np.percentile(slopes, [2.5, 97.5]))
    k = len(drafts)
    ordered, distinct = _distinct_resamples(k)
    return (a, b), (
        f"  fitted pick noise from {inside.height} picks with ecr <= {ceiling:.0f} (pool "
        f"{pool:.0f}; past the ceiling the drafted sample is one-sided) over {k} "
        f"drafts: sigma = {a:.2f} + {b:.3f} * pick "
        f"(slope 95% CI [{lo:.3f}, {hi:.3f}], clustered on the draft; {k} clusters give "
        f"{k}^{k} = {ordered} ordered resamples, {distinct} distinct, so {draws} draws "
        f"revisit at most {ordered} of them, and the interval cannot resolve a difference "
        f"between nested subsets of the same drafts)")


def sweep_ceilings(df: pl.DataFrame, ceilings: Sequence[int] = PICK_NOISE_SWEEP_CEILINGS, *,
                   draws: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """The fit at each ceiling, so the cut is published as a sensitivity and not asserted.

    One row per ceiling: the ceiling asked for, the one applied (a ceiling past the pool
    collapses to the pool), `n`, the line, the draft-clustered slope interval and sigma at
    pick 100. A ceiling too thin to fit carries the fallback the fitter returned and no
    interval. `scripts/fit_pick_noise.py` prints it; #287 asked for it.
    """
    out: list[dict[str, Any]] = []
    pool = float(df["pick"].to_numpy().max()) if df.height else float("nan")
    for c in ceilings:
        applied = min(pool, float(c))
        n = df.filter(pl.col("ecr") <= applied).height
        (a, b), said = noise_from_picks(df, draws=draws, seed=seed, ceiling=float(c))
        m = re.search(r"slope 95% CI \[([0-9.]+), ([0-9.]+)\]", said)
        ci = (float(m.group(1)), float(m.group(2))) if m else None
        out.append({"ceiling": int(c), "applied": applied, "n": n, "a": a, "b": b, "ci": ci,
                    "sigma_100": a + 100.0 * b, "said": said})
    return out


def sweep_lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The sweep as the table the constants' comment carries. A ceiling the fitter fell
    back on prints `--` and `not established` in the fitted cells: the prior it returned is
    not a fit, and a table that printed 2.00 / 0.180 there would read as one."""
    out = [f"  {'ceiling':>7} {'applied':>7} {'n':>5} {'a':>6} {'slope':>6} "
           f"{'95% CI':>16} {'sigma@100':>10}"]
    for r in rows:
        mark = "  <- shipped" if r["ceiling"] == PICK_NOISE_FIT_CEILING else ""
        if r["ci"] is None:
            out.append(f"  {r['ceiling']:>7} {r['applied']:>7.0f} {r['n']:>5} {'--':>6} "
                       f"{'--':>6} {'not established':>16} {'--':>10}{mark}")
            continue
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        out.append(f"  {r['ceiling']:>7} {r['applied']:>7.0f} {r['n']:>5} {r['a']:>6.2f} "
                   f"{r['b']:>6.3f} {ci:>16} {r['sigma_100']:>10.1f}{mark}")
    return out


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
