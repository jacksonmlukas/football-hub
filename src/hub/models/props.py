"""The prop card: every player prop the statline can price, and the scorecard it is kept on.

**Why props are the right first consumer of the statline object.** For a prop we price a
*different* market from the one we conditioned on -- the component line behind a player's
projection is anchored on his draft pick and his points, and the quote being priced is the
betting market's number on his receiving yards -- so there is no circularity and every
market feature is available. For a team-level game prediction that is not true, because the
spread is both the input and the thing being scored. That asymmetry is what decides where
the object goes first (issue #217).

**What is priced, and from what.** `hub.models.predict.components` hands back the per-game
component line the pick and projection imply: attempts, carries, targets, receptions, and
the yards and touchdowns on each. The distribution over one week of any one of those is
the one `hub.models.components.sample_weeks` already draws on the way to a points total --
a count with measured dispersion (`COUNT_DISPERSION`, `TD_DISPERSION`) and yards as a Gamma
compounded on the count (`PER_UNIT_CV`). This module draws the *component* rather than the
sum, which is the only thing a prop needs that the points draw did not expose. The mean is
the line's; the spread law and the skew fall out of the count-and-Gamma structure rather
than being imposed, which is the property `docs/component-projection.md` measured.

**What is logged, and where.** Three numbers per prop, and a timestamp that makes the
decision point unambiguous (`docs/method.md` rule 2): our number and the distribution behind
it; the quote that was *live at the moment we would have decided* -- the latest poll at or
before `decided_at`, carrying `polls_unmoved` and `unmoved_since` so a reader can tell a
live quote from a posted lookahead nobody had touched (#210); and the quote at the close.
Written to the `prop_log` table under `PROP_LOG`, one row per (player, market, decision).
A prop with no posted quote is recorded as `no_line` rather than dropped, and a posted
quote on a player the statline cannot price is recorded as `no_number`.

**Closing line value rather than profit.** CLV converges in weeks where P&L takes years,
and it does not need the game to be played: if the betting market's close moved toward the
side our number implied, our number carried information the quote at decision time did
not. Two forms per prop -- in the quote's own units (yards, receptions) and in vig-free
implied probability, which is the only form the anytime-touchdown market has.

**The ceiling is computed before the gap is chased** (`docs/method.md` rule 8). The most
CLV any side-picker can log on a market is the mean absolute move between decision and
close -- a model that always sided with the move -- so the report prints that beside what
was captured, and a captured share rather than a bare mean. **Repeated props on one player
are not independent** (rule 3): a player's pass yards, pass touchdowns and rush yards share
a game script, and his week 3 and week 4 share him, so every standard error here is taken
over players, and the count of players is printed beside the count of props.

**The baseline to beat is already recorded**: twelve stat-lines against the Week 1 opener,
mean bias +3.5 yards at +16%, MAE 8.8 yards (`docs/next.md`), corroborated at four seasons
of n by `docs/component-projection.md`. The report restates our number against the close
on the same three quantities so the comparison is on the page rather than in a head.

There is no props pull in this repo and this module makes none. `hub.fetch.odds` refuses
every props market by name, because one runs about four credits an *event*; what this
module reads is the `prop_lines` archive that `hub.fetch.odds.record_props` writes from a
payload someone has already chosen to pay for.

    uv run python -m hub.models.props --log --players data/processed/roster.parquet \\
        --decided-at 2026-09-13T12:00 --week 2
    uv run python -m hub.models.props --report
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from hub import store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD, fitted_digest
from hub.contracts import PROP_LOG
from hub.models import components as C
from hub.models.predict import components as component_line
from hub.names import player_key

NOT_FITTED_BECAUSE = (
    "every input to a prop price is imported -- the component line from `hub.models.volume` "
    "through `hub.models.predict`, and the count and yardage dispersions from "
    "`hub.models.components` -- and is hashed where it lives or named in `NOT_IN_DIGEST`. "
    "The floats assigned here are the recorded baseline the scorecard is compared against, "
    "which scores predictions and is an input to none, and `DRAWS` sets the precision of a "
    "Monte Carlo estimate of a fixed distribution rather than the extent of a simulated "
    "season: doubling it moves a price in its third decimal and no side with it. "
)

# The recorded baseline (`docs/next.md`, 2026-08-24): twelve stat-lines against the Week 1
# opener. Our number was high by 3.5 yards, 16% of the posted line, at 8.8 yards MAE. The
# report prints these beside what the log now says, on the same three quantities.
BASELINE_N = 12
BASELINE_BIAS_YARDS = 3.5
BASELINE_BIAS_SHARE = 0.16
BASELINE_MAE_YARDS = 8.8

# How many weeks of one prop are drawn to price it. Deterministic under `SEED`, so the same
# line and the same point give the same number on every machine.
DRAWS = 20000
SEED = 0

# An unposted market is recorded as a prop of the player's when at least this share of his
# weeks would clear zero on it. A receiver's rushing yards clear zero one week in eight and
# no book hangs the prop; his anytime touchdown clears it two weeks in five and every book
# does. A share of weeks rather than a threshold in yards, because the seven markets are in
# four units and one rule has to read across them. A posted quote is priced regardless.
MIN_WEEKS_NONZERO = 0.25

MODEL = "statline_props"

# Which Odds API market prices which component, and how the component is drawn: the phase
# whose dispersion applies, and whether the quantity is the phase's count, its yards, or its
# touchdowns. `anytime_td` is the one composite -- rushing plus receiving touchdowns, and the
# question is whether the sum reached one.
MARKET_STATS: dict[str, str] = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_rush_attempts": "carries",
    "player_pass_tds": "passing_tds",
    "player_anytime_td": "anytime_td",
}
_DRAW: dict[str, tuple[str, str]] = {
    "passing_yards": ("pass", "yards"), "rushing_yards": ("rush", "yards"),
    "receiving_yards": ("rec", "yards"),
    "attempts": ("pass", "units"), "carries": ("rush", "units"), "receptions": ("rec", "units"),
    "passing_tds": ("pass", "tds"), "rushing_tds": ("rush", "tds"),
    "receiving_tds": ("rec", "tds"),
}
_UNITS = {"pass": "attempts", "rush": "carries", "rec": "receptions"}
_YARDS = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}
_TDS = {"pass": "passing_tds", "rush": "rushing_tds", "rec": "receiving_tds"}
# The markets where our number and the quote share a unit, so a CLV in points and a bias in
# yards mean something. Counts are in their own unit; the anytime market has no point.
YARDS_MARKETS = ("player_pass_yds", "player_rush_yds", "player_reception_yds")
POINT_MARKETS = (*YARDS_MARKETS, "player_receptions", "player_rush_attempts",
                 "player_pass_tds")

PRICED, NO_LINE, NO_NUMBER = "priced", "no_line", "no_number"
OVER, UNDER = "over", "under"


# --- the distribution behind one component ---------------------------------------

def _counts(rng: np.random.Generator, mean: float, phi: float, n: int) -> np.ndarray:
    """`n` counts with mean `mean` and variance `phi * mean` -- the family `hub.models.components`
    chose from the measured dispersion: negative binomial above Poisson, binomial below."""
    if mean <= 0:
        return np.zeros(n, dtype=int)
    if phi > 1.0:
        p = 1.0 / phi
        return rng.negative_binomial(max(mean * p / (1.0 - p), 1e-9), p, n)
    if phi < 1.0:
        p = 1.0 - phi
        trials = round(mean / p)
        return rng.binomial(max(trials, 1), min(mean / max(trials, 1), 1.0), n)
    return rng.poisson(mean, n)


def _phase_draws(rng: np.random.Generator, line: Mapping[str, float], phase: str,
                 n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One phase's (units, yards, touchdowns) over `n` weeks, drawn the way `sample_weeks` does.

    Yards are a Gamma compounded on the drawn count -- a week's yardage is the sum of one
    gain per unit -- so their variance grows with the count rather than with the square of
    the mean, which is the square-root spread law arriving as an output. A line that carries
    yards and no count infers one from the phase's typical yards per unit, exactly as the
    points draw does, so the compound structure holds either way.
    """
    mu_y = float(line.get(_YARDS[phase]) or 0.0)
    mu_u = float(line.get(_UNITS[phase]) or 0.0)
    if mu_y > 0 and mu_u <= 0:
        mu_u = mu_y / C.YARDS_PER_UNIT[phase]
    units = _counts(rng, mu_u, C.COUNT_DISPERSION[phase], n)
    yards = np.zeros(n)
    if mu_y > 0 and mu_u > 0:
        cv = C.PER_UNIT_CV[phase]
        shape = units / cv ** 2
        nz = shape > 0
        yards[nz] = rng.gamma(shape[nz], (mu_y / mu_u) * cv ** 2)
    tds = _counts(rng, float(line.get(_TDS[phase]) or 0.0), C.TD_DISPERSION[phase], n)
    return units, yards, tds


def stat_draws(line: Mapping[str, float], stat: str, *, n: int = DRAWS,
               seed: int = SEED) -> np.ndarray:
    """`n` weeks of one component for a player with this per-game line.

    `anytime_td` is rushing plus receiving touchdowns, drawn from one generator so the two
    phases' draws are the ones a single simulated week would contain.
    """
    rng = np.random.default_rng(seed)
    if stat == "anytime_td":
        _, _, rush = _phase_draws(rng, line, "rush", n)
        _, _, rec = _phase_draws(rng, line, "rec", n)
        return (rush + rec).astype(float)
    phase, kind = _DRAW[stat]
    units, yards, tds = _phase_draws(rng, line, phase, n)
    return {"units": units.astype(float), "yards": yards, "tds": tds.astype(float)}[kind]


# --- one price -------------------------------------------------------------------

def implied(price: float | None) -> float | None:
    """The probability an American price charges for, vig included."""
    if price is None or not math.isfinite(price) or abs(price) < 100:
        return None
    return (-price / (-price + 100.0)) if price < 0 else (100.0 / (price + 100.0))


def novig_over(over_price: float | None, under_price: float | None) -> float | None:
    """The Over's implied probability with the vig taken out, or with it in when only the
    Over is quoted. Two-sided quotes are normalised to sum to one; a one-sided quote cannot
    be, and is returned as charged rather than invented a partner for."""
    q_over, q_under = implied(over_price), implied(under_price)
    if q_over is None:
        return None
    if q_under is None:
        return q_over
    return q_over / (q_over + q_under)


def fair_price(p: float) -> float:
    """The American price at which probability `p` is even money, clipped away from 0 and 1
    so a certainty from a finite sample does not divide by zero."""
    p = min(max(p, 1.0 / DRAWS), 1.0 - 1.0 / DRAWS)
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


@dataclass(frozen=True)
class PropPrice:
    """Our number on one prop and the distribution behind it, against a posted point.

    `p_push` is the mass on the point itself, which is nothing for a yardage prop and
    something for a count prop quoted at a whole number. `p_over` is strictly-over, so the
    three sum to one and a reader adding the push back to whichever side a book refunds it
    on can do so.
    """

    market: str
    stat: str
    mean: float
    sd: float
    p10: float
    p50: float
    p90: float
    point: float | None
    p_over: float | None
    p_under: float | None
    p_push: float | None
    # The share of weeks that clear zero: what says whether an unposted market is a prop.
    p_nonzero: float = 1.0

    @property
    def fair_over(self) -> float | None:
        return None if self.p_over is None else fair_price(self.p_over)


def price(line: Mapping[str, float], market: str, point: float | None, *,
          n: int = DRAWS, seed: int = SEED) -> PropPrice:
    """Price one prop off a component line.

    The anytime market has no point: its over is "at least one", and `point` is ignored
    for it. Every other market with no point is priced for its distribution alone --
    `p_over` null -- which is what a `no_line` row carries.
    """
    stat = MARKET_STATS[market]
    draws = stat_draws(line, stat, n=n, seed=seed)
    sd = float(draws.std())
    p10, p50, p90 = (float(x) for x in np.percentile(draws, (10, 50, 90)))
    nonzero = float((draws > 0).mean())
    if stat == "anytime_td":
        over = float((draws >= 1).mean())
        return PropPrice(market, stat, float(draws.mean()), sd, p10, p50, p90,
                         None, over, 1.0 - over, 0.0, nonzero)
    if point is None:
        return PropPrice(market, stat, float(draws.mean()), sd, p10, p50, p90,
                         None, None, None, None, nonzero)
    over = float((draws > point).mean())
    push = float((draws == point).mean())
    return PropPrice(market, stat, float(draws.mean()), sd, p10, p50, p90,
                     float(point), over, 1.0 - over - push, push, nonzero)


# --- the card ---------------------------------------------------------------------

_PICK_COLUMNS = ("adp", "consensus_rank", "ecr")
_PROJ_COLUMNS = ("proj_blend", "proj_ppg")
_POS_COLUMNS = ("pos", "position")


def component_lines(players: pl.DataFrame) -> dict[str, tuple[str, str, dict[str, float]]]:
    """Each player's (display name, position, per-game component line), by player key.

    The pick comes from `adp` where the frame has one and consensus rank where it does not,
    and the projection from `proj_blend` over `proj_ppg` -- the same coalescing the roster
    and the board do. A player whose position the volume curve was never fitted for (K,
    DST) gets no line, and a posted quote on him is logged as `no_number`.
    """
    pos_col = next((c for c in _POS_COLUMNS if c in players.columns), None)
    if "player" not in players.columns or pos_col is None:
        raise ValueError(f"players needs `player` and a position column; got {players.columns}")
    picks = [pl.col(c) for c in _PICK_COLUMNS if c in players.columns]
    projs = [pl.col(c) for c in _PROJ_COLUMNS if c in players.columns]
    if not picks or not projs:
        raise ValueError(f"players needs one of {_PICK_COLUMNS} and one of {_PROJ_COLUMNS}; "
                         f"got {players.columns}")
    frame = players.select(pl.col("player"), pl.col(pos_col).alias("pos"),
                           pl.coalesce(*picks).cast(pl.Float64).alias("pick"),
                           pl.coalesce(*projs).cast(pl.Float64).alias("proj"))
    out: dict[str, tuple[str, str, dict[str, float]]] = {}
    for r in frame.iter_rows(named=True):
        if r["pick"] is None or r["proj"] is None or r["pos"] is None:
            continue
        line = component_line(float(r["pick"]), str(r["pos"]), float(r["proj"]))
        if line:
            out.setdefault(player_key(r["player"]), (r["player"], str(r["pos"]), line))
    return out


_QUOTE = ("game_id", "player_key", "player", "market", "point", "over_price",
          "under_price", "captured_at", "polls_unmoved", "unmoved_since")
_KEY = ("player_key", "market")


def quotes_as_of(polls: pl.DataFrame, at: datetime) -> pl.DataFrame:
    """The quote live at `at` for every prop polled at or before it: the latest such poll.

    A prop whose only polls come *after* `at` is absent rather than null -- it was not
    priced then, and returning its later quote is the lookahead the as-of join exists to
    prevent. `polls_unmoved` and `unmoved_since` ride with the row, so what the quote's
    liveness was at that moment is what the log carries.
    """
    have = [c for c in _QUOTE if c in polls.columns]
    return (polls.select(have).filter(pl.col("captured_at") <= at)
                 .sort("captured_at").group_by(*_KEY, maintain_order=True).last())


def card(players: pl.DataFrame, quotes: pl.DataFrame, *, decided_at: datetime,
         n: int = DRAWS, seed: int = SEED) -> pl.DataFrame:
    """Every prop the statline can price, and every prop the betting market posted, priced.

    One row per (player, market). A player with a line is priced on every market posted on
    him (`priced`), and on every unposted market that `MIN_WEEKS_NONZERO` says is a prop of
    his (`no_line`) -- a receiver's anytime touchdown is recorded whether or not it was
    posted, his rushing yards are not. A posted quote on a player with no line is kept as
    `no_number`. Nothing is dropped, which is what lets the report say how much of the slate
    each side of the comparison covers.

    `quotes` is what `quotes_as_of` returns for `decided_at`; the caller passes the moment
    and this stamps it on every row, so the decision point is a timestamp and not a week.
    """
    lines = component_lines(players)
    posted = {(r["player_key"], r["market"]): r
              for r in quotes.iter_rows(named=True) if r["market"] in MARKET_STATS}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pk, (name, pos, line) in lines.items():
        for market, stat in MARKET_STATS.items():
            q = posted.get((pk, market))
            p = price(line, market, None if q is None else q.get("point"), n=n, seed=seed)
            if q is None and p.p_nonzero < MIN_WEEKS_NONZERO:
                continue
            seen.add((pk, market))
            rows.append({**_blank(pk, name, pos, market, stat, decided_at),
                         **_ours(p), **_decision(q, p)})
    for (pk, market), q in posted.items():
        if (pk, market) in seen:
            continue
        rows.append({**_blank(pk, q["player"], None, market, MARKET_STATS[market], decided_at),
                     **_decision(q, None), "status": NO_NUMBER})
    return pl.DataFrame(rows, schema=_CARD_SCHEMA)


_CARD_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "player_key": pl.Utf8, "player": pl.Utf8, "position": pl.Utf8,
    "market": pl.Utf8, "stat": pl.Utf8, "status": pl.Utf8, "decided_at": pl.Datetime,
    "our_mean": pl.Float64, "our_sd": pl.Float64, "our_p50": pl.Float64,
    "our_p_over": pl.Float64, "side": pl.Utf8, "edge": pl.Float64,
    "decision_point": pl.Float64, "decision_over_price": pl.Float64,
    "decision_under_price": pl.Float64, "decision_captured_at": pl.Datetime,
    "decision_polls_unmoved": pl.Int64, "decision_unmoved_since": pl.Datetime,
}


def _blank(pk: str, name: str, pos: str | None, market: str, stat: str,
           decided_at: datetime) -> dict[str, Any]:
    return {**dict.fromkeys(_CARD_SCHEMA), "player_key": pk, "player": name,
            "position": pos, "market": market, "stat": stat, "decided_at": decided_at,
            "status": NO_LINE}


def _ours(p: PropPrice) -> dict[str, Any]:
    return {"our_mean": p.mean, "our_sd": p.sd, "our_p50": p.p50, "our_p_over": p.p_over}


def _decision(q: Mapping[str, Any] | None, p: PropPrice | None) -> dict[str, Any]:
    """The quote at the decision, and the side our number takes against it.

    The side is the one our probability exceeds the vig-free implied on. `edge` is that
    excess on the chosen side, which is why it is never negative: a prop where the betting
    market and the statline agree exactly has an edge of zero and a side chosen by the
    tie-break toward the Under, which is the side a push refunds toward on most books.
    """
    if q is None:
        return {}
    out: dict[str, Any] = {
        "game_id": q.get("game_id"), "status": PRICED,
        "decision_point": q.get("point"), "decision_over_price": q.get("over_price"),
        "decision_under_price": q.get("under_price"),
        "decision_captured_at": q.get("captured_at"),
        "decision_polls_unmoved": q.get("polls_unmoved"),
        "decision_unmoved_since": q.get("unmoved_since"),
    }
    if p is None or p.p_over is None:
        return out
    q_over = novig_over(q.get("over_price"), q.get("under_price"))
    if q_over is None:
        q_over = 0.5
    # A push is refunded, so the two sides are compared on the mass that is not a push.
    p_over = p.p_over / (1.0 - (p.p_push or 0.0)) if (p.p_push or 0.0) < 1.0 else 0.5
    if p_over > q_over:
        out.update(side=OVER, edge=p_over - q_over)
    else:
        out.update(side=UNDER, edge=q_over - p_over)
    return out


# --- the close, and the scorecard -------------------------------------------------

def closing_quotes(polls: pl.DataFrame, at: datetime | None = None) -> pl.DataFrame:
    """The last quote of every prop, at or before `at` when given.

    `at` is kickoff when the caller knows it. Without it the close is the latest poll the
    archive holds, and `close_captured_at` on the row says how close to kickoff that was.
    """
    got = polls if at is None else polls.filter(pl.col("captured_at") <= at)
    return (got.sort("captured_at").group_by(*_KEY, maintain_order=True).last()
               .select(*_KEY, pl.col("point").alias("close_point"),
                       pl.col("over_price").alias("close_over_price"),
                       pl.col("under_price").alias("close_under_price"),
                       pl.col("captured_at").alias("close_captured_at")))


def _novig_col(over: str, under: str) -> pl.Expr:
    return pl.struct(over, under).map_elements(
        lambda s: novig_over(s[over], s[under]), return_dtype=pl.Float64)


def scorecard(card_rows: pl.DataFrame, close: pl.DataFrame) -> pl.DataFrame:
    """The card with the close beside every decision, and the two CLVs derived.

    `clv_points` is the close's point minus the decision's, signed toward our side: positive
    means the betting market moved toward the number we had. `clv_prob` is the same thing in
    vig-free implied probability on our side, which is the one form every market has. Both
    are null where there was no decision -- a `no_line` or `no_number` row -- so a mean over
    the column is a mean over decisions and nothing else.
    """
    joined = card_rows.join(close, on=list(_KEY), how="left")
    # `side` is null exactly where there was no decision, so the sign carries the null and
    # both CLVs are null on a `no_line` or `no_number` row without a second condition.
    sign = (pl.when(pl.col("side") == OVER).then(1.0)
              .when(pl.col("side") == UNDER).then(-1.0).otherwise(None))
    q_dec = _novig_col("decision_over_price", "decision_under_price")
    q_close = _novig_col("close_over_price", "close_under_price")
    return joined.with_columns(
        (sign * (pl.col("close_point") - pl.col("decision_point"))).alias("clv_points"),
        (sign * (q_close - q_dec)).alias("clv_prob"),
        pl.lit(MODEL).alias("model"),
        pl.lit(version()).alias("version"),
    )


def version() -> str:
    """What identifies a prop price: the fitted constants, plus the four dispersions.

    The dispersions are named in `hub.config.NOT_IN_DIGEST` as read only by the points draw,
    which stopped being true the day this module read them, so they are folded in here
    rather than left to a claim that no longer holds.
    """
    tables = (C.PER_UNIT_CV, C.YARDS_PER_UNIT, C.COUNT_DISPERSION, C.TD_DISPERSION)
    text = "|".join(repr(sorted(t.items())) for t in tables)
    return f"{MODEL}-{fitted_digest()}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def write_log(log: pl.DataFrame, season: int, week: int, *,
              base: Path | None = None) -> Path:
    """One partition per decision, named by its timestamp, so two decisions are two rows."""
    at = log["decided_at"].max()
    part = PROP_LOG.validate(log.select(list(PROP_LOG.required)))
    return store.write(part, "prop_log", "nfl", season, week, base=base,
                       name=f"decided-{at:%Y%m%dT%H%M%S}")


# --- the report -------------------------------------------------------------------

def _by_player(log: pl.DataFrame, col: str) -> pl.DataFrame:
    """One value per player: the mean of that player's props, which is the unit a standard
    error is taken over here (`docs/method.md` rule 3)."""
    return (log.filter(pl.col(col).is_not_null()).group_by("player_key")
               .agg(pl.col(col).mean().alias(col), pl.len().alias("props")))


def _mean_se(values: pl.Series) -> tuple[float | None, float | None]:
    v = values.drop_nulls().to_numpy().astype(float)
    if v.size == 0:
        return None, None
    mean = float(v.mean())
    if v.size < 2:
        return mean, None
    return mean, float(v.std(ddof=1)) / math.sqrt(v.size)


def _ceiling(log: pl.DataFrame, col: str) -> float | None:
    """The mean absolute move, per player and then over players -- the same unit as the
    captured mean it is the denominator of. Taken over props it was a different average,
    and a player with many flat props diluted it under a numerator he barely touched."""
    per_player = _by_player(log.with_columns(pl.col(col).abs()), col)
    v = per_player[col].to_numpy().astype(float)
    return float(v.mean()) if v.size else None


def clv_by_market(log: pl.DataFrame) -> pl.DataFrame:
    """Per market: how much closing line value the decisions captured, against the ceiling.

    `ceiling_points` and `ceiling_prob` are the mean absolute move between decision and
    close -- what a side-picker who was always right would have logged -- and `share` is
    the captured mean over that. Both means are over players, so the ratio is of one
    unit; a per-prop ceiling under a per-player numerator read 500% on ten props. `hit`
    is the share of moved quotes that moved toward us; a quote that did not move is
    neither a hit nor a miss and is counted in `unmoved`.
    Standard errors are over players, and `players` is printed beside `props` so the reader
    sees which count the precision came from.
    """
    rows = []
    for market in MARKET_STATS:
        got = log.filter((pl.col("market") == market) & (pl.col("status") == PRICED))
        moved = got.filter(pl.col("clv_prob").is_not_null() & (pl.col("clv_prob") != 0))
        per_player = _by_player(got, "clv_prob")
        mean_prob, se_prob = _mean_se(per_player["clv_prob"])
        per_player_pts = _by_player(got, "clv_points")
        mean_pts, se_pts = _mean_se(per_player_pts["clv_points"])
        hits = (moved["clv_prob"] > 0).to_numpy()
        with_point = market in POINT_MARKETS
        rows.append({
            "market": market, "props": got.height,
            "players": per_player.height,
            "unmoved": got.height - moved.height,
            "hit": float(hits.mean()) if hits.size else None,
            "clv_prob": mean_prob, "se_prob": se_prob,
            "ceiling_prob": _ceiling(got, "clv_prob"),
            "clv_points": mean_pts if with_point else None,
            "se_points": se_pts if with_point else None,
            "ceiling_points": _ceiling(got, "clv_points") if with_point else None,
        })
    out = pl.DataFrame(rows)
    share = pl.when(pl.col("ceiling_prob") > 0).then(
        pl.col("clv_prob") / pl.col("ceiling_prob")).otherwise(None)
    return out.with_columns(share.alias("share"))


def baseline_comparison(log: pl.DataFrame) -> dict[str, Any]:
    """Our number against the close on the yardage markets, on the recorded baseline's three
    quantities: mean bias in yards, bias as a share of the quote, and MAE.

    Against the close rather than the decision quote because the close is the better
    estimate of the truth the baseline was measuring against; the decision quote is what
    CLV is for. One row per prop, which is the basis the twelve stat-lines were on.
    """
    got = log.filter(pl.col("market").is_in(YARDS_MARKETS) & (pl.col("status") == PRICED)
                     & pl.col("close_point").is_not_null() & pl.col("our_mean").is_not_null())
    if got.is_empty():
        return {"n": 0, "bias_yards": None, "bias_share": None, "mae_yards": None,
                "players": 0}
    ours = got["our_mean"].to_numpy().astype(float)
    close = got["close_point"].to_numpy().astype(float)
    diff = ours - close
    return {"n": got.height, "players": got["player_key"].n_unique(),
            "bias_yards": float(diff.mean()),
            "bias_share": float(diff.sum() / close.sum()),
            "mae_yards": float(np.abs(diff).mean())}


def coverage(log: pl.DataFrame) -> dict[str, int]:
    """How many props landed in each state, which is the acceptance criterion that nothing
    was dropped made countable."""
    counts = log.group_by("status").len()
    return {s: int(counts.filter(pl.col("status") == s)["len"].sum())
            for s in (PRICED, NO_LINE, NO_NUMBER)}


def _fmt(v: Any, width: int = 7, digits: int = 3) -> str:
    return f"{'-':>{width}}" if v is None else f"{v:>{width}.{digits}f}"


def report(log: pl.DataFrame) -> None:
    cov = coverage(log)
    decisions = log.filter(pl.col("status") == PRICED)["decided_at"].n_unique()
    print(f"  prop scorecard: {log.height} rows, {decisions} decision points; "
          f"{cov[PRICED]} priced, {cov[NO_LINE]} with no posted line, "
          f"{cov[NO_NUMBER]} posted on a player the statline cannot price")
    print("  market                  props players unmoved    hit   clv_prob  se  ceiling "
          "share   clv_pts  se  ceiling")
    for r in clv_by_market(log).iter_rows(named=True):
        print(f"  {r['market']:<22} {r['props']:>6} {r['players']:>7} {r['unmoved']:>7} "
              f"{_fmt(r['hit'], 6, 2)} {_fmt(r['clv_prob'], 9, 4)} {_fmt(r['se_prob'], 6, 4)} "
              f"{_fmt(r['ceiling_prob'], 7, 4)} {_fmt(r['share'], 5, 2)} "
              f"{_fmt(r['clv_points'], 8, 2)} {_fmt(r['se_points'], 6, 2)} "
              f"{_fmt(r['ceiling_points'], 7, 2)}")
    b = baseline_comparison(log)
    print(f"  yardage markets against the close: n={b['n']} props on {b['players']} players, "
          f"bias {_fmt(b['bias_yards'], 6, 2)} yds "
          f"({_fmt(None if b['bias_share'] is None else 100 * b['bias_share'], 5, 1)}%), "
          f"MAE {_fmt(b['mae_yards'], 5, 2)} yds")
    print(f"  recorded baseline (docs/next.md, n={BASELINE_N}): bias +{BASELINE_BIAS_YARDS} yds "
          f"(+{100 * BASELINE_BIAS_SHARE:.0f}%), MAE {BASELINE_MAE_YARDS} yds")


# --- the CLI ---------------------------------------------------------------------

def _polls(season: int, week: int | None, base: Path | None) -> pl.DataFrame:
    if "prop_lines" not in store.tables(base):
        raise FileNotFoundError("no prop_lines archive in the store; "
                                "`hub.fetch.odds --record-props` writes one")
    q = "SELECT * FROM prop_lines WHERE league = 'nfl' AND season = ?"
    params: list[object] = [season]
    if week is not None:
        q += " AND week = ?"
        params.append(store.week_key(week))
    return store.sql(q, params=params, base=base)


def log_decisions(players: pl.DataFrame, polls: pl.DataFrame, *, decided_at: datetime,
                  close_at: datetime | None = None, n: int = DRAWS,
                  seed: int = SEED) -> pl.DataFrame:
    """Price the card at `decided_at` against the quotes live then, and score it at the close."""
    priced = card(players, quotes_as_of(polls, decided_at), decided_at=decided_at,
                  n=n, seed=seed)
    return scorecard(priced, closing_quotes(polls, close_at))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.props",
        description="Price every player prop the statline can, log the quote at the decision "
                    "and at the close, and report closing line value per market. Reads the "
                    "store and spends nothing.")
    ap.add_argument("--log", action="store_true",
                    help="price the players in --players against the prop_lines archive as of "
                         "--decided-at and write the scorecard to prop_log")
    ap.add_argument("--report", action="store_true",
                    help="closing line value per market from prop_log, against the ceiling and "
                         "the recorded baseline")
    ap.add_argument("--players", default=None, help="parquet with player, pos, adp/ecr and "
                                                    "proj_blend/proj_ppg (the roster or board)")
    ap.add_argument("--decided-at", default=None, help="ISO timestamp of the decision point")
    ap.add_argument("--close-at", default=None, help="ISO timestamp the close is read as of "
                                                     "(default: the latest poll)")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--base", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    base = Path(a.base) if a.base else None

    if a.log:
        if not a.players or not a.decided_at or a.week is None:
            print("hub.models.props: --log needs --players, --decided-at and --week",
                  file=sys.stderr)
            return 2
        try:
            players = pl.read_parquet(a.players)
            polls = _polls(a.season, a.week, base)
        except Exception as e:
            return unavailable("hub.models.props", "the players file or the prop_lines archive", e)
        decided = datetime.fromisoformat(a.decided_at)
        close_at = datetime.fromisoformat(a.close_at) if a.close_at else None
        log = log_decisions(players, polls, decided_at=decided, close_at=close_at)
        path = write_log(log, a.season, a.week, base=base)
        print(f"  wrote {log.height} rows to {path}")
        report(log)
        return 0

    if "prop_log" not in store.tables(base):
        print("hub.models.props: no prop_log in the store; `--log` writes one from a "
              "players file and the prop_lines archive", file=sys.stderr)
        return 1
    log = store.sql("SELECT * FROM prop_log WHERE league = 'nfl' AND season = ?",
                    params=[a.season], base=base)
    if log.is_empty():
        print(f"hub.models.props: no {a.season} rows in prop_log", file=sys.stderr)
        return 1
    report(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
