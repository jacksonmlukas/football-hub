"""The line-move study (#221) and its verdict (#329): a reader of the starter-change events
(`hub.models.starter_events`), one of the two the split at #346 left.

The home-spread move from the frozen price to the last snapshot before the game day, less the
mean move of the week's other games between the same two poll days, regressed on the net
ex-ante quality gap -- arriving starter's value minus departing, home minus away, both off the
pinned file -- against 538's 0.132 points per value unit. The standard error is #214's noise
floor per window over the gap's spread and root n - 1 (OLS's own denominator, #303), because
the event rows alone cannot resolve their own residual. An event game with no result yet is
excluded rather than priced off a truncated in-flight snapshot, and the count is reported; a
week whose only other archived games are themselves event games has no control and is refused
rather than fitted at a manufactured zero week-mean. Censored events (no snapshot before the
change could be known) are split on the run line into never-polled and polled-only-after-the-
change, two different facts a single count used to conflate, and the change-point -- scanned
over poll days, bounded at the game day, every date compared through the poll-day conversion --
is reported in days from the previous game day, beside the coefficient. **Since 2026-09-17
(#300) this coefficient is the module's ADOPT condition**: sign and magnitude against the 0.132
benchmark, season-clustered once two seasons exist (`docs/gate-power.md`).

**The verdict is `experiment`'s machinery plus one rule (#346).** The interval is
`experiment.t_interval`, the MDE `experiment.minimum_detectable_effect`, the
events-needed search `experiment.smallest_n_resolving` -- the same functions the Gate reads, not
a second copy of each. What this module owns is the *rule* (`verdict`): non-inferiority against
`DELTA`, its own branch set, which no Gate has (method.md rule 1, *Where the rule lives in
code*). The statistic is shared; the branches are not.

Nothing here fetches but one guarded call: `--study` asks `hub.fetch.odds.load_qb_starters` for
the depth chart #214's own floor conditions on, exactly as `odds.noise_floor_report` does, and
degrades the same way that report does when the chart is unavailable -- the floor is still
printed, over every live interval, and the line says the same-quarterback condition was not
applied rather than printing an unconditioned number under that label (#330).
"""
from __future__ import annotations

import datetime as dt
import math
import statistics
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from hub.declare import not_an_input
from hub.fetch import odds
from hub.models import experiment
from hub.models.starter_events import in_season_events, poll_day, priced, results

# 538's conversion, 3.3 Elo per value unit over 25 Elo per point: the coefficient the
# study's fitted slope is read against. A benchmark no prediction reads.
BENCHMARK = not_an_input(
    3.3 / 25,
    "the study's benchmark slope, 538's own construction; a comparison figure the line-move "
    "study prints, and nothing that predicts reads it")

def _exclude_unplayed(have: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """`have` (uncensored, priced event games) with no result yet dropped. `gate_rows` has
    always joined `results` before scoring an arm; the study never did (#303), so an
    in-flight game's `close` -- the last snapshot before its own game day -- was just its
    latest snapshot, not the last one before a move that had finished happening, and the
    move it fed into the regression was truncated with nothing marking the row. A semi join
    on `results(tg)` keeps exactly the rows the gate would keep."""
    return have.join(results(tg).select("game_id"), on="game_id", how="semi")


def unplayed_study_games(polls: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame) -> int:
    """How many uncensored, priced event games `study_rows` excludes as unplayed (#303) --
    off the same `priced` frame `study_rows` itself filters, so the count and the exclusion
    can never disagree about which games they mean. Reported on the run line rather than
    left to shrink `n` silently. Counted among the valued games, so this and
    `unvalued_study_games` partition what `study_rows` drops and never both claim a game."""
    have = _valued(_priced_both_ways(polls, games))
    if have.is_empty():
        return 0
    return have.height - _exclude_unplayed(have, tg).height


def _priced_both_ways(polls: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Event games with a frozen price and a *later* pre-game one: what `study_rows` starts
    from. A windowless game (`priced`) is out -- it would enter the regression as a move of
    zero that nobody observed (#420)."""
    return priced(polls, games).filter(~pl.col("censored") & pl.col("close").is_not_null()
                                       & ~pl.col("windowless"))


def _valued(have: pl.DataFrame) -> pl.DataFrame:
    """The priced games whose net gap is known; a game it is null for is `unvalued`."""
    return have.filter(pl.col("net_gap").is_not_null())


def unvalued_study_games(polls: pl.DataFrame, games: pl.DataFrame) -> int:
    """How many uncensored, priced event games `study_rows` excludes because the pinned
    file holds no value for a starter who changed (#420) -- a play-by-play event in a week
    the file does not reach has a price and a result and no regressor. Reported on the run
    line, so the coefficient's `n` is never short for a reason it does not say."""
    have = _priced_both_ways(polls, games)
    return have.height - _valued(have).height


# --- the study --------------------------------------------------------------------------------

def _week_means(polls: pl.DataFrame, rows: pl.DataFrame,
                event_ids: Sequence[str]) -> pl.DataFrame:
    """Per event game, the mean move over every *other* archived game of the same week
    between the same two poll days -- the week fixed effect as the subtraction it is --
    excluding every game in `event_ids`, not just the row's own (#303): `event_ids` is every
    event game in the span under study, so a week with two starter changes never uses one
    treated game as the other's control. `week_mean` is null, and `week_others` zero, where
    no untreated game was polled on both days; `study_rows` refuses such a row rather than
    fitting it as though the week's mean move were zero."""
    excluded = set(event_ids)
    days = (polls.with_columns(poll_day().alias("poll_day"))
                 .sort("game_id", "captured_at")
                 .group_by("game_id", "week", "poll_day", maintain_order=True)
                 .agg(pl.col("close_spread").last()))
    out = []
    for r in rows.iter_rows(named=True):
        f_day, c_day = r["frozen_day"], r["close_day"]
        others = days.filter((pl.col("week") == r["week"])
                             & ~pl.col("game_id").is_in(list(excluded))
                             & pl.col("poll_day").is_in([f_day, c_day]))
        pivot = (others.group_by("game_id").agg(
            pl.col("close_spread").filter(pl.col("poll_day") == f_day).first().alias("a"),
            pl.col("close_spread").filter(pl.col("poll_day") == c_day).first().alias("b"))
                       .drop_nulls())
        moves = (pivot["b"] - pivot["a"]).to_list()
        out.append({"game_id": r["game_id"],
                    "week_mean": statistics.fmean(moves) if moves else None,
                    "week_others": len(moves)})
    return pl.DataFrame(out, schema={"game_id": pl.Utf8, "week_mean": pl.Float64,
                                     "week_others": pl.Int64})


def _change_points(polls: pl.DataFrame, rows: pl.DataFrame, floor: float) -> list[float | None]:
    """Per event game, days from the previous game day to the first poll *day* -- the last
    poll of an Eastern date standing for the date, exactly as every other date comparison in
    this module goes through `poll_day` -- strictly after the frozen poll's day and
    strictly before the game day, whose move from the frozen price clears
    `floor * sqrt(days since the frozen poll)`. None where no such poll day does.

    Before #303 this scanned individual polls rather than poll days, so several captures on
    one Eastern date could fire the threshold intraday rather than at the day the archive's
    own "one poll per Eastern date" convention (`hub.fetch.odds`) means; it also had no
    upper bound, so a poll captured after the game's own kickoff -- which `priced`'s `close`
    never reads either -- could set a change-point no poll before kickoff had seen. And the
    date arithmetic itself compared the poll's raw UTC calendar date against `frozen_before`,
    an Eastern one; a capture in the small hours UTC is the previous Eastern day, and the old
    line counted it a day too many."""
    days = (polls.with_columns(poll_day().alias("poll_day"))
                 .sort("game_id", "captured_at")
                 .group_by("game_id", "poll_day", maintain_order=True)
                 .agg(pl.col("close_spread").last(), pl.col("captured_at").last()))
    out: list[float | None] = []
    for r in rows.iter_rows(named=True):
        later = (days.filter((pl.col("game_id") == r["game_id"])
                             & (pl.col("poll_day") > r["frozen_day"])
                             & (pl.col("poll_day") < r["date"]))
                     .sort("poll_day"))
        seen: float | None = None
        for p in later.iter_rows(named=True):
            elapsed = (p["captured_at"] - r["frozen_at"]).total_seconds() / 86400.0
            if abs(p["close_spread"] - r["frozen"]) > floor * math.sqrt(max(elapsed, 0.0)):
                seen = float((dt.date.fromisoformat(p["poll_day"])
                              - dt.date.fromisoformat(r["frozen_before"])).days)
                break
        out.append(seen)
    return out


STUDY_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "net_gap": pl.Float64,
    "changes": pl.UInt32, "frozen": pl.Float64, "close": pl.Float64, "move": pl.Float64,
    "week_mean": pl.Float64, "week_others": pl.Int64, "adjusted_move": pl.Float64,
    "window_days": pl.Float64, "days_to_change": pl.Float64,
}


def study_rows(polls: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame,
               floor_per_root_day: float | None = None) -> pl.DataFrame:
    """One row per uncensored, played event game: the move from the frozen price to the last
    snapshot before the game day, the week's mean move over the same two poll days --
    excluding every other event game of the week, not just this one -- the week-adjusted
    move, the net gap, the window in days, and -- given a floor per root-day -- the
    change-point in days from the previous game day.

    **An event game with no result yet is excluded (#303)**, the same way `gate_rows` has
    always excluded one: an in-flight game's `close` is just its latest snapshot, not the
    last one before a move that has finished happening, and reading it as the move would
    truncate the regressor. `unplayed_study_games` reports how many, off the same frame this
    filters. **A row whose week has no game left to serve as a control is refused**, not
    fitted at a manufactured zero week-mean (`_week_means`'s own null `week_mean`). **A game
    whose net gap is unknown is refused too (#420)**, counted by `unvalued_study_games`."""
    have = _valued(_priced_both_ways(polls, games))
    if have.is_empty():
        return pl.DataFrame(schema=STUDY_SCHEMA)
    have = _exclude_unplayed(have, tg)
    if have.is_empty():
        return pl.DataFrame(schema=STUDY_SCHEMA)
    have = have.with_columns(poll_day("frozen_at").alias("frozen_day"),
                             poll_day("close_at").alias("close_day"),
                             (pl.col("close") - pl.col("frozen")).alias("move"),
                             ((pl.col("close_at") - pl.col("frozen_at")).dt.total_seconds()
                              / 86400.0).alias("window_days"))
    have = have.join(_week_means(polls, have, games["game_id"].to_list()),
                     on="game_id", how="left")
    have = have.filter(pl.col("week_mean").is_not_null())
    if have.is_empty():
        return pl.DataFrame(schema=STUDY_SCHEMA)
    changes = (_change_points(polls, have, floor_per_root_day)
               if floor_per_root_day is not None else [None] * have.height)
    return (have.with_columns((pl.col("move") - pl.col("week_mean")).alias("adjusted_move"),
                              pl.Series("days_to_change", changes, dtype=pl.Float64))
                .select(*STUDY_SCHEMA))


def study_mde(*, n: int, sd_gap: float, window_days: float, floor_per_root_day: float) -> float:
    """The coefficient's MDE before the run, as pre-registered: `(t(0.975, n-1) + z(0.80))
    * floor_window / (sd(gap) * sqrt(n - 1))`, the floor per window `floor_per_root_day *
    sqrt(window_days)`; the cluster is the game. The denominator is `n - 1`, OLS's own
    (#303), matching `study_fit`'s `se` so the MDE stated here and the one a fitted run
    reports are never two formulas."""
    if n < 2 or not (sd_gap > 0):
        return float("nan")
    se = floor_per_root_day * math.sqrt(window_days) / (sd_gap * math.sqrt(n - 1))
    return experiment.minimum_detectable_effect(se, n)


def study_fit(rows: pl.DataFrame, *, floor_per_root_day: float) -> dict[str, float]:
    """The slope of the week-adjusted move on the net gap, with its error from the floor.

    Ordinary least squares with an intercept; the standard error is the noise floor per
    window over the gap's spread and root n - 1 rather than the residual's, because a season
    of events cannot resolve its own residual against a floor measured on twelve games --
    OLS's own denominator (#303: dividing by root n instead undercounted the error by 41% at
    n=2 and about 1% at n=53). `t` against the benchmark says whether the betting market
    moved as the source would.
    """
    n = rows.height
    nan = float("nan")
    if n < 2:
        return {"n": float(n), "beta": nan, "se": nan, "mde": nan, "benchmark": BENCHMARK,
                "t_vs_benchmark": nan, "sd_gap": nan, "floor_window": nan}
    x = rows["net_gap"].to_numpy().astype(float)
    y = rows["adjusted_move"].to_numpy().astype(float)
    sd_gap = float(np.std(x, ddof=1))
    beta = float(np.polyfit(x, y, 1)[0]) if sd_gap > 0 else nan
    window = rows["window_days"].to_numpy().astype(float).mean()
    floor_window = floor_per_root_day * math.sqrt(float(window))
    se = floor_window / (sd_gap * math.sqrt(n - 1)) if sd_gap > 0 else nan
    return {"n": float(n), "beta": beta, "se": se,
            "mde": experiment.minimum_detectable_effect(se, n), "benchmark": BENCHMARK,
            "t_vs_benchmark": (beta - BENCHMARK) / se if se and math.isfinite(se) else nan,
            "sd_gap": sd_gap, "floor_window": floor_window}


# --- the study's verdict (#329) -----------------------------------------------------------------

# ADOPTED 2026-09-17 (#329; the maintainer's `ADOPTED:` comment on #300;
# docs/gate-power.md's *Amended 2026-09-17 (#300) -- the ADOPT condition*). **Derivation**: the
# smallest line-move coefficient worth pricing is one half-point tick (the betting market's own
# resolution) on a one-sd starter change -- gap sd 66.3 value units over 231 in-season events
# 2022-2025 (docs/gate-power.md's per-season table: 80.4 / 62.2 / 69.2 / 50.3 on n=62/61/51/57,
# 62+61+51+57=231). 0.5 / 66.3 = 0.0075. DELTA is a constant, pre-registered here, never
# re-derived from the events under test -- the restatement trigger below is what would move
# it, and moving it is a print, never a silent recomputation.
DELTA = not_an_input(
    0.0075,
    "the smallest line-move coefficient worth pricing (#329): one half-point tick on a "
    "one-sd starter change, gap sd 66.3 value units over 231 in-season events 2022-2025 "
    "(docs/gate-power.md)")

# The band the per-season gap sds DELTA was derived from span, 50.3 to 80.4 (rounded out to
# 50-85). A test archive's own gap sd outside it flags the derivation for restatement; inside
# it nobody decides anything -- a print `main` makes beside DELTA, never a branch `verdict`
# reads.
GAP_SD_RESTATEMENT_BAND = not_an_input(
    (50.0, 85.0),
    "the restatement-trigger band DELTA's own per-season gap sds (50.3-80.4) were derived "
    "from, rounded out (#329); a print beside DELTA, never a number a prediction reads")


def gap_sd_restatement_flag(sd_gap: float) -> str | None:
    """None when `sd_gap` falls inside `GAP_SD_RESTATEMENT_BAND`; otherwise the sentence
    naming that DELTA's derivation should be restated (#329's pre-registered trigger). A
    print the run makes beside DELTA and never a branch `verdict` reads."""
    lo, hi = GAP_SD_RESTATEMENT_BAND
    if not math.isfinite(sd_gap) or lo <= sd_gap <= hi:
        return None
    return (f"gap sd {sd_gap:.1f} value units is outside the {lo:.0f}-{hi:.0f} band delta "
            f"was derived from -- derivation flagged for restatement")


def study_events_needed(sd_gap: float, floor_window: float, *, cap: int = 500) -> int | None:
    """The smallest n (event games) at which the coefficient's MDE is at or below DELTA --
    `experiment.smallest_n_resolving`, the search the gate's own event-seasons are counted by,
    here over event games and off `study_fit`'s own `sd_gap` and `floor_window`, so the number
    `verdict`'s NOT-RUNNABLE sentence names and the number this computes cannot be two numbers.
    None when the inputs cannot support the search, or no n up to `cap` clears DELTA. The
    denominator is `n - 1`, matching `study_fit`'s `se` (#303)."""
    if not (math.isfinite(sd_gap) and math.isfinite(floor_window)) or sd_gap <= 0:
        return None
    return experiment.smallest_n_resolving(
        DELTA, lambda n: floor_window / (sd_gap * math.sqrt(n - 1)), cap=cap)


def benchmark_reading(beta: float, lo: float, hi: float) -> str:
    """0.132 (`BENCHMARK`) read beside the fitted coefficient -- **reported, never gated on**
    (docs/gate-power.md: *0.132 is reported, never gated on*): replication if the interval
    contains it, otherwise below or above. Called from `main`, printed beside the coefficient
    -- never read by `verdict`, and it appears in none of that function's branches."""
    if math.isfinite(lo) and math.isfinite(hi) and lo <= BENCHMARK <= hi:
        return "replication (the interval contains 0.132)"
    if not math.isfinite(beta):
        return "not established"
    return "below 0.132" if beta < BENCHMARK else "above 0.132"


def verdict(fit: dict[str, float]) -> tuple[str, str]:
    """The study's own rule over `study_fit`'s coefficient (#329) -- **non-inferiority against
    DELTA**, the machinery stage 2 (`experiment.gate`'s NOT-RUNNABLE precondition) already
    uses. The *statistic* is shared (`study_fit`, `study_mde`); the *rule* is this function's
    own (method.md rule 1, *Where the rule lives in code*) -- a verdict separate from the
    log-loss gate's `run()` above, which has been a diagnostic since #300 and decides neither
    ADOPT nor REMOVE for this module.

    **NOT-RUNNABLE comes first, ahead of every branch**, exactly as `experiment.gate`'s stage 2
    orders its own precondition: `fit['mde']` not finite (fewer than two rows) or exceeding
    DELTA means the interval below is not read at all, and the sentence names how many event
    games DELTA needs (`study_events_needed`, off `study_fit`'s own `sd_gap` and
    `floor_window`, so it never disagrees with the MDE that triggered it).

    Past that gate, `experiment.t_interval` -- the Gate's own interval, on `fit['se']` and the
    event-game count as the cluster count, the same t reference `minimum_detectable_effect`
    reads, so the interval and the MDE cannot disagree about what distribution they are drawn
    from -- decides among three. **At one event-season this is the game-level, floor-based SE
    `study_fit` returns**; the season cluster once two event-seasons exist (docs/gate-power.md's
    *Amended 2026-09-17*) is named, not implemented -- `study_fit` does not yet expose a
    season-clustered SE, and #221/#303 own the day it does:

    * **ADOPT** -- the interval's lower bound exceeds DELTA. The route back opens: a ticket
      restores `quarterback.apply` to the published path with the `-qb` mark, the separate
      partition and the track-record split kept.
    * **REMOVE** -- the interval excludes zero on the negative side. The route back closes:
      the module becomes an Exhibit under `hub.exhibits` per ADR-0007, the refuting
      measurement re-runnable, the module gone from `hub.models`.
    * **SHOW** -- anything else: excludes zero positively but the lower bound does not clear
      DELTA (a real effect too small to price), or does not exclude zero at all. Stays
      harness-only either way; the sentence names what a second event-season's
      season-clustered MDE would resolve.

    0.132 never appears here. `benchmark_reading` is a separate, informational read, called
    from `main` beside the coefficient and not from this function.
    """
    mde = fit["mde"]
    if not math.isfinite(mde) or mde > DELTA:
        needed = study_events_needed(fit.get("sd_gap", float("nan")),
                                     fit.get("floor_window", float("nan")))
        need = (f"the pilot says {needed} event games are needed to clear delta at 80% power"
                if needed is not None
                else "the event games delta needs cannot be stated from these inputs (no "
                     "usable spread across games yet)")
        mde_text = f"{mde:.4f}" if math.isfinite(mde) else "not established"
        return "NOT-RUNNABLE", (
            f"NOT RUNNABLE: the smallest coefficient this run could resolve at 80% power is "
            f"{mde_text} points per value unit against delta = {DELTA:.4f} -- the study "
            f"cannot tell a real, price-worthy effect from noise at this n. No branch below "
            f"is read; {need}.")
    lo, hi = experiment.t_interval(fit["beta"], fit["se"], int(fit["n"]))
    if lo > DELTA:
        return "ADOPT", (
            f"ADOPT: the interval's lower bound ({lo:+.4f}) exceeds delta ({DELTA:.4f} points "
            f"per value unit). The route back opens: a ticket restores quarterback.apply to "
            f"the published path with the -qb mark, the separate partition and the "
            f"track-record split kept -- the study shows the betting market moves by the "
            f"coefficient per value unit; the mark leaves only on direct evidence about the "
            f"module's own predictions.")
    if hi < 0:
        return "REMOVE", (
            f"REMOVE: the interval excludes zero on the negative side ([{lo:+.4f}, "
            f"{hi:+.4f}]) -- the betting market moving the wrong way on a quarterback "
            f"downgrade is a refutation. The route back closes: the module becomes an "
            f"Exhibit under hub.exhibits per ADR-0007, the refuting measurement re-runnable, "
            f"the module gone from hub.models.")
    if lo > 0:
        return "SHOW", (
            f"SHOW: the interval [{lo:+.4f}, {hi:+.4f}] excludes zero on the positive side "
            f"but its lower bound does not clear delta ({DELTA:.4f}) -- a real effect too "
            f"small to price. Stays harness-only; a second event-season's season-clustered "
            f"MDE is what would resolve it.")
    return "SHOW", (
        f"SHOW: the interval [{lo:+.4f}, {hi:+.4f}] does not exclude zero. Stays harness-only; "
        f"a second event-season's season-clustered MDE is what would resolve it.")


def noise_floor_per_root_day(
        parts: Sequence[tuple[int, pl.DataFrame, pl.DataFrame | None]],
) -> tuple[float, int, tuple[int, ...]]:
    """#214's own same-quarterback floor per root-day, computed the way
    `hub.fetch.odds.noise_floor_report` computes its "floor" line -- not re-derived: frozen
    lookaheads excluded first, then intervals `odds.line_moves` marks `same_qb` false or
    unknown, off a starters frame handed to that same call.

    **One `(season, polls, starters)` triple per season, because the chart is per season
    too.** `odds._starter_at` is a backward as-of join, so a poll needs a chart *of its own
    season* before it to be conditioned at all -- a chart fetched for one season and handed
    to every season's polls (#331) leaves every other season's `same_qb` null, which
    `odds.line_moves` cannot tell apart from a genuine unknown and this function used to
    drop the same way: silently, off the floor, with the run line still calling it
    same-quarterback. `polls` is a season's own slice of the archive (`archive`'s shape);
    `starters` is the depth-chart frame `odds.load_qb_starters` returns for that season, or
    `None` where the chart could not be read. Each season's live intervals are filtered on
    `same_qb` only when that season has a chart; a season with none contributes its live
    intervals unconditioned -- not dropped -- and is named in the returned `not_applied`
    tuple rather than silently counted as though excluded.

    **`starters` is never this module's own team-game rows.** `apply`'s as-of join reads
    "who was listed from this moment on"; this module's `date` is *kickoff*, so a change
    dated at kickoff is invisible to every poll of that game, which by construction all
    precede its own kickoff (#330). A starters frame built off kickoff dates therefore never
    excludes the interval it exists to exclude.

    NaN and zero with nothing left to measure across every season; `not_applied` names every
    season handed no chart (or an empty one), in the order given, whether or not it had any
    live intervals to contribute.
    """
    live_parts: list[pl.DataFrame] = []
    not_applied: list[int] = []
    for season, polls, starters in parts:
        if polls.is_empty():
            continue
        have = starters is not None and not starters.is_empty()
        moves = odds.line_moves(odds.staleness(polls), starters if have else None)
        live = moves.filter(~pl.col("frozen"))
        if have:
            live = live.filter(pl.col("same_qb"))
        else:
            not_applied.append(season)
        live_parts.append(live)
    if not live_parts:
        return float("nan"), 0, tuple(not_applied)
    combined = pl.concat(live_parts)
    if combined.is_empty():
        return float("nan"), 0, tuple(not_applied)
    floor = odds.noise_floor(combined, bootstrap=1)
    return float(floor["sd_per_root_day"]), int(floor["games"]), tuple(not_applied)


def same_quarterback_floor_label(n_seasons: int, not_applied: Sequence[int],
                                  qb_notes: Sequence[str]) -> tuple[str, str]:
    """The same-quarterback floor's label and the parenthetical naming which seasons its
    chart failed for and why (#345). `n_seasons` is how many seasons had polls to condition
    at all (`len(study_parts)`); `not_applied` and `qb_notes` are `noise_floor_per_root_day`'s
    own and the depth-chart fetch loop's, paired by position (season order).

    Three branches: *applied* -- `"same-quarterback floor"` -- where every season with polls
    also had a usable chart, or where there was nothing to try (`n_seasons == 0`); *partial*
    where some seasons' charts resolved and others did not; *not applied* --
    `"all-games floor, SAME-QUARTERBACK NOT APPLIED"` -- where none did. The parenthetical is
    empty on the first branch and names every failed season and why on the other two."""
    if n_seasons == 0 or not not_applied:
        label = "same-quarterback floor"
    elif len(not_applied) == n_seasons:
        label = "all-games floor, SAME-QUARTERBACK NOT APPLIED"
    else:
        label = "same-quarterback floor, PARTIAL"
    qb_note = (f" (same-quarterback NOT applied for "
              f"{', '.join(str(s) for s in not_applied)}: {'; '.join(qb_notes)})"
              if not_applied else "")
    return label, qb_note


class StudyReport(NamedTuple):
    """Every value the `--study` line group prints (#345), computed once so `main`'s
    `--study` block is dispatch and printing over it and nothing more. `established` is
    whether `study_rows` found an uncensored event game with both a frozen and a pre-game
    price; `fit`, `benchmark_sentence`, `n_change_seen` and `change_point_median` are only
    meaningful when it is, exactly as `rows_.is_empty()` gated them before this was a typed
    result rather than inline branches in `main`."""

    label: str
    qb_note: str
    floor: float
    floor_games: int
    used: float
    sd_gap: float
    n_gaps: int
    restatement_flag: str | None
    n_typical: int
    mde: float
    unplayed: int
    unvalued: int
    fit: dict[str, float]
    established: bool
    n_events: int
    verdict_label: str
    verdict_sentence: str
    benchmark_sentence: str | None
    n_change_seen: int
    change_point_median: Any


def study_report(study_parts: Sequence[tuple[int, pl.DataFrame, pl.DataFrame | None]],
                  qb_notes: Sequence[str], ev: pl.DataFrame, games: pl.DataFrame,
                  polls: pl.DataFrame, season_games: pl.DataFrame,
                  tg: pl.DataFrame) -> StudyReport:
    """The line-move study's report (#221), computed once as a typed result (#345) rather
    than as local variables `main`'s `--study` block built and printed from inline.
    `study_parts` and `qb_notes` are the depth-chart fetch loop's own -- the one network call
    `--study` makes, and the one thing this function does not do itself, so a season whose
    chart failed is named here without reaching the network to find out."""
    floor, floor_games, not_applied = noise_floor_per_root_day(study_parts)
    label, qb_note = same_quarterback_floor_label(len(study_parts), not_applied, qb_notes)
    used = floor if math.isfinite(floor) else 0.40
    gaps = in_season_events(ev)["gap"].drop_nulls().to_list()
    sd_gap = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
    per_season = games.group_by("season").len()["len"].to_list()
    n_typical = int(statistics.median(per_season)) if per_season else 0
    restatement_flag = gap_sd_restatement_flag(sd_gap)
    mde = study_mde(n=n_typical, sd_gap=sd_gap, window_days=7.0, floor_per_root_day=used)
    unplayed = unplayed_study_games(polls, season_games, tg)
    unvalued = unvalued_study_games(polls, season_games)
    rows_ = study_rows(polls, season_games, tg,
                       floor_per_root_day=floor if math.isfinite(floor) else None)
    fit = study_fit(rows_, floor_per_root_day=used)
    established = not rows_.is_empty()
    verdict_label, verdict_sentence = verdict(fit)
    benchmark_sentence: str | None = None
    n_change_seen = 0
    change_point_median: Any = None
    if established:
        seen_change = rows_["days_to_change"].drop_nulls()
        n_change_seen = seen_change.len()
        change_point_median = seen_change.median() if seen_change.len() else float("nan")
        lo, hi = experiment.t_interval(fit["beta"], fit["se"], int(fit["n"]))
        benchmark_sentence = benchmark_reading(fit["beta"], lo, hi)
    return StudyReport(label=label, qb_note=qb_note, floor=floor, floor_games=floor_games,
                       used=used, sd_gap=sd_gap, n_gaps=len(gaps),
                       restatement_flag=restatement_flag, n_typical=n_typical, mde=mde,
                       unplayed=unplayed, unvalued=unvalued, fit=fit, established=established,
                       n_events=rows_.height, verdict_label=verdict_label,
                       verdict_sentence=verdict_sentence, benchmark_sentence=benchmark_sentence,
                       n_change_seen=n_change_seen, change_point_median=change_point_median)
