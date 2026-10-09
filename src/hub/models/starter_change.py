"""The quarterback adjustment's own gate (#291) over the starter-change events, and the CLI
that runs it beside the line-move study (#221).

The events are built once in `hub.models.starter_events` and read twice: here, by the gate,
and by `hub.models.starter_study`, by the study and its verdict (#346). This module holds the
gate, the event-count and horizon reports, and the entry point. `python -m
hub.models.starter_change` is unchanged by the split.

**The gate** (`docs/gate-power.md`, pre-registered 2026-09-13 before this module existed):
over event games, log-loss of the frozen price moved by the shipped seam --
`nfeloqb.state(rows, as_of=<the week's first game day>)` handed to
`hub.models.quarterback.apply` on a row labelled as having no live price, exactly as
`ratings._rated_by_week` rated a played week until #299 pulled the adjustment from the
published path -- against the frozen price unmoved. Since #299 this module is the
adjustment's only reader (`tests/contracts/test_the_quarterback_adjustment_is_not_a_dependency.py`),
and the seam it replays is the one that would ship again if the gate cleared. That state
holds every team's previous game, so it prices the departing starter: the source carries a
new starter on the row of his first game and no earlier, and a replay cannot know him
before it. An **oracle** arm with the arriving starter known is reported beside it as a
diagnostic and is never read by the rule. The frozen
price is the last snapshot captured before the Eastern game day of the changed team's
previous game: the price the adjustment would actually have replaced, and not the close.
The cluster is the season; the rule is the house rule and nothing beside it; fewer than
three event-seasons is NOT-RUNNABLE ahead of every branch (ADR-0019's floor), and the
verdict sentence names the event-seasons the pilot says are needed.

**Amended 2026-09-17 (#300): this gate is a diagnostic.** No branch of it licenses ADOPT or
pulls the module any longer -- #270's pull trigger above is superseded. The ADOPT condition
is the study's coefficient (`hub.models.starter_study`) read against 0.132; this gate is
reported beside it, its own power requirement stated beside it (29 event-seasons at 80% power
against the pilot's target), and its NOT-RUNNABLE branch is an exemption from firing below
that power, not a bar to the coefficient's own verdict (`docs/gate-power.md`,
`docs/qb-adjustment.md`).

An event game with no result yet is excluded rather than priced off a truncated in-flight
snapshot, and the count is reported. Nothing here fetches but one guarded call: `--study` asks
`hub.fetch.odds.load_qb_starters` for the depth chart the study's floor conditions on, and a
season either cache does not hold is reported as not established.

    uv run python -m hub.models.starter_change --events --gate --study
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub import store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.fetch import nfeloqb, odds
from hub.ledger import Ledger
from hub.ledger import recipe as _recipe
from hub.models import experiment, quarterback
from hub.models.experiment import SEASON_CLUSTER, Harness
from hub.models.market import MARGIN_SD, normal_cdf
from hub.models.starter_events import (
    event_games,
    events,
    in_season_events,
    observed,
    pbp_team_games,
    poll_day,
    priced,
    reconcile,
    results,
    team_games,
    unreadable_games,
    with_values,
)
from hub.models.starter_study import (
    BENCHMARK,
    DELTA,
    study_report,
)

PROG = "hub.models.starter_change"

# ADR-0019: no gate in the repo runs at fewer than three seasons, and one that did should
# say so. Pre-registered as the gate's first precondition.
EVENT_SEASONS_MINIMUM = 3

# What the gate's ceiling arm is called where it prints (#138): the betting market's own
# repricing, scored against the same frozen line.
CEILING_ARM = "the betting market's own repricing, the last snapshot before the game day"

# Amended 2026-09-17 (#300): this gate is a diagnostic. None of its three branches license
# ADOPT or pull the module any longer -- the sentences below said so until this amendment,
# kept as written per docs/method.md rule 13 at docs/qb-adjustment.md and docs/gate-power.md,
# which carry what replaced them and why. The ADOPT condition is #221's line-move coefficient.
ACTIONS = experiment.Actions(
    adopt="Diagnostic only, since #300: this branch does not license shipping the module. "
          "The ADOPT condition is #221's line-move coefficient, sign and magnitude against "
          "0.132 (docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read "
          "beside it.",
    remove="Diagnostic only, since #300: this branch does not pull the module. The ADOPT "
           "condition is #221's line-move coefficient, sign and magnitude against 0.132 "
           "(docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read "
           "beside it.",
    show="Diagnostic only, since #300: this branch decides nothing about the module. The "
         "ADOPT condition is #221's line-move coefficient, sign and magnitude against 0.132 "
         "(docs/qb-adjustment.md, docs/gate-power.md); this log-loss picture is read beside "
         "it.")

# #335, ADR-0019's amendment, corrected 2026-09-21 (the ADOPTED line, item 2): this gate's
# within-season unit is a **real** no-op, by the same device `injury` and `margin` use -- one
# cluster per season, always below `TIE_MIN_CLUSTERS`, so `per_season` reads the sign alone
# and the gate stays on the rule #300/#329 pre-registered. The first version said `game_id`
# was the no-op because "the frame is one row per event-season". The *clustering* was --
# grouping one row per event groups nothing -- but the frame is one row per scored event
# *game*, 48-57 a season (`docs/gate-power.md`), four times the floor, so the tie test was
# live under a comment that said it was inert. Whether a row-level tie test belongs on this
# gate is #381's question, decided on evidence this gate does not contaminate by accident.
# The alias, not a re-spelling: `test_gates_cluster_on_the_season` refuses a second
# `("season",)` object, and the within unit here *is* the season cluster -- one claim.
WITHIN: tuple[str, ...] = SEASON_CLUSTER

# #387: this module's Harness -- one of the seven -- read by `run` through `Harness.run`.
# `ceiling_column="ceiling"` names `gate_rows`' own column, not `"ceiling_diff"`, the name
# `Harness.ceiling`'s default assumes.
HARNESS = Harness(name="quarterback_gate", arm_a="the frozen line", arm_b="quarterback-adjusted",
                  within=WITHIN, ceiling_arm=CEILING_ARM, actions=ACTIONS,
                  unit="log-loss per event game", places=4, ceiling_column="ceiling",
                  arm_roots=("hub.models.starter_change",))

def _log_loss(prob: float, y: float, eps: float = 1e-15) -> float:
    p = min(max(prob, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _home_prob(spread: float) -> float:
    """`MarketBaseline`'s conversion, so the arms differ in the spread and nothing else."""
    return normal_cdf(spread / MARGIN_SD)


# --- the gate ---------------------------------------------------------------------------------

def _moved(part: pl.DataFrame, state: pl.DataFrame, name: str) -> pl.DataFrame:
    """`quarterback.apply` over one week's event games, labelled `stale`, priced from the
    frozen line, with `state`; the moved spread comes back as `<name>`, the points added as
    `<name>_adjustment`, and `apply`'s own marker as `<name>_by` -- null on a game `apply`
    did not touch (either side missing from `state`) and the source's name on one it did.
    `<name>` itself is never null either way: `apply` leaves it at the frozen price, unmoved,
    on a game it did not touch, which is why a reader asking "did this move" reads the
    marker and not the spread."""
    games = part.select(pl.col("game_id"), pl.col("home_team"), pl.col("away_team"),
                        pl.col("frozen").alias("close_spread"),
                        pl.lit("stale").alias("price_source"))
    moved = quarterback.apply(games, state)
    return part.join(moved.select(pl.col("game_id"), pl.col("close_spread").alias(name),
                                  pl.col("qb_adjustment").alias(f"{name}_adjustment"),
                                  pl.col("adjusted_by").alias(f"{name}_by")),
                     on="game_id", how="left")


def _adjusted(frame: pl.DataFrame, rows: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """The two arms, per week.

    **`adjusted` is the gate's arm and it is the shipped seam**: `nfeloqb.state(rows,
    as_of=<the week's first game day>)` -- rows strictly before that day (#272) -- handed
    to `quarterback.apply`, exactly as `hub.models.ratings._rated_by_week` rated a week that
    had kicked off until #299 pulled the adjustment. That state's latest row for every team
    is its *previous* game, so on an event game the arm prices the departing starter's
    adjustment: the source's file carries a new starter on the row of his first game and on
    no earlier row, and the replay cannot know him before it. Until the review of
    2026-09-13 this arm read the event game's own row -- the arriving starter known with
    certainty -- which is a better estimator than the one being gated, and a `diff` biased
    toward ADOPT.

    **`oracle` is a diagnostic and not the gate**: the same estimator with the week's own
    rows as the state, the arriving starter known. It answers what the mechanism could do
    if the state were timely, which #270's disposition needs beside the gate's answer, and
    it is never read by the rule.
    """
    out = []
    for (season, week), part in frame.group_by(["season", "week"], maintain_order=True):
        this = tg.filter((pl.col("season") == season) & (pl.col("week") == week))
        first_day = dt.date.fromisoformat(str(this["date"].min()))
        shipped = nfeloqb.state(rows, as_of=first_day)
        known = this.select(pl.col("team"), pl.col("qb"), pl.col("value").alias("qb_value"),
                            pl.col("adj").alias("qb_adj"),
                            pl.lit(0, dtype=pl.Int64).alias("tenure"),
                            pl.col("date").alias("as_of")).sort("team")
        out.append(_moved(_moved(part, shipped, "adjusted"), known, "oracle"))
    return pl.concat(out)


PAIRED_SCHEMA: dict[str, Any] = {
    "game_id": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8,
    "away_team": pl.Utf8, "net_gap": pl.Float64, "frozen": pl.Float64,
    "adjusted_adjustment": pl.Float64, "adjusted": pl.Float64,
    "oracle_adjustment": pl.Float64, "oracle": pl.Float64, "close": pl.Float64,
    "y": pl.Float64, "diff": pl.Float64, "oracle_diff": pl.Float64, "ceiling": pl.Float64,
}


def gate_rows(polls: pl.DataFrame, games: pl.DataFrame, tg: pl.DataFrame,
              rows: pl.DataFrame) -> pl.DataFrame:
    """The paired frame the gate reads: one row per scored, uncensored event game.

    `diff` is log-loss of the frozen price minus log-loss of the shipped arm's, positive
    when the adjustment helped; `oracle_diff` the same for the diagnostic arm, never read by
    the rule; `ceiling` is log-loss of the frozen price minus that of the last snapshot
    before the game day -- what the betting market's own repricing recovered, the declared
    ceiling arm. All on the home result, ties 0.5, through `MarketBaseline`'s conversion.
    `rows` are the source's rows the shipped seam builds its state from.

    **The not-null guard is on `adjusted_by` and `oracle_by`, the adjustment's own markers,
    never on `adjusted` or `oracle` themselves.** `quarterback.apply` leaves the spread
    column at the frozen price -- unmoved, and therefore never null -- on a game it did not
    touch, so a game whose team has no prior row in the state (a debut, or a name the source
    never carries before this week) used to survive this filter and enter at a manufactured
    zero difference, inflating n and shrinking both the mean and the clustered sd. The
    marker is null exactly where `apply` did not touch the row, on either arm, so excluding
    on it drops the game rather than counting it as no effect.
    """
    have = priced(polls, games).filter(~pl.col("censored") & pl.col("close").is_not_null())
    have = have.join(results(tg).select("game_id", "y"), on="game_id", how="inner")
    if have.is_empty():
        return pl.DataFrame(schema=PAIRED_SCHEMA)
    have = _adjusted(have, rows, tg).filter(pl.col("adjusted_by").is_not_null()
                                            & pl.col("oracle_by").is_not_null())
    diff, oracle, ceiling = [], [], []
    for r in have.iter_rows(named=True):
        base = _log_loss(_home_prob(r["frozen"]), r["y"])
        diff.append(base - _log_loss(_home_prob(r["adjusted"]), r["y"]))
        oracle.append(base - _log_loss(_home_prob(r["oracle"]), r["y"]))
        ceiling.append(base - _log_loss(_home_prob(r["close"]), r["y"]))
    return (have.with_columns(pl.Series("diff", diff, dtype=pl.Float64),
                              pl.Series("oracle_diff", oracle, dtype=pl.Float64),
                              pl.Series("ceiling", ceiling, dtype=pl.Float64))
                .select(*PAIRED_SCHEMA))


def pilot(tg: pl.DataFrame, games: pl.DataFrame) -> dict[str, Any]:
    """The power input the pre-registration names: the source's own base and
    quarterback-adjusted probabilities scored on the same event games, per season. The
    target is the absolute mean of the season means -- the effect there is to find if this
    estimator reproduces the source's gain -- and `season_sd` their spread, NaN with fewer
    than two seasons. Not a verdict: the source's arm on nflfastR outcomes, not this
    estimator on a frozen line."""
    scored = (games.select("game_id", "season")
                   .join(results(tg), on="game_id", how="inner")
                   .drop_nulls(["base_prob", "qb_prob"]))
    per = []
    for season, part in scored.group_by("season", maintain_order=True):
        diffs = [_log_loss(b, y) - _log_loss(q, y)
                 for b, q, y in zip(part["base_prob"], part["qb_prob"], part["y"], strict=True)]
        per.append({"season": int(season[0]), "n": len(diffs),
                    "mean": statistics.fmean(diffs),
                    "sd": statistics.stdev(diffs) if len(diffs) > 1 else float("nan")})
    per.sort(key=lambda d: d["season"])
    means = [d["mean"] for d in per]
    return {"seasons": len(per), "n": scored.height, "per_season": per,
            "target": abs(statistics.fmean(means)) if means else float("nan"),
            "season_sd": statistics.stdev(means) if len(means) > 1 else float("nan")}


def event_seasons_needed(target: float, season_sd: float, *, cap: int = 100) -> int | None:
    """The smallest number of event-seasons at which the MDE is at or below the target, or
    None when none up to `cap` reaches it -- or when either input is not a number. The MDE at
    `k` event-seasons is `(t(0.975, k-1) + z(0.80)) * s / sqrt(k)`, on the t reference
    `docs/gate-power.md` restated on 2026-09-07: `experiment.smallest_n_resolving` handed the
    season-clustered standard error `s / sqrt(k)` (#346; the study's events-needed search is
    the same function over its own standard error)."""
    if not (math.isfinite(target) and math.isfinite(season_sd)) or target <= 0:
        return None
    return experiment.smallest_n_resolving(
        target, lambda k: season_sd / math.sqrt(k), cap=cap)


def run(paired: pl.DataFrame, *, needed: int | None, ceiling: bool = True,
        ledger: Ledger | None = None) -> experiment.GateRun:
    """The pre-registered precondition first, then -- only if it clears -- one gate run
    through `experiment.run_gate`. `ceiling` hands the declared arm's rows to the rule so
    stage 2 can fire.

    **Fewer than `EVENT_SEASONS_MINIMUM` event-seasons is NOT-RUNNABLE ahead of the summary,
    the house verdict, the width history and the rendered lines -- not a verdict string
    swapped in after they have already run.** `run_gate` bootstraps an interval, decides
    ADOPT/REMOVE/SHOW, renders the CI and the stamp, and records this run's width through
    `ledger` as a side effect of being called at all; computing any of that from an
    underpowered frame and then only replacing the verdict sentence would publish the
    interval it exists to keep unpublished and record a width history entry for a run with
    no license to have one. So the count is read directly off `paired` and, below the
    minimum, `run_gate` is never called: the verdict is NOT-RUNNABLE, `lines` is empty, and
    `ledger` is never touched.

    **`ledger` replaces `width_path`/`record_width` (#385)**, the single parameter `run_gate`
    itself takes now: `None` (the default) is `run_gate`'s own default, a file-backed
    `hub.ledger.Ledger()` writing to `hub.ledger.WIDTH_STATE`. `main` builds its own explicit
    `Ledger` reading `experiment.WIDTH_STATE` at call time -- not this function's default,
    which is bound once -- so a test that monkeypatches `experiment.WIDTH_STATE` still
    isolates the CLI from the repo's own history the way it always could.
    """
    k = paired["season"].n_unique() if paired.height else 0
    if k < EVENT_SEASONS_MINIMUM:
        need = (f"the pilot says {needed} event-seasons are needed at 80% power"
                if needed is not None else "the pilot cannot say how many event-seasons are "
                                           "needed (no spread across seasons yet)")
        verdict = (
            "NOT-RUNNABLE",
            f"NOT RUNNABLE: {k} event-season(s) in the archive against a pre-registered "
            f"minimum of {EVENT_SEASONS_MINIMUM} (ADR-0019: no gate runs at fewer than three "
            f"seasons); {need}. No verdict is read. Since #300 this diagnostic's NOT-RUNNABLE "
            f"is an exemption from firing below the minimum, not a bar to the module's ADOPT "
            f"condition, which is #221's line-move coefficient (docs/qb-adjustment.md); this "
            f"is not-runnable, not a null.")
        return experiment.GateRun(summary={}, seasons=pl.DataFrame(), verdict=verdict,
                                  lines=[], stamped=paired)
    # An empty frame, not None: `Harness.run` reads None as "use `paired`", and a run
    # without --ceiling must reach the gate with no bound, as it always has.
    return HARNESS.run(
        paired, ceiling_frame=paired if ceiling else pl.DataFrame(), ledger=ledger,
        recipe=_recipe(ceiling=ceiling))


def archive(season: int, base: Path | None) -> pl.DataFrame:
    """The season's polls off the store, in the shape the readers take; empty on a fresh
    clone. `store.lines` is the one reader of every poll; this takes its four columns."""
    return store.lines(season, base=base).select("game_id", "close_spread", "captured_at", "week")


class SeasonEventSummary(NamedTuple):
    """One season's line in the event-count report (#345): the in-season changes, the event
    games `games` carries for the season, the gap sd (`nan` under two gaps) and how many
    gaps it is over, and the offseason changes `in_season` flagged out and did not count."""

    season: int
    changes: int
    games: int
    gap_sd: float
    n_gaps: int
    offseason: int


def season_event_summaries(ev: pl.DataFrame, games: pl.DataFrame) -> list[SeasonEventSummary]:
    """One `SeasonEventSummary` per season `ev` carries, in season order -- the computation
    `_event_lines` used to do inline (#345), split out so a test can assert on the numbers
    rather than parsing the sentence they are printed into."""
    out = []
    for season in sorted(set(ev["season"].to_list())):
        part = ev.filter(pl.col("season") == season)
        ins = part.filter(pl.col("in_season"))
        off = part.height - ins.height
        gaps = ins["gap"].drop_nulls().to_list()
        sd = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
        n_games = games.filter(pl.col("season") == season).height
        out.append(SeasonEventSummary(season=season, changes=ins.height, games=n_games,
                                       gap_sd=sd, n_gaps=len(gaps), offseason=off))
    return out


def _event_lines(ev: pl.DataFrame, games: pl.DataFrame, since: int, *,
                 source: str = "the source's starter column",
                 beyond: Mapping[int, Beyond] | None = None) -> list[str]:
    """The event-count report. `beyond` names each season whose played games extend past
    what `source` has seen (#420): that season's count is not a count of the season, so it
    is printed as **not established past <date>**, with what was seen through that date
    beside it, rather than as a number that reads the same whether or not anything
    happened after it -- rule 18's shape, and what "2026: 0 changes" was."""
    beyond = beyond or {}
    lines = [f"  starter changes since {since}, observed off {source} (never the injury "
             f"report); an event is a change between two games of one season:"]
    summaries = {s.season: s for s in season_event_summaries(ev, games)}
    for season in sorted(summaries.keys() | beyond.keys()):
        s = summaries.get(season) or SeasonEventSummary(season, 0, 0, float("nan"), 0, 0)
        seen = (f"{s.changes} changes on {s.games} event games, gap sd {s.gap_sd:.1f} value "
                f"units (n={s.n_gaps}); {s.offseason} offseason change(s) not events")
        if season in beyond:
            past = beyond[season]
            through = past.through if past.through is not None else "its first game"
            after = ("games after it are not in the source" if past.unseen is None else
                     f"{past.unseen} played team-game(s) after it have no observed starter")
            lines.append(f"    {season}: NOT ESTABLISHED past {through} -- {after}; "
                         f"through it: {seen}")
        else:
            lines.append(f"    {season}: {seen}")
    return lines


class Beyond(NamedTuple):
    """A season whose played games run past a source's horizon: the last game day the source
    has a starter for (`through`, None when it has none), and how many played team-games lie
    beyond it (None where the source cannot say, the pinned file having no schedule)."""

    through: str | None
    unseen: int | None


def beyond_event_horizon(tg: pl.DataFrame) -> dict[int, Beyond]:
    """Per season, whether a played team-game has no observed starter (#420): the schedule
    says the game has a result and the source names nobody. The horizon is the season's last
    game day with a starter. A season whose every played game has one is absent -- it is
    established; a season with no played game is too, because nothing was played past
    anything."""
    out: dict[int, Beyond] = {}
    for season in sorted(set(tg["season"].to_list())):
        part = tg.filter(pl.col("season") == season)
        unseen = part.filter(pl.col("score").is_not_null() & pl.col("qb").is_null()).height
        if unseen:
            through = part.filter(pl.col("qb").is_not_null())["date"].max()
            out[int(season)] = Beyond(None if through is None else str(through), unseen)
    return out


def beyond_price_horizon(polls: pl.DataFrame, tg: pl.DataFrame) -> dict[int, Beyond]:
    """The same question for prices: per season with polls, the played team-games whose game
    day is after the last poll's Eastern day (#420). A game after the archive's last poll
    cannot have a frozen price, whatever the events say; `through` is that last poll day."""
    out: dict[int, Beyond] = {}
    if polls.is_empty():
        return out
    last = (polls.with_columns(poll_day().alias("_day"))
                 .join(tg.select("game_id", "season").unique(), on="game_id", how="inner")
                 .group_by("season").agg(pl.col("_day").max().alias("through")))
    for season, through in zip(last["season"], last["through"], strict=True):
        late = tg.filter((pl.col("season") == season) & pl.col("score").is_not_null()
                         & (pl.col("date") > through)).height
        if late:
            out[int(season)] = Beyond(str(through), late)
    return out


def horizon_line(season: int, study_tg: pl.DataFrame, file_tg: pl.DataFrame,
                 polls: pl.DataFrame) -> str:
    """Each source's horizon beside the season's played weeks (#420): the last game with a
    result, the last game a starter was observed for, the pinned file's last game, and the
    archive's last poll day. The four dates in one sentence, because a count printed without
    them cannot be told from a count of a season the sources have not reached."""
    mine = study_tg.filter(pl.col("season") == season)
    played = mine.filter(pl.col("score").is_not_null())
    last_played = played["date"].max() if played.height else None
    seen = mine.filter(pl.col("qb").is_not_null())["date"].max()
    file_last = file_tg.filter(pl.col("season") == season)["date"].max()
    ours = polls.filter(pl.col("game_id").str.starts_with(f"{season}_"))
    last_poll = None if ours.is_empty() else ours.select(poll_day().max().alias("d"))["d"][0]
    last_week = played["week"].max()
    week = last_week if isinstance(last_week, int) else None
    return (f"  horizons {season}: played through {last_played} (week {week}); starters "
            f"observed through {seen}; the pinned file's last game {file_last}; the "
            f"archive's last poll {last_poll}")


class StudyEvents(NamedTuple):
    """The events the study and the event counts read, and what to say about where they came
    from: `tg` the team-games they were built on (results and dates included), `ev` the
    events with the pinned file's values joined, `games` the event games, `source` a phrase for
    the report, `beyond` the seasons whose played games run past the source, `notes` the lines
    to print first."""

    tg: pl.DataFrame
    ev: pl.DataFrame
    games: pl.DataFrame
    source: str
    beyond: dict[int, Beyond]
    notes: list[str]


def study_events(a: argparse.Namespace, rows: pl.DataFrame, tg: pl.DataFrame,
                 ev: pl.DataFrame, games: pl.DataFrame, seasons: Sequence[int]) -> StudyEvents:
    """The study's event source (#420, ADOPTED (B) on #421): the passer of each team's first
    pass play, off the nflverse store, reconciled against the pinned file where both exist.
    `tg`, `ev` and `games` are the pinned file's own, handed back unchanged when the study is
    not asked for (the gate alone reads only the file) and when play-by-play cannot be read --
    in which case the season in progress is named as not established past the file's last
    game, because a pinned file cannot hold a change made after its pin, and a count of zero
    from it is a fact about the file."""
    file_only = StudyEvents(tg, ev, games, "the pinned file's starter column", {}, [])
    if not (a.events or a.study):
        return file_only
    try:
        got = pbp_team_games(seasons, cache=None)
    except Exception as exc:
        through = tg["date"].max()
        return file_only._replace(
            beyond={s: Beyond(None if through is None else str(through), None)
                    for s in seasons if s >= SEASON_AHEAD},
            notes=[f"  play-by-play unavailable ({type(exc).__name__}: {exc}); events read off "
                   f"the pinned file, which cannot hold a change made after its pin"])
    notes = [f"  {name} refresh failed; served last-good from the cache" for name in got.stale]
    pbp_ev = events(observed(got.tg))
    rec = reconcile(got.tg, tg)
    notes.append(f"  reconciliation against the pinned file: over {rec.compared} team-games both "
                 f"observe, {rec.file} in-season change(s) by the file, {rec.pbp} by "
                 f"play-by-play, {rec.both} by both")
    valued = with_values(pbp_ev, tg)
    return StudyEvents(got.tg, valued, event_games(in_season_events(valued)),
                       "the first pass play's passer in play-by-play",
                       beyond_event_horizon(got.tg), notes)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Starter-change events, the quarterback adjustment's gate over the "
                    "frozen line (#291), and the line-move study (#221). The gate reads "
                    "the pinned nfeloqb file; the events and the study read play-by-play "
                    "(#420), refreshing the season in progress and serving last-good if the "
                    "wire is down.")
    ap.add_argument("--events", action="store_true", help="count the events per season")
    ap.add_argument("--gate", action="store_true",
                    help="run the pre-registered gate on the archive's event games")
    ap.add_argument("--ceiling", action=argparse.BooleanOptionalAction, default=True,
                    help=f"hand the rule the declared ceiling arm ({CEILING_ARM}); on by "
                         f"default, the shipped configuration -- pass --no-ceiling to turn "
                         f"stage 2 off (#302)")
    ap.add_argument("--study", action="store_true", help="run the line-move study")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD,
                    help="the last season whose snapshot archive is read; every season from "
                         "--since through this one is assembled into one run (default "
                         "SEASON_AHEAD)")
    ap.add_argument("--since", type=int, default=2022,
                    help="the first season the event counts, the pilot cover, and the "
                         "gate's archive assembles from")
    ap.add_argument("--cache", type=Path, default=None, help="the nfeloqb cache directory")
    ap.add_argument("--store", type=Path, default=None,
                    help="the processed store; omit to read the repo's, which also unions "
                         "the committed state/odds snapshots (#383). A store elsewhere reads "
                         "the _state/odds tree inside it, not the repo's")
    a = ap.parse_args(argv)
    if not (a.events or a.gate or a.study):
        ap.print_usage()
        return 2
    if a.season < a.since:
        print(f"{PROG}: --season {a.season} is before --since {a.since}, so there is no "
              f"season for the gate's archive to assemble", file=sys.stderr)
        return 2

    rows = nfeloqb.read_rows(a.cache)
    if rows is None:
        return unavailable(PROG, "the cached nfeloqb file",
                           FileNotFoundError("nothing cached; run hub.fetch.nfeloqb --refresh"))
    since_rows = rows.filter(pl.col("season") >= a.since)
    # The pinned file's own events: the gate's, and the adjustment's. Never moved by this
    # ticket -- the study's events are the next block's (#420).
    tg = team_games(since_rows)
    ev = events(tg)
    games = event_games(in_season_events(ev))
    seasons = list(range(a.since, a.season + 1))
    study = study_events(a, rows, tg, ev, games, seasons)
    for line in study.notes:
        print(line)
    for line in _event_lines(study.ev, study.games, a.since, source=study.source,
                             beyond=study.beyond):
        print(line)
    unreadable = unreadable_games(since_rows)
    print(f"  {unreadable} team-game(s) since {a.since} a blank side made unreadable (that "
          f"side dropped; the previous-game link is still built off the full schedule)")

    # #302: read every season's archive from --since through --season and concatenate,
    # rather than only --season's -- with `games` filtered to match. A gate over one
    # season's polls can never reach the three-season floor no matter how many the caches
    # hold; assembling the whole asked-for range is what lets it.
    parts = [archive(s, a.store) for s in seasons]
    have_seasons = [s for s, p in zip(seasons, parts, strict=True) if not p.is_empty()]
    polls = pl.concat(parts)
    span = str(a.season) if a.since == a.season else f"{a.since}-{a.season}"
    if polls.is_empty():
        print(f"  no snapshot archive for {span} here: the gate and the study have no "
              f"frozen price to read, and both are not established")
    else:
        print(f"  archive: {polls['game_id'].n_unique()} games, "
              f"{polls['captured_at'].n_unique()} polls, "
              f"{polls['captured_at'].min():%Y-%m-%d} to {polls['captured_at'].max():%Y-%m-%d}, "
              f"{len(have_seasons)} of {len(seasons)} season(s) asked for have polls "
              f"({', '.join(str(s) for s in have_seasons)})")
    # #420: `store.lines` unions the committed `state/odds/` tree (#383) under the repo's own
    # store -- and a `--store` somewhere else carries its own `_state/odds`, so it is counted
    # here, where a run that read none looks different from one that read them.
    committed = store.committed_snapshots(a.store)
    print(f"  committed snapshots read: {committed} file(s) under "
          f"{store.snapshot_root(a.store)}"
          + ("" if committed or a.store is None else
             " (a --store carries its own _state/odds; omit --store to read the repo's)"))
    season_games = games.filter(pl.col("season").is_in(seasons))
    study_games = study.games.filter(pl.col("season").is_in(seasons))
    seen = priced(polls, study_games)
    never_polled = int(seen["never_polled"].sum())
    polled_after = int((seen["censored"] & ~seen["never_polled"]).sum())
    print(f"  {span}: {study_games.height} event games; "
          f"{int(seen['censored'].sum())} censored (no snapshot before the change could be "
          f"known: {never_polled} never polled at all, {polled_after} polled only after the "
          f"change), {seen.filter(~pl.col('censored') & pl.col('close').is_not_null()).height} "
          f"with a frozen price and a pre-game one, of which "
          f"{int(seen['windowless'].sum())} windowless (the frozen poll is the only one before "
          f"the game day, so no move was observed)")
    print(horizon_line(a.season, study.tg, tg, polls))
    for season, past in sorted(beyond_price_horizon(polls, study.tg).items()):
        print(f"  prices: {season} NOT ESTABLISHED past {past.through} -- {past.unseen} "
              f"played team-game(s) after the archive's last poll cannot have a frozen price")

    if a.gate:
        pil = pilot(tg, games)
        needed = event_seasons_needed(pil["target"], pil["season_sd"])
        print(f"  pilot (the source's own two columns on event games, {pil['n']} games over "
              f"{pil['seasons']} seasons): target {pil['target']:.4f} log-loss, between-season "
              f"sd {pil['season_sd']:.4f}; event-seasons needed at 80% power: "
              f"{needed if needed is not None else 'more than 100, or not computable'}")
        for d in pil["per_season"]:
            print(f"    {d['season']}: n={d['n']}, mean {d['mean']:+.4f}, sd {d['sd']:.4f}")
        paired = gate_rows(polls, season_games, tg, rows)
        # The width history is kept only when a run has an interval to keep; a run over
        # no rows would file a NaN width under the gate's name.
        # The width history is kept only when a run has an interval to keep; a run over no
        # rows would file a NaN width under the gate's name. `experiment.WIDTH_STATE`, read
        # here rather than defaulted on `Ledger.__init__`, so a test that monkeypatches it
        # still isolates this call from the repo's own history.
        got = run(paired, needed=needed, ceiling=a.ceiling,
                 ledger=Ledger(experiment.WIDTH_STATE, write=paired.height > 0))
        print(f"  gate: {paired.height} scored event games over "
              f"{paired['season'].n_unique() if paired.height else 0} event-season(s); the "
              f"arm is the shipped seam, nfeloqb.state as of the week's first game day")
        if paired.height:
            print(f"  diagnostic, not the gate -- the oracle arm (arriving starter known): "
                  f"mean {paired['oracle_diff'].mean():+.4f} log-loss against the shipped "
                  f"arm's {paired['diff'].mean():+.4f} over n={paired.height}")
        for line in got.lines:
            print(line)
        print(f"  {got.verdict[0]}: {got.verdict[1]}")

    if a.study:
        # #330: the depth chart #214's own floor conditions on -- never this module's own
        # kickoff-dated rows, which a change can never be seen through (a game's polls all
        # precede its own kickoff). #331: one chart per season in the assembled range, a
        # bounded loop over `seasons` (never over teams or games) -- `odds._starter_at`'s
        # as-of join means a chart fetched for one season cannot condition another season's
        # polls, so the last season's chart alone left every earlier season's intervals
        # silently unconditioned. Each fetch is guarded exactly as `odds.noise_floor_report`
        # guards its own, and skipped for a season with no polls to condition.
        study_parts: list[tuple[int, pl.DataFrame, pl.DataFrame | None]] = []
        qb_notes: list[str] = []
        for s, p in zip(seasons, parts, strict=True):
            if p.is_empty():
                continue
            chart: pl.DataFrame | None = None
            try:
                chart = odds.load_qb_starters(s)
            except Exception as exc:                         # pragma: no cover - network
                qb_notes.append(f"{s} ({type(exc).__name__}: {exc})")
            study_parts.append((s, p, chart))
        rep = study_report(study_parts, qb_notes, study.ev, study.games, polls,
                           study_games, study.tg)
        print(f"  study: {rep.label} {rep.floor:.3f} points per root-day off {rep.floor_games} "
              f"live games of this archive{rep.qb_note} (#214 recorded 0.40 on 12; used "
              f"{rep.used:.2f}); gap sd {rep.sd_gap:.1f} value units over {rep.n_gaps} events "
              f"since {a.since} against delta = {DELTA:.4f} points per value unit")
        # #329's pre-registered restatement trigger: a print beside delta, never a branch.
        if rep.restatement_flag is not None:
            print(f"  {rep.restatement_flag}")
        print(f"  MDE before the run at a season of {rep.n_typical} event games, a 7-day "
              f"window: {rep.mde:.4f} points per value unit against a benchmark of "
              f"{BENCHMARK:.3f}")
        print(f"  {rep.unvalued} uncensored, priced event game(s) excluded as unvalued -- the "
              f"pinned file holds no value for a starter who changed, so the game has no "
              f"regressor and is refused rather than read as a zero gap")
        print(f"  {rep.unplayed} uncensored, priced event game(s) excluded as unplayed -- an "
              f"in-flight game's last snapshot before its own game day is not the last one "
              f"before a move that has finished happening")
        if not rep.established:
            print(f"  coefficient: not established -- no uncensored event game with a frozen "
                  f"and a pre-game price in the {span} archive")
            # #329: the NOT-RUNNABLE path still reads through `verdict` -- `study_fit` on no
            # rows hands back an MDE that is not finite, which is `verdict`'s own
            # NOT-RUNNABLE condition, not a special case wired around it here.
            print(f"  {rep.verdict_label}: {rep.verdict_sentence}")
        else:
            fit = rep.fit
            print(f"  coefficient: {fit['beta']:+.4f} points per value unit (se {fit['se']:.4f} "
                  f"from the floor, MDE {fit['mde']:.4f}) over n={int(fit['n'])} event games "
                  f"(cluster = game); t against {BENCHMARK:.3f}: {fit['t_vs_benchmark']:+.2f}")
            print(f"  0.132: {rep.benchmark_sentence}")
            print(f"  {rep.verdict_label}: {rep.verdict_sentence}")
            print(f"  change-point: seen on {rep.n_change_seen} of {rep.n_events} games, "
                  f"median {rep.change_point_median} days after the previous game day; the "
                  f"depth-chart date is not cached, so timing against the report date is not "
                  f"established")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
