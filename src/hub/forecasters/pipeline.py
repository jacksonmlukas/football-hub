"""One day's forecast run, assembled from the adapters (#447; build step 6).

`docs/prereg/llm-forecasters.md`, *Forecast procedure*, in the order the safety argument
needs:

1. **Keys first.** Every provider the config names gets its client built -- and so its key
   read -- before anything is fetched: a missing key refuses before a credit is spent.
2. **The day's games** -- nflverse's schedule, filtered by `days.due(kickoffs, day)`.
3. **Nothing left to do is nothing spent.** If every due game already has all eleven records
   (or a logged exclusion), the run returns before the betting-market snapshot, so a rerun writes
   nothing and spends no Odds API credit.
4. **One market snapshot** for the day (`market.snapshot("forecast")`). Its balance file is the
   study's own, `ODDS_STATE` under the `forecasts` checkout (#445, adopted (a)), never the
   hub's `state/odds.json`: the hub's polls and the study's pulls re-read the same account
   balance from `x-requests-remaining` each time, and the two floors absorb the lag.
5. **Per game, a blind and a line-aware packet** (`packet.build_packet`) from the schedule,
   the injury report and the starting-QB source -- the latest nfeloqb commit before run start
   (`qb.qb_source_at`), not the pin. A refused packet becomes an `exclusions.jsonl` line.
6. **The hub contestant** over `hub.models.ratings` (#446), logging its commit.
7. **`runner.run`**: lock, idempotency, the spending guard, then the models -- and, for the
   paired retrieval arm (#511, `hub.forecasters.retrieval`), each game's news searches, made
   lazily inside the lock so a rerun or a refused budget spends no Tavily credit. The Tavily
   client is built in step 1 with the model clients, so a missing `TAVILY_API_KEY` refuses
   before the snapshot too. Its credit count is the study's own file, `TAVILY_STATE`.

**Graceful degradation.** A failed schedule, injury or QB fetch serves the last good copy
(`last_good.serve`), which still has to pass the as-of checks. A failed market snapshot does
not stop the blind half: blind forecasts, the home constant and the coin flip are written,
the line-aware and hub contestants wait, and the run reports `partial` -- so the workflow does
not mark the day done and the next trigger fills the gaps with a fresh snapshot.

Everything that touches the world is a field of `Deps`, so the whole day dry-runs on a fake
client and recorded payloads; `production` is the one place the real sources are named.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import paths
from hub.fetch import odds, tavily
from hub.forecasters import days, last_good, market, qb, retrieval, runner, sources
from hub.forecasters.config import StudyConfig
from hub.forecasters.inputs import (
    GameInfo,
    InjuryReport,
    QbSource,
    Schedule,
    SourceStamp,
)
from hub.forecasters.packet import Condition, PacketRefused, build_packet

# The study's own Odds API balance file, relative to the `forecasts` checkout (#445 (a)).
ODDS_STATE = Path("state") / "odds.json"
# The study's Tavily credit count, beside it (#511): credits are not dollars.
TAVILY_STATE = retrieval.CREDITS_STATE
LAST_GOOD = paths.PROCESSED / "forecasters"


@dataclass(frozen=True)
class Outcome:
    """What a run did: `status` is "done", "nothing_due", "nothing_left" or "partial"."""
    status: str
    games: tuple[str, ...] = ()
    snapshot_id: str | None = None
    path: Path | None = None
    excluded: tuple[str, ...] = ()
    why: str | None = None


@dataclass
class Deps:
    """Every reach outside the process, injectable. `client` is built by `make_client` from
    the config, so the keys are read before anything else happens."""
    schedule_frame: Callable[[int], pl.DataFrame]
    injuries_payload: Callable[[], Mapping[str, Any]]
    qb_source: Callable[[datetime, SourceStamp], QbSource]
    make_client: Callable[[StudyConfig], runner.Client]
    make_hub: Callable[[int, int], runner.HubModel]
    team_names: Callable[[Mapping[str, GameInfo]], Mapping[str, str]]
    odds_get: market.Get | None = None
    odds_key: str | None = None
    # The retrieval arm's search client, built (and keyed) with the model clients. The default
    # is the real one, so a config with the arm and no key refuses rather than skipping it.
    make_searcher: Callable[[StudyConfig], retrieval.Searcher] = field(
        default=lambda cfg: tavily.Client())
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    run_id: str = "local"
    last_good_root: Path = LAST_GOOD


def production(*, run_id: str) -> Deps:                             # pragma: no cover - network
    from hub.forecasters import providers
    return Deps(schedule_frame=sources.fetch_schedule,
                injuries_payload=sources.fetch_espn_injuries,
                qb_source=lambda rs, stamp: qb.qb_source_at(rs, stamp=stamp),
                make_client=lambda cfg: providers.Router(cfg),
                make_hub=lambda season, week: sources.RatingsHub(season, week),
                team_names=sources.team_names, run_id=run_id)


def _left(due: list[str], config: StudyConfig, out: Path) -> bool:
    """Whether any due game still lacks a record or an exclusion."""
    seen = runner.written(out)
    excluded = runner.excluded(out)
    names = runner.contestants(config)
    return any(g not in excluded and any((g, n) not in seen for n in names) for g in due)


def run_day(day: date, out: Path, *, config: StudyConfig, deps: Deps) -> Outcome:
    """Forecast `day`'s games into `out` (the `forecasts` checkout)."""
    client = deps.make_client(config)                       # 1. keys, before any spend
    searcher = deps.make_searcher(config) if runner.paired_spec(config) is not None else None
    run_start = deps.clock().astimezone(UTC)
    season = sources.season_of(day)

    def stamp(name: str) -> SourceStamp:
        return SourceStamp(name, deps.clock().astimezone(UTC))

    root = deps.last_good_root
    frame = deps.schedule_frame(season)                     # 2. the day's games
    infos = sources.game_infos(frame)
    due = days.due({g: i.kickoff for g, i in infos.items()}, day)
    if not due:
        return Outcome("nothing_due")
    if not _left(due, config, out):                         # 3. a rerun spends nothing
        return Outcome("nothing_left", tuple(due))

    sched = last_good.serve("schedule",
                            lambda: sources.schedule_from_nflverse(frame,
                                                                   stamp(sources.SCHEDULE_SOURCE)),
                            Schedule, root).value
    injuries = last_good.serve(
        "injuries", lambda: sources.injuries_from_espn(deps.injuries_payload(),
                                                       run_start=run_start,
                                                       stamp=stamp(sources.INJURY_SOURCE)),
        InjuryReport, root).value
    qbs = last_good.serve("qb", lambda: deps.qb_source(run_start, stamp("nfeloqb:qb_elos.csv")),
                          QbSource, root).value

    snap: market.Snapshot | None = None                     # 4. one snapshot, own balance file
    why: str | None = None
    games = {g: infos[g] for g in due}
    try:
        snap = market.snapshot("forecast", now=deps.clock(), season=season, get=deps.odds_get,
                               key=deps.odds_key, state_path=out / ODDS_STATE,
                               match=sources.matcher(games, deps.team_names(games)))
    except (odds.QuotaFloor, odds.MarketUnreachable, ValueError) as exc:
        why = f"no market snapshot: {exc}"

    inputs: list[runner.GameInputs] = []                    # 5. packets
    for gid in due:
        info = infos[gid]
        common: dict[str, Any] = {"run_start": run_start, "schedule": sched, "qb_source": qbs,
                                  "injuries": injuries}
        try:
            blind = build_packet(info, condition=Condition.BLIND, **common)
        except PacketRefused as exc:
            inputs.append(runner.GameInputs(gid, info.kickoff, None, None, refusal=str(exc)))
            continue
        aware = None
        quote = snap.for_packet(gid) if snap is not None else None
        if snap is not None and quote is not None:
            try:
                aware = build_packet(info, condition=Condition.LINE_AWARE, market=quote,
                                     snapshot_id=snap.snapshot_id, **common)
            except PacketRefused:
                aware = None
        inputs.append(runner.GameInputs(gid, info.kickoff, blind, aware))

    week = max(i.week for i in games.values())
    hub = deps.make_hub(season, week)                       # 6. the hub contestant
    retriever = None
    if searcher is not None and config.retrieval is not None:
        full = {abbr: name for name, abbr in deps.team_names(games).items()}
        retriever = retrieval.Retriever(searcher, full_names=full, state_path=out / TAVILY_STATE,
                                        season_credit_cap=config.retrieval.season_credit_cap,
                                        clock=deps.clock)
    path = runner.run(inputs, snapshot_id=snap.snapshot_id if snap else None, config=config,
                      client=client, hub=hub, out_dir=out, run_start=run_start,
                      run_id=deps.run_id, clock=deps.clock, retriever=retriever)  # 7.
    excluded = tuple(g.game_id for g in inputs if g.refusal is not None)
    if retriever is not None and retriever.exhausted:
        why = "; ".join(w for w in (why, "retrieval arm stopped: Tavily credit cap reached")
                        if w)
    return Outcome("partial" if snap is None else "done", tuple(due),
                   snap.snapshot_id if snap else None, path, excluded, why)


def summary(o: Outcome) -> str:
    """One line for the job log: counts and ids only, never a payload."""
    return json.dumps({"status": o.status, "games": len(o.games), "excluded": len(o.excluded),
                       "snapshot_id": o.snapshot_id, "why": o.why}, sort_keys=True)
