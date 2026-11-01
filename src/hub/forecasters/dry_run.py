"""A manual live dry run of one study day that writes nowhere (#534).

The provider keys exist only as repository secrets and `forecast.yml` is gated by
`LLM_FORECASTS_ENABLED`, so before the burn-in there was no way to see one real day through the
four providers, Tavily and The Odds API without switching the study on. `run_dry` runs the day
pipeline once (`hub.forecasters.pipeline.run_day`) with everything it writes -- records,
exclusions, the study's Odds API balance file, the Tavily credit file, the last-good copies --
redirected under one `out` directory, which `.github/workflows/forecast-dry-run.yml` uploads as
an artifact. That workflow has `contents: read` and checks out without credentials, so it
cannot push; nothing here touches `state/`, `main` or the `forecasts` branch.

**The ceiling refuses before any paid call.** The worst case of the day -- the runner's own
`max_run_cost` over the pinned prices, every sample retried to the limit -- is compared with the
ceiling before the keys are read, the betting-market snapshot is taken or a model is called.
The same ceiling then replaces the season cap handed to the runner, so its spending guard is
a second fence behind the first.

**What the summary holds** (`SUMMARY`, JSON, and a printed table): per contestant model the
calls, how many parsed, the real output tokens against the pinned cap, latency and cost;
Tavily credits; the Odds API balance and what the pull cost; the ESPN injury feed's parsed
shape as counts and key names only; and any refusal. No header, key, prompt or paid payload
is read into it: the meter sees a `Completion` (text, token counts), keeps the counts, and
drops the text.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.forecasters import days, pipeline, prompt, providers, retrieval, runner, sources
from hub.forecasters.config import LlmSpec, StudyConfig
from hub.forecasters.inputs import SourceStamp
from hub.forecasters.packet import Condition

SUMMARY = "dry-run-summary.json"
LAST_GOOD = "last_good"
CEILING_EXIT = 9
NOT_EMPTY_EXIT = 2


class CeilingExceeded(RuntimeError):
    """The day's worst-case spend is over the ceiling. Nothing was called."""


class OutNotEmpty(RuntimeError):
    """`out` already holds files: a record in it would be skipped as done, and its cost would
    be counted against this run. Nothing was called."""


# --- the meter ----------------------------------------------------------------------------

@dataclass
class _Stats:
    spec: LlmSpec
    cost_usd: float
    calls: int = 0
    parsed: int = 0
    provider_errors: int = 0
    usage_missing: int = 0
    cap_hits: int = 0
    out_tokens: list[int] = field(default_factory=list)
    in_tokens: int = 0
    latencies: list[float] = field(default_factory=list)


class Meter:
    """`runner.Client` around the real one: counts what each call returned and how long it took,
    and passes the `Completion` through untouched. It keeps numbers only -- never the text."""

    def __init__(self, inner: runner.Client, *,
                 timer: Callable[[], float] = time.perf_counter) -> None:
        self._inner, self._timer = inner, timer
        self.stats: dict[str, _Stats] = {}

    @staticmethod
    def label(spec: LlmSpec, system: str) -> str:
        if system == prompt.RETRIEVAL_SYSTEM:
            return f"{spec.name}+{retrieval.ARM}:{Condition.LINE_AWARE}"
        for cond, text in prompt.SYSTEM.items():
            if system == text:
                return f"{spec.name}:{cond}"
        return spec.name

    def complete(self, spec: LlmSpec, system: str, user: str) -> runner.Completion:
        t0 = self._timer()
        c = self._inner.complete(spec, system, user)
        dt = self._timer() - t0
        s = self.stats.setdefault(self.label(spec, system), _Stats(spec, 0.0))
        s.calls += 1
        s.latencies.append(dt)
        s.in_tokens += c.input_tokens
        s.out_tokens.append(c.output_tokens)
        s.cost_usd += (c.input_tokens * spec.usd_per_mtok_in
                       + c.output_tokens * spec.usd_per_mtok_out) / 1e6
        if c.text.startswith(providers.PROVIDER_ERROR_PREFIX):
            s.provider_errors += 1
        else:
            if runner.parse_sample(c.text) is not None:
                s.parsed += 1
            if not c.usage_reported:
                s.usage_missing += 1
        if c.output_tokens >= spec.max_output_tokens:
            s.cap_hits += 1
        return c

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for label, s in sorted(self.stats.items()):
            n = max(s.calls, 1)
            out.append({
                "contestant": label, "provider": s.spec.provider,
                "model_version": s.spec.model_version, "calls": s.calls, "parsed": s.parsed,
                "unparsed": s.calls - s.parsed, "provider_errors": s.provider_errors,
                "usage_missing": s.usage_missing,
                "input_tokens": s.in_tokens,
                "output_tokens_max": max(s.out_tokens, default=0),
                "output_tokens_mean": round(sum(s.out_tokens) / n, 1),
                "output_tokens_cap": s.spec.max_output_tokens, "cap_hits": s.cap_hits,
                "latency_s_mean": round(sum(s.latencies) / n, 3),
                "latency_s_max": round(max(s.latencies, default=0.0), 3),
                "cost_usd": round(s.cost_usd, 6)})
        return out


# --- the estimate -------------------------------------------------------------------------

def estimate(day: date, config: StudyConfig, frame: pl.DataFrame) -> tuple[float, int]:
    """The day's worst-case spend and its number of due games, by the runner's own pricing
    (every model's every forecast pending, every sample retried to the limit)."""
    infos = sources.game_infos(frame)
    due = days.due({g: i.kickoff for g, i in infos.items()}, day)
    n = len(due)
    worst = runner.max_run_cost(
        config, pending_llm_forecasts=n * len(config.llms) * len(tuple(Condition)),
        pending_retrieval=n if runner.paired_spec(config) is not None else 0)
    return worst, n


# --- what the feed and the credit files say -------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def injury_shape(payload: Mapping[str, Any] | None, *, as_of: datetime) -> dict[str, Any]:
    """The ESPN injury feed as counts and key names. Values are never copied, except the
    status vocabulary (a handful of short labels) as counts."""
    if payload is None:
        return {"captured": False}
    teams = [t for t in payload.get("injuries") or [] if isinstance(t, Mapping)]
    items = [i for t in teams for i in t.get("injuries") or [] if isinstance(i, Mapping)]
    athletes = [a for i in items if isinstance(a := i.get("athlete"), Mapping)]
    report = sources.injuries_from_espn(payload, run_start=as_of,
                                        stamp=SourceStamp(sources.INJURY_SOURCE, as_of))
    statuses: dict[str, int] = {}
    for e in report.entries:
        statuses[e.status] = statuses.get(e.status, 0) + 1
    return {"captured": True,
            "top_level_keys": sorted(str(k) for k in payload),
            "team_blocks": len(teams), "items": len(items),
            "item_keys": sorted({str(k) for i in items for k in i}),
            "athlete_keys": sorted({str(k) for a in athletes for k in a}),
            "parsed_entries": len(report.entries),
            "parsed_teams": len({e.team for e in report.entries}),
            "dropped_items": len(items) - len(report.entries),
            "status_counts": dict(sorted(statuses.items())),
            "published_at": None if report.published_at is None
            else report.published_at.isoformat()}


# --- the run ------------------------------------------------------------------------------

def _isolate(deps: pipeline.Deps, out: Path, meter_box: list[Meter],
             captured: list[Mapping[str, Any]], frames: dict[int, pl.DataFrame],
             timer: Callable[[], float]) -> pipeline.Deps:
    """The same deps with the client metered, the injuries payload remembered (as shape, later),
    the schedule fetched once, and the last-good copies under `out`."""
    def frame(season: int) -> pl.DataFrame:
        if season not in frames:
            frames[season] = deps.schedule_frame(season)
        return frames[season]

    def make_client(cfg: StudyConfig) -> runner.Client:
        meter = Meter(deps.make_client(cfg), timer=timer)
        meter_box.append(meter)
        return meter

    def injuries() -> Mapping[str, Any]:
        payload = deps.injuries_payload()
        captured.append(payload)
        return payload

    return dataclasses.replace(deps, schedule_frame=frame, make_client=make_client,
                               injuries_payload=injuries, last_good_root=out / LAST_GOOD)


def _check_out(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise OutNotEmpty(f"{out} is not empty; a dry run needs a fresh directory")


def run_dry(day: date, out: Path, *, ceiling_usd: float, config: StudyConfig,
            deps: pipeline.Deps, refusals: Sequence[tuple[type[Exception], int]] = (),
            echo: Callable[[str], object] = print,
            timer: Callable[[], float] = time.perf_counter) -> int:
    """Run `day` once into `out`, write `out/dry-run-summary.json` and print the table.
    Returns the exit code: 0, a refusal's code, or `CEILING_EXIT`."""
    if ceiling_usd <= 0:
        raise ValueError("the ceiling must be positive")
    _check_out(out)
    meters: list[Meter] = []
    captured: list[Mapping[str, Any]] = []
    frames: dict[int, pl.DataFrame] = {}
    wired = _isolate(deps, out, meters, captured, frames, timer)
    capped = dataclasses.replace(config, season_cap_usd=ceiling_usd)
    summary: dict[str, Any] = {"day": day.isoformat(), "ceiling_usd": ceiling_usd,
                               "estimate_usd": None, "due_games": None, "status": None,
                               "refusal": None}
    code = 0
    try:
        worst, n = estimate(day, capped, wired.schedule_frame(sources.season_of(day)))
        summary.update(estimate_usd=round(worst, 4), due_games=n)
        if worst > ceiling_usd:
            raise CeilingExceeded(
                f"the day's worst case ${worst:.4f} over {n} due game(s) is above the "
                f"${ceiling_usd:.2f} ceiling; no key read and no call made")
        outcome = pipeline.run_day(day, out, config=capped, deps=wired)
        summary.update(status=outcome.status, games=len(outcome.games),
                       excluded=len(outcome.excluded), snapshot_id=outcome.snapshot_id,
                       why=outcome.why)
        code = 6 if outcome.status == "partial" else 0
    except CeilingExceeded as exc:
        summary.update(status="refused", refusal=str(exc))
        code = CEILING_EXIT
    except tuple(kind for kind, _ in refusals) as exc:
        summary.update(status="refused", refusal=str(exc))
        code = next(c for kind, c in refusals if isinstance(exc, kind))

    rows = meters[0].rows() if meters else []
    now = wired.clock()
    odds_state = _read_json(out / pipeline.ODDS_STATE)
    tavily_state = _read_json(out / pipeline.TAVILY_STATE)
    credits = retrieval.credits_used(out / pipeline.TAVILY_STATE)
    usd_per_credit = config.retrieval.usd_per_credit if config.retrieval is not None else 0.0
    llm_cost = round(sum(r["cost_usd"] for r in rows), 6)
    summary.update(
        models=rows,
        tavily={"credits": credits, "searches": int(tavily_state.get("searches", 0)),
                "cost_usd": round(credits * usd_per_credit, 6)},
        odds_api={"remaining": odds_state.get("remaining"),
                  "last_cost_credits": odds_state.get("last_cost"),
                  "declared_cost_credits": odds_state.get("declared_cost")},
        injuries=injury_shape(captured[-1] if captured else None, as_of=now),
        spend_usd={"llm": llm_cost, "tavily": round(credits * usd_per_credit, 6),
                   "total": round(llm_cost + credits * usd_per_credit, 6)})
    out.mkdir(parents=True, exist_ok=True)
    atomic.write_text(out / SUMMARY, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for line in table(summary):
        echo(line)
    return code


def table(s: Mapping[str, Any]) -> list[str]:
    """The summary as the job log shows it: numbers and names only."""
    lines = [f"dry run {s['day']}: {s['status']}"
             f" (worst case ${s['estimate_usd']} of ${s['ceiling_usd']:.2f} ceiling,"
             f" {s['due_games']} due game(s))"]
    if s["refusal"]:
        lines.append(f"refused: {s['refusal']}")
    head = (f"{'contestant':<24}{'calls':>6}{'parsed':>7}{'errs':>5}{'out max/cap':>13}"
            f"{'out mean':>9}{'capped':>7}{'lat mean/max s':>16}{'usd':>10}")
    lines += [head, "-" * len(head)]
    for r in s["models"]:
        lines.append(f"{r['contestant']:<24}{r['calls']:>6}{r['parsed']:>7}"
                     f"{r['provider_errors']:>5}"
                     f"{str(r['output_tokens_max']) + '/' + str(r['output_tokens_cap']):>13}"
                     f"{r['output_tokens_mean']:>9}{r['cap_hits']:>7}"
                     f"{str(r['latency_s_mean']) + '/' + str(r['latency_s_max']):>16}"
                     f"{r['cost_usd']:>10.4f}")
    t, o, i = s["tavily"], s["odds_api"], s["injuries"]
    lines.append(f"tavily credits: {t['credits']} over {t['searches']} search(es)")
    lines.append(f"odds api: {o['remaining']} credits remaining, last pull cost "
                 f"{o['last_cost_credits']} (declared {o['declared_cost_credits']})")
    if i.get("captured"):
        lines.append(f"espn injuries: {i['team_blocks']} team blocks, {i['items']} items -> "
                     f"{i['parsed_entries']} parsed over {i['parsed_teams']} teams "
                     f"({i['dropped_items']} dropped); item keys {i['item_keys']}")
    else:
        lines.append("espn injuries: not fetched")
    lines.append(f"spend: ${s['spend_usd']['total']:.4f} "
                 f"(models ${s['spend_usd']['llm']:.4f}, tavily ${s['spend_usd']['tavily']:.4f})")
    return lines


