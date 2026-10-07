"""One forecast run: every due game, every contestant, written once.

`docs/prereg/llm-forecasters.md`, *Forecast procedure*. The order is the safety argument:

1. take the lock -- a second process on this checkout stops here, before any model call;
2. work out what is still unwritten -- a (game, contestant) already in the log is skipped, so a
   retry after a crash fills the gaps and never pays twice;
3. price the worst case of what is left and compare it with the season cap minus everything
   already spent -- below it, stop, before any model call (not every provider has a hard cap);
4. forecast, appending each record the moment it exists, so a crash loses at most one.

Eleven contestants: each LLM blind and line-aware, the hub's own model on the same snapshot
(logging the commit that produced it), the home constant, and the coin flip. Every record
carries this run's market snapshot id, and every probability is clipped to [0.01, 0.99].

The model client is an interface; the provider clients are not written yet and no real call
is made until the maintainer says so. Tests use a fake.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from hub.forecasters import prompt
from hub.forecasters.config import UNSET, LlmSpec, StudyConfig
from hub.forecasters.packet import Condition, MarketBlock, Packet, packet_hash

LOCK = ".run.lock"
EXCLUSIONS = "exclusions.jsonl"
MAX_RATIONALE_WORDS = 60
HUB, HOME, COIN = "hub", "home_constant", "coin_flip"


class Locked(RuntimeError):
    """Another run holds the lock. Nothing was called."""


class BudgetExhausted(RuntimeError):
    """The cap minus what has been spent is below this run's worst case. Nothing was called."""


class NotPinned(RuntimeError):
    """The config still has v0 placeholders; no model may be called under it."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class Client(Protocol):
    def complete(self, spec: LlmSpec, system: str, user: str) -> Completion: ...


class HubModel(Protocol):
    """The hub's own forecast for a game, from this run's market snapshot."""
    version: str                    # the commit that produced it

    def predict(self, game_id: str, market: MarketBlock) -> float | None: ...


@dataclass(frozen=True)
class GameInputs:
    """A due game: its packets (None where unavailable) or the reason none could be built."""
    game_id: str
    kickoff: datetime
    blind: Packet | None
    line_aware: Packet | None
    refusal: str | None = None


# --- settings and contestants -------------------------------------------------------------

def call_settings(spec: LlmSpec) -> dict[str, Any]:
    """What a client sends besides the prompt: temperature only where the model takes one."""
    out: dict[str, Any] = {}
    if spec.temperature is not None:
        out["temperature"] = spec.temperature
    if spec.reasoning_effort is not None:
        out["reasoning_effort"] = spec.reasoning_effort
    out["max_output_tokens"] = spec.max_output_tokens
    return out


def contestants(config: StudyConfig) -> list[str]:
    return ([f"{m.name}:{c}" for m in config.llms for c in Condition] + [HUB, HOME, COIN])


def require_pinned(config: StudyConfig) -> None:
    """Refuse a config with any v0 placeholder, a non-positive cap, or no home rate."""
    gaps = [f"{m.name}.{k}" for m in config.llms for k, v in asdict(m).items() if v == UNSET]
    gaps += [f"{m.name}.price" for m in config.llms
             if m.usd_per_mtok_in <= 0 or m.usd_per_mtok_out <= 0]
    if config.season_cap_usd <= 0:
        gaps.append("study.season_cap_usd")
    if not 0 < config.home_rate < 1 or config.home_rate_seasons == UNSET:
        gaps.append("study.home_rate")
    if len(config.llms) != 4:
        gaps.append(f"study.llms ({len(config.llms)}, the prereg fixes four)")
    if gaps:
        raise NotPinned("conf/forecasters.toml is not pinned: " + ", ".join(gaps))


# --- one model, one packet ----------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    p: float
    rationale: str
    truncated: bool = False
    attempts: int = 1


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_sample(text: str) -> Sample | None:
    """One answer, or None if it is not the required JSON: `p_home_win` a number in [0, 1] (a
    bool is not a number here) and `rationale` a string. A rationale over 60 words is kept to
    60 and flagged; it is logged, never scored."""
    m = _FENCE.match(text)
    try:
        obj = json.loads(m.group(1) if m else text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    p, why = obj.get("p_home_win"), obj.get("rationale")
    if isinstance(p, bool) or not isinstance(p, int | float) or not isinstance(why, str):
        return None
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        return None
    words = why.split()
    if len(words) > MAX_RATIONALE_WORDS:
        return Sample(float(p), " ".join(words[:MAX_RATIONALE_WORDS]), truncated=True)
    return Sample(float(p), why)


def clip(p: float, config: StudyConfig) -> float:
    return min(max(p, config.clip_lo), config.clip_hi)


def _cost(spec: LlmSpec, c: Completion) -> float:
    return (c.input_tokens * spec.usd_per_mtok_in + c.output_tokens * spec.usd_per_mtok_out) / 1e6


@dataclass(frozen=True)
class LlmForecast:
    samples: tuple[Sample, ...]
    lost: int
    mean: float | None
    p: float | None
    cost_usd: float
    raw_failures: tuple[str, ...] = field(default=())

    @property
    def valid(self) -> int:
        return len(self.samples)


def forecast_llm(client: Client, spec: LlmSpec, packet: Packet, config: StudyConfig) -> LlmForecast:
    """`config.samples` samples, each retried up to `max_parse_retries` times on a parse failure;
    the mean of the valid ones, or no forecast with fewer than `min_valid`."""
    system, user = prompt.SYSTEM[Condition(packet.condition)], prompt.render(packet)
    kept: list[Sample] = []
    failures: list[str] = []
    cost = 0.0
    for _ in range(config.samples):
        for attempt in range(1, config.max_parse_retries + 2):
            c = client.complete(spec, system, user)
            cost += _cost(spec, c)
            s = parse_sample(c.text)
            if s is not None:
                kept.append(Sample(s.p, s.rationale, s.truncated, attempt))
                break
            failures.append(c.text[:500])
    lost = config.samples - len(kept)
    if len(kept) < config.min_valid:
        return LlmForecast(tuple(kept), lost, None, None, cost, tuple(failures))
    mean = statistics.fmean(s.p for s in kept)
    return LlmForecast(tuple(kept), lost, mean, clip(mean, config), cost, tuple(failures))


def max_run_cost(config: StudyConfig, *, pending_llm_forecasts: int,
                 spec: LlmSpec | None = None) -> float:
    """The worst case: every sample retried to the limit, every call at the output ceiling.
    Priced at the dearest model unless `spec` is given."""
    calls = config.samples * (config.max_parse_retries + 1)
    def per_call(m: LlmSpec) -> float:
        return (config.est_input_tokens * m.usd_per_mtok_in
                + m.max_output_tokens * m.usd_per_mtok_out) / 1e6
    worst = per_call(spec) if spec else max((per_call(m) for m in config.llms), default=0.0)
    return pending_llm_forecasts * calls * worst


# --- the log ------------------------------------------------------------------------------

def _written(out_dir: Path) -> tuple[set[tuple[str, str]], float]:
    """Every (game, contestant) already in the log, and the dollars already spent."""
    seen: set[tuple[str, str]] = set()
    spent = 0.0
    for f in out_dir.rglob("*.jsonl"):
        if f.name == EXCLUSIONS:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seen.add((r["game_id"], r["contestant"]))
            spent += float(r.get("cost_usd") or 0.0)
    return seen, spent


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def _lock(out_dir: Path) -> Iterator[None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / LOCK, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Locked(f"{out_dir / LOCK} is held by another run") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- the run ------------------------------------------------------------------------------

def run(games: Sequence[GameInputs], *, snapshot_id: str, config: StudyConfig, client: Client,
        hub: HubModel, out_dir: Path, run_start: datetime, run_id: str,
        clock: Callable[[], datetime]) -> Path:
    """Forecast every unwritten (game, contestant) and append it to this run's file."""
    path = out_dir / f"{run_start:%Y}" / f"{run_start:%Y-%m-%d}.jsonl"
    with _lock(out_dir):
        seen, spent = _written(out_dir)

        def todo(gid: str, name: str) -> bool:
            return (gid, name) not in seen

        pending = sum(1 for g in games if g.refusal is None for m in config.llms for c in Condition
                      if (g.blind if c is Condition.BLIND else g.line_aware) is not None
                      and todo(g.game_id, f"{m.name}:{c}"))
        worst = max_run_cost(config, pending_llm_forecasts=pending)
        if pending and config.season_cap_usd - spent < worst:
            raise BudgetExhausted(f"cap ${config.season_cap_usd:.2f} - spent ${spent:.4f} is below "
                                  f"this run's worst case ${worst:.4f}; no model called")

        for g in games:
            if g.refusal is not None:
                if todo(g.game_id, "*refused*"):
                    _append(out_dir / EXCLUSIONS, {"game_id": g.game_id, "run_start": run_start,
                                                   "reason": g.refusal, "workflow_run_id": run_id})
                continue

            def write(name: str, *, condition: str | None, model_version: str | None,
                      p: float | None, packet: Packet | None, game: GameInputs = g,
                      **extra: Any) -> None:
                finished = clock()
                lead = (game.kickoff - finished).total_seconds() / 60
                _append(path, {
                    "game_id": game.game_id, "contestant": name, "condition": condition,
                    "model_version": model_version,
                    "prompt_hash": prompt.prompt_hash(Condition(condition))
                    if condition in tuple(Condition) else None,
                    "packet_hash": packet_hash(packet) if packet is not None else None,
                    "market_snapshot_id": snapshot_id, "p": p,
                    "run_start": run_start, "finished_at": finished, "workflow_run_id": run_id,
                    "excluded": "finished_within_60min_of_kickoff"
                    if lead < config.min_lead_minutes else None,
                    "cost_usd": 0.0, **extra})

            for m in config.llms:
                for c in Condition:
                    packet = g.blind if c is Condition.BLIND else g.line_aware
                    name = f"{m.name}:{c}"
                    if packet is None or not todo(g.game_id, name):
                        continue
                    f = forecast_llm(client, m, packet, config)
                    write(name, condition=str(c), model_version=m.model_version, p=f.p,
                          packet=packet, mean=f.mean, cost_usd=f.cost_usd, lost=f.lost,
                          samples=[asdict(s) for s in f.samples],
                          parse_failures=list(f.raw_failures),
                          settings=call_settings(m), training_cutoff=m.training_cutoff)

            market = g.line_aware.market if g.line_aware is not None else None
            if market is not None and todo(g.game_id, HUB):
                raw = hub.predict(g.game_id, market)
                write(HUB, condition=None, model_version=hub.version,
                      p=None if raw is None else clip(raw, config), packet=g.line_aware, mean=raw)
            if todo(g.game_id, HOME):
                write(HOME, condition=None, model_version=f"home_rate:{config.home_rate_seasons}",
                      p=clip(config.home_rate, config), packet=None, mean=config.home_rate)
            if todo(g.game_id, COIN):
                write(COIN, condition=None, model_version=None, p=clip(0.5, config), packet=None,
                      mean=0.5)
    return path
