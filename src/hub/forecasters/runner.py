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
carries this run's market snapshot id and, where this run has a line for the game, its
de-vigged home probability (`market_home_win_prob`, the benchmark `hub.forecasters.score` reads
back); every contestant's probability is clipped to [0.01, 0.99], the betting market's is not.

The model client is an interface; `hub.forecasters.providers` holds the provider clients
behind it, and no real call is made until the maintainer sets `LLM_FORECASTS_ENABLED`. Tests
use a fake. `snapshot_id` is `None` only on a run whose market snapshot failed, which writes
no line-aware or hub record (`hub.forecasters.pipeline`).
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

from hub import atomic
from hub.fetch import tavily
from hub.forecasters import prompt, retrieval
from hub.forecasters.config import UNSET, LlmSpec, StudyConfig, analysis_gaps
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


def paired_spec(config: StudyConfig) -> LlmSpec | None:
    """The bare LLM the retrieval arm pairs with, or None when there is no arm (or the config
    names a model it does not have, which `require_pinned` refuses)."""
    if config.retrieval is None:
        return None
    return next((m for m in config.llms if m.name == config.retrieval.paired_with), None)


def retrieval_contestant(config: StudyConfig) -> str | None:
    """`<model>+retrieval:line_aware`: the paired twin of `<model>:line_aware` (#511)."""
    spec = paired_spec(config)
    return None if spec is None else f"{spec.name}+{retrieval.ARM}:{Condition.LINE_AWARE}"


def contestants(config: StudyConfig) -> list[str]:
    paired = retrieval_contestant(config)
    return ([f"{m.name}:{c}" for m in config.llms for c in Condition]
            + ([paired] if paired else []) + [HUB, HOME, COIN])


def require_pinned(config: StudyConfig) -> None:
    """Refuse a config with any v0 placeholder, a non-positive cap, or no home rate -- and
    while delta or the H1 threshold is unset (#448, adopted 2026-10-22, item 8): a forecast
    made before the analysis that will judge it is fixed is not a pre-registered one. The H2
    implementation pin is `score.require_pinned`'s, since it names the scorer's commit."""
    gaps = [f"{m.name}.{k}" for m in config.llms for k, v in asdict(m).items() if v == UNSET]
    gaps += analysis_gaps(config.analysis, implementation=False)
    gaps += [f"{m.name}.price" for m in config.llms
             if m.usd_per_mtok_in <= 0 or m.usd_per_mtok_out <= 0]
    if config.season_cap_usd <= 0:
        gaps.append("study.season_cap_usd")
    if not 0 < config.home_rate < 1 or config.home_rate_seasons == UNSET:
        gaps.append("study.home_rate")
    if len(config.llms) != 4:
        gaps.append(f"study.llms ({len(config.llms)}, the prereg fixes four)")
    r = config.retrieval
    if r is not None:
        if paired_spec(config) is None:
            gaps.append(f"retrieval.paired_with ({r.paired_with!r} is not a configured llm)")
        if r.season_credit_cap <= 0:
            gaps.append("retrieval.season_credit_cap")
        if r.usd_per_credit < 0:
            gaps.append("retrieval.usd_per_credit")
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
    return _draw(client, spec, prompt.SYSTEM[Condition(packet.condition)], prompt.render(packet),
                 config)


def forecast_retrieved(client: Client, spec: LlmSpec, rp: retrieval.RetrievedPacket,
                       config: StudyConfig) -> LlmForecast:
    """The retrieval arm: the same sampling rule as its bare twin, on the retrieval prompt."""
    return _draw(client, spec, prompt.RETRIEVAL_SYSTEM,
                 prompt.render_retrieved(rp.base, rp.retrieval.items), config)


def _draw(client: Client, spec: LlmSpec, system: str, user: str,
          config: StudyConfig) -> LlmForecast:
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
                 spec: LlmSpec | None = None, pending_retrieval: int = 0) -> float:
    """The worst case: every sample retried to the limit, every call at the output ceiling.
    Priced at the dearest model unless `spec` is given. A pending retrieval forecast is its
    paired model's calls on a prompt longer by the news block's ceiling, plus every search a
    game may make priced at `usd_per_credit`."""
    calls = config.samples * (config.max_parse_retries + 1)
    def per_call(m: LlmSpec, extra: int = 0) -> float:
        return ((config.est_input_tokens + extra) * m.usd_per_mtok_in
                + m.max_output_tokens * m.usd_per_mtok_out) / 1e6
    worst = per_call(spec) if spec else max((per_call(m) for m in config.llms), default=0.0)
    total = pending_llm_forecasts * calls * worst
    paired = paired_spec(config)
    if pending_retrieval and paired is not None and config.retrieval is not None:
        searches = retrieval.SEARCHES_PER_GAME * tavily.CREDITS_PER_SEARCH
        total += pending_retrieval * (calls * per_call(paired, retrieval.PROMPT_EXTRA_TOKENS)
                                      + searches * config.retrieval.usd_per_credit)
    return total


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


def written(out_dir: Path) -> set[tuple[str, str]]:
    """Every (game, contestant) already in the log."""
    return _written(out_dir)[0] if out_dir.exists() else set()


def excluded(out_dir: Path) -> set[str]:
    """Every game already logged as refused. A refused packet is forecast by no one (the
    prereg's exclusions table), so the game is settled: a rerun neither forecasts it nor
    logs it twice."""
    path = out_dir / EXCLUSIONS
    if not path.exists():
        return set()
    return {json.loads(line)["game_id"] for line in path.read_text().splitlines()
            if line.strip()}


def _append(path: Path, record: dict[str, Any]) -> None:
    """Add one record to the log, landing whole (`hub.atomic`).

    Not an `open(path, "a")`: an append killed mid-line leaves a torn last line, which
    `_written` cannot parse -- so every later run would refuse at step 2 until someone edited
    the log by hand -- and the next append would land on the same line as the fragment and
    tear a second record. Rewriting the day's file through `hub.atomic` costs one read of at
    most a few hundred lines a record, and a kill now leaves the previous file exactly as it
    was: the crash loses the one record in flight, which step 2 re-forecasts.
    """
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    atomic.write_text(path, (path.read_text() if path.exists() else "") + line)


@contextlib.contextmanager
def _lock(out_dir: Path) -> Iterator[None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Opened, never written: the lock is the flock on the descriptor, not the file's bytes,
    # so it is not a writer `tests/contracts/test_writers_are_crash_atomic.py` governs.
    fd = os.open(out_dir / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Locked(f"{out_dir / LOCK} is held by another run") from None
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --- the run ------------------------------------------------------------------------------

def run(games: Sequence[GameInputs], *, snapshot_id: str | None, config: StudyConfig,
        client: Client, hub: HubModel, out_dir: Path, run_start: datetime, run_id: str,
        clock: Callable[[], datetime], retriever: retrieval.Retriever | None = None) -> Path:
    """Forecast every unwritten (game, contestant) and append it to this run's file.

    The retrieval arm searches lazily -- after the lock, the idempotency check and the spending
    guard, and only for a game whose paired record is still unwritten -- so a second process, a
    rerun or a refused budget spends no Tavily credit either."""
    path = out_dir / f"{run_start:%Y}" / f"{run_start:%Y-%m-%d}.jsonl"
    with _lock(out_dir):
        seen, spent = _written(out_dir)
        refused = excluded(out_dir)

        def todo(gid: str, name: str) -> bool:
            return (gid, name) not in seen

        pending = sum(1 for g in games if g.refusal is None for m in config.llms for c in Condition
                      if (g.blind if c is Condition.BLIND else g.line_aware) is not None
                      and todo(g.game_id, f"{m.name}:{c}"))
        rname, paired = retrieval_contestant(config), paired_spec(config)
        pending_r = sum(1 for g in games if retriever is not None and rname is not None
                        and g.refusal is None and g.line_aware is not None
                        and todo(g.game_id, rname))
        worst = max_run_cost(config, pending_llm_forecasts=pending, pending_retrieval=pending_r)
        if (pending or pending_r) and config.season_cap_usd - spent < worst:
            raise BudgetExhausted(f"cap ${config.season_cap_usd:.2f} - spent ${spent:.4f} is below "
                                  f"this run's worst case ${worst:.4f}; no model called")

        for g in games:
            # A game logged as refused is settled: forecast by no one, logged once (#447).
            if g.game_id in refused:
                continue
            if g.refusal is not None:
                _append(out_dir / EXCLUSIONS, {"game_id": g.game_id, "run_start": run_start,
                                               "reason": g.refusal, "workflow_run_id": run_id})
                continue

            def write(name: str, *, condition: str | None, model_version: str | None,
                      p: float | None, packet: Packet | None, game: GameInputs = g,
                      prompt_digest: str | None = None, packet_digest: str | None = None,
                      **extra: Any) -> None:
                finished = clock()
                lead = (game.kickoff - finished).total_seconds() / 60
                if prompt_digest is None and condition in tuple(Condition):
                    prompt_digest = prompt.prompt_hash(Condition(condition))
                if packet_digest is None and packet is not None:
                    packet_digest = packet_hash(packet)
                quote = game.line_aware.market if game.line_aware is not None else None
                _append(path, {
                    "game_id": game.game_id, "contestant": name, "condition": condition,
                    "model_version": model_version, "prompt_hash": prompt_digest,
                    "packet_hash": packet_digest,
                    "market_snapshot_id": snapshot_id, "p": p,
                    # The benchmark scoring compares against (#448 item 10): this run's
                    # snapshot, de-vigged, unclipped -- on every record of the game, blind too.
                    "market_home_win_prob": None if quote is None else quote.home_win_prob,
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

            if (retriever is not None and rname is not None and paired is not None
                    and g.line_aware is not None and todo(g.game_id, rname)):
                _retrieval_record(write, g.line_aware, rname, paired, config, client, retriever,
                                  snapshot_id)

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


def _retrieval_record(write: Callable[..., None], base: Packet, name: str, spec: LlmSpec,
                      config: StudyConfig, client: Client, retriever: retrieval.Retriever,
                      snapshot_id: str | None) -> None:
    """One game's paired retrieval forecast. The record names its twin (`paired_with`) and the
    twin's packet hash (`base_packet_hash`), so scoring differences the pair on games both
    have and can check they saw the same packet. A failed search is a lost forecast, logged
    with its reason and the credits it spent; an exhausted credit cap writes nothing."""
    usd = config.retrieval.usd_per_credit if config.retrieval is not None else 0.0
    common: dict[str, Any] = {
        "condition": str(Condition.LINE_AWARE), "model_version": spec.model_version,
        "packet": None, "prompt_digest": prompt.retrieval_prompt_hash(), "arm": retrieval.ARM,
        "paired_with": f"{spec.name}:{Condition.LINE_AWARE}",
        "base_packet_hash": packet_hash(base), "settings": call_settings(spec),
        "training_cutoff": spec.training_cutoff}
    try:
        rp = retriever.retrieve(base, snapshot_id=snapshot_id)
    except retrieval.CreditsExhausted:
        return
    except retrieval.RetrievalFailed as exc:
        write(name, p=None, cost_usd=exc.credits * usd, search_credits=exc.credits,
              retrieval_failure=str(exc), **common)
        return
    f = forecast_retrieved(client, spec, rp, config)
    write(name, p=f.p, packet_digest=retrieval.retrieved_hash(rp), mean=f.mean,
          cost_usd=f.cost_usd + rp.retrieval.credits * usd, lost=f.lost,
          samples=[asdict(s) for s in f.samples], parse_failures=list(f.raw_failures),
          search_credits=rp.retrieval.credits, retrieval=retrieval.plain_block(rp.retrieval),
          **common)
