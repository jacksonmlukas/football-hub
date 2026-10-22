"""The paired retrieval arm (#511): `hub.fetch.tavily` and `hub.forecasters.retrieval`.

No network: every search goes through a stubbed transport that records what it was handed,
and the default transport is refused under pytest before anything leaves the process. What
the ticket asks of it, each held below:

* refuses without `TAVILY_API_KEY`, in one sentence, before anything is sent -- and the key
  travels only in the header, never in a URL, a body or an error;
* an item published at or after run start, or with no usable date, never reaches a packet,
  and every drop is logged and counted;
* the search cap: no game gets more than `SEARCHES_PER_GAME` searches, and the season's credit
  cap stops the arm, not the run;
* the record names its twin (`paired_with`) and the twin's packet hash (`base_packet_hash`);
* the dry run on fakes includes the arm end to end, and a known pre-kickoff injury story is
  in the retrieval prompt and in no bare one (the positive control #511 names).
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pytest
from forecasters_fixtures import RUN_START as PKT_RUN_START
from forecasters_fixtures import SNAPSHOT, _build
from forecasters_slate import (
    DAY,
    DUE,
    FULL,
    NAMES,
    RUN_START,
    injuries_payload,
    odds_payload,
    schedule_frame,
)

from hub.fetch import cached, odds, tavily
from hub.forecasters import cli, pipeline, prompt, qb, retrieval, runner, sources
from hub.forecasters import config as study_config
from hub.forecasters.config import Analysis, LlmSpec, RetrievalSpec, StudyConfig
from hub.forecasters.inputs import QbSource, SourceStamp
from hub.forecasters.packet import Condition, PacketRefused, packet_hash

KEY = "tvly-test-0123456789abcdef"
ENV = {"TAVILY_API_KEY": KEY}
STORY = "ruled out for Sunday with a hamstring injury"


def _rfc(t: datetime) -> str:
    return format_datetime(t.astimezone(UTC), usegmt=True)


def _hits(query: str, run_start: datetime) -> list[dict[str, Any]]:
    """One search's reply: a story from three hours before the run (kept), one a minute after
    (late), one dated exactly at run start (late: the rule is strict), and one with no date."""
    slug = query.replace(" ", "-").lower()
    return [
        {"url": f"https://news.example/{slug}/kept", "title": f"{query}: update",
         "content": f"{query.split(' ')[0]} starter {STORY}.",
         "published_date": _rfc(run_start - timedelta(hours=3))},
        {"url": f"https://news.example/{slug}/late", "title": "after the run started",
         "content": "LEAK", "published_date": _rfc(run_start + timedelta(minutes=1))},
        {"url": f"https://news.example/{slug}/at", "title": "at run start", "content": "LEAK",
         "published_date": _rfc(run_start)},
        {"url": f"https://news.example/{slug}/undated", "title": "no date", "content": "LEAK"},
    ]


class Stub:
    """A Tavily transport that never leaves the process."""

    def __init__(self, run_start: datetime = RUN_START, *, fail_on: int | None = None,
                 credits: int | None = 1) -> None:
        self.run_start, self.fail_on, self.credits = run_start, fail_on, credits
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, headers: Mapping[str, str],
                 body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((url, dict(headers), dict(body)))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise ConnectionError(f"boom {headers.get('authorization')}")
        reply: dict[str, Any] = {"results": _hits(str(body["query"]), self.run_start)}
        if self.credits is not None:
            reply["usage"] = {"credits": self.credits}
        return reply


def _client(stub: Stub) -> tavily.Client:
    return tavily.Client(env=ENV, transport=stub)


# --- the adapter ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", [{}, {"TAVILY_API_KEY": ""}, {"TAVILY_API_KEY": "  "}])
def test_no_key_refuses_in_one_sentence_before_anything_is_sent(env):
    stub = Stub()
    with pytest.raises(tavily.MissingKey) as exc:
        tavily.Client(env=env, transport=stub)
    msg = str(exc.value)
    assert "TAVILY_API_KEY" in msg and msg.count(". ") == 0 and msg.endswith(".")
    assert stub.calls == []


def test_the_default_transport_is_refused_under_pytest_before_anything_leaves():
    with pytest.raises(cached.LiveCallRefused):
        tavily._post(tavily.URL, {}, {})
    c = tavily.Client(env=ENV)                                   # no transport: the real one
    with pytest.raises(tavily.TavilyError) as exc:               # and search() wraps it
        c.search("Kansas City Chiefs injury report", days=7, max_results=5)
    assert KEY not in str(exc.value)


def test_the_key_travels_in_the_header_only():
    stub = Stub()
    _client(stub).search("Buffalo Bills inactives", days=7, max_results=5)
    url, headers, body = stub.calls[0]
    assert url == "https://api.tavily.com/search" and KEY not in url
    assert headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in json.dumps(body)
    assert body["topic"] == "news" and body["search_depth"] == "basic"
    assert (body["days"], body["max_results"]) == (7, 5)


def test_a_failed_call_never_quotes_the_transport():
    stub = Stub(fail_on=1)                                       # its message carries the key
    with pytest.raises(tavily.TavilyError) as exc:
        _client(stub).search("q", days=7, max_results=5)
    assert KEY not in str(exc.value) and "ConnectionError" in str(exc.value)


def test_an_unreadable_reply_is_a_tavily_error():
    c = tavily.Client(env=ENV, transport=lambda u, h, b: {"no": "results"})
    with pytest.raises(tavily.TavilyError, match="unreadable"):
        c.search("q", days=7, max_results=5)


def test_credits_are_the_replys_own_count_or_the_declared_one():
    assert _client(Stub(credits=2)).search("q", days=7, max_results=5).credits == 2
    assert _client(Stub(credits=None)).search("q", days=7, max_results=5).credits == \
        tavily.CREDITS_PER_SEARCH


@pytest.mark.parametrize(("raw", "want"), [
    ("Sun, 18 Oct 2026 10:31:00 GMT", datetime(2026, 10, 18, 10, 31, tzinfo=UTC)),
    ("2026-10-18T10:31:00Z", datetime(2026, 10, 18, 10, 31, tzinfo=UTC)),
    ("2026-10-18T06:31:00-04:00", datetime(2026, 10, 18, 10, 31, tzinfo=UTC)),
    ("2026-10-18T10:31:00", None),                              # naive: could be any hour
    ("yesterday", None), ("", None), (None, None),
])
def test_a_publication_time_is_parsed_or_refused(raw, want):
    assert tavily.published_at(raw) == want


# --- the point-in-time rule ------------------------------------------------------------------

def _search(query: str, run_start: datetime, extra: list[dict] | None = None) -> tavily.Search:
    hits = [tavily.Hit(h["url"], h["title"], h["content"], h.get("published_date"))
            for h in _hits(query, run_start) + (extra or [])]
    return tavily.Search(query, tuple(hits), 1)


def test_only_items_published_strictly_before_run_start_are_admitted():
    old = {"url": "https://news.example/old", "title": "old", "content": "LEAK",
           "published_date": _rfc(RUN_START - timedelta(days=8))}
    items, dropped = retrieval.admit([_search("q", RUN_START, [old])], RUN_START)
    assert [i.url for i in items] == ["https://news.example/q/kept"]
    assert all(i.published_at < RUN_START for i in items)
    assert sorted(d.reason for d in dropped) == ["at_or_after_run_start",
                                                  "at_or_after_run_start",
                                                  "older_than_lookback", "undated"]
    assert all("LEAK" not in i.snippet for i in items)
    assert {d.url for d in dropped} >= {"https://news.example/q/late", "https://news.example/q/at"}


def test_a_story_seen_twice_is_kept_once_and_the_budget_cuts_newest_first(monkeypatch):
    monkeypatch.setattr(retrieval, "MAX_ITEMS", 2)
    monkeypatch.setattr(retrieval, "SNIPPET_CHARS", 10)
    searches = [_search(q, RUN_START) for q in ("a", "b", "c")]
    searches.append(_search("a", RUN_START))                    # the same URLs again
    items, dropped = retrieval.admit(searches, RUN_START)
    assert len(items) == 2 and all(len(i.snippet) <= 10 for i in items)
    reasons = [d.reason for d in dropped]
    assert reasons.count("duplicate_url") == 1 and reasons.count("over_budget") == 1


def test_the_validator_refuses_an_item_at_run_start_planted_in_a_finished_packet():
    base = _build(Condition.LINE_AWARE)
    rp = retrieval.Retriever(_client(Stub(PKT_RUN_START)), full_names=FULL,
                             state_path=Path("/nonexistent/never-written.json"),
                             season_credit_cap=0, clock=lambda: PKT_RUN_START)
    with pytest.raises(retrieval.CreditsExhausted):            # cap 0: nothing is searched
        rp.retrieve(base, snapshot_id=SNAPSHOT)
    good = retrieval.RetrievedPacket(base, retrieval.RetrievalBlock(
        published_before=PKT_RUN_START, basis=retrieval.TIME_BASIS, queries=("q",), searches=1,
        credits=1, items=(retrieval.Item("q", "u", "t", PKT_RUN_START - timedelta(hours=1),
                                         "s"),),
        dropped=(), stamp=SourceStamp(retrieval.SOURCE, PKT_RUN_START)))
    retrieval.validate_retrieved(good, snapshot_id=SNAPSHOT)
    leak = replace(good, retrieval=replace(good.retrieval, items=(
        replace(good.retrieval.items[0], published_at=PKT_RUN_START),)))
    with pytest.raises(PacketRefused, match="retrieval_published"):
        retrieval.validate_retrieved(leak, snapshot_id=SNAPSHOT)
    blind = replace(good, base=_build(Condition.BLIND))
    with pytest.raises(PacketRefused, match="retrieval_condition"):
        retrieval.validate_retrieved(blind, snapshot_id=SNAPSHOT)
    shifted = replace(good, retrieval=replace(good.retrieval,
                                              published_before=PKT_RUN_START + timedelta(1)))
    with pytest.raises(PacketRefused, match="retrieval_cutoff"):
        retrieval.validate_retrieved(shifted, snapshot_id=SNAPSHOT)
    over = replace(good, retrieval=replace(good.retrieval, searches=5))
    with pytest.raises(PacketRefused, match="retrieval_cap"):
        retrieval.validate_retrieved(over, snapshot_id=SNAPSHOT)
    with pytest.raises(PacketRefused, match="market_snapshot"):       # the bare packet's checks
        retrieval.validate_retrieved(good, snapshot_id="another-run")


def test_a_retrieved_packet_for_another_runs_snapshot_fails_the_game(tmp_path):
    r = retrieval.Retriever(_client(Stub(PKT_RUN_START)), full_names=FULL,
                            state_path=tmp_path / "t.json", season_credit_cap=100,
                            clock=lambda: PKT_RUN_START)
    with pytest.raises(retrieval.RetrievalFailed, match="market_snapshot") as exc:
        r.retrieve(_build(Condition.LINE_AWARE), snapshot_id="another-run")
    assert exc.value.credits == 4                                # spent, and said so


def test_every_snippet_is_in_the_hash_and_the_bare_packet_is_unchanged(tmp_path):
    base = _build(Condition.LINE_AWARE)
    r = retrieval.Retriever(_client(Stub(PKT_RUN_START)), full_names=FULL,
                            state_path=tmp_path / "t.json", season_credit_cap=100,
                            clock=lambda: PKT_RUN_START)
    rp = r.retrieve(base, snapshot_id=SNAPSHOT)
    assert rp.base is base and packet_hash(rp.base) == packet_hash(_build(Condition.LINE_AWARE))
    item = rp.retrieval.items[0]
    moved = replace(rp, retrieval=replace(rp.retrieval, items=(
        replace(item, snippet=item.snippet + "!"), *rp.retrieval.items[1:])))
    assert retrieval.retrieved_hash(moved) != retrieval.retrieved_hash(rp)
    assert retrieval.retrieved_hash(rp) != packet_hash(base)


# --- the search cap and the credits ----------------------------------------------------------

def _retriever(tmp_path: Path, stub: Stub, cap: int = 1000) -> retrieval.Retriever:
    return retrieval.Retriever(_client(stub), full_names=FULL, state_path=tmp_path / "t.json",
                               season_credit_cap=cap, clock=lambda: PKT_RUN_START)


def test_a_game_gets_the_fixed_queries_and_no_more_than_the_cap(tmp_path):
    stub = Stub(PKT_RUN_START)
    rp = _retriever(tmp_path, stub).retrieve(_build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)
    queries = [b["query"] for _, _, b in stub.calls]
    assert queries == ["Kansas City Chiefs injury report", "Kansas City Chiefs inactives",
                       "Buffalo Bills injury report", "Buffalo Bills inactives"]
    assert len(stub.calls) == retrieval.SEARCHES_PER_GAME == 4
    assert rp.retrieval.searches == 4 and rp.retrieval.credits == 4
    assert len(rp.retrieval.items) == 4 and len(rp.retrieval.dropped) == 12


# The cap's refusal is `test_forecasters_retrieval_cap.py`, which imports nothing from `tests/`
# so the guard harness can run it against a mutant.


def test_each_search_is_counted_in_the_studys_own_state_file(tmp_path):
    _retriever(tmp_path, Stub(PKT_RUN_START, credits=1)).retrieve(
        _build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)
    state = json.loads((tmp_path / "t.json").read_text())
    assert state["credits_used"] == 4 and state["searches"] == 4
    assert retrieval.credits_used(tmp_path / "t.json") == 4
    assert retrieval.credits_used(tmp_path / "missing.json") == 0


def test_the_credit_cap_stops_the_arm_before_a_search(tmp_path):
    stub = Stub(PKT_RUN_START)
    r = _retriever(tmp_path, stub, cap=7)
    r.retrieve(_build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)     # 4 of 7
    with pytest.raises(retrieval.CreditsExhausted):
        r.retrieve(_build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)  # 4 more would be 8
    assert len(stub.calls) == 4 and r.exhausted


def test_a_failed_search_fails_the_game_and_counts_what_it_spent(tmp_path):
    stub = Stub(PKT_RUN_START, fail_on=3)
    with pytest.raises(retrieval.RetrievalFailed) as exc:
        _retriever(tmp_path, stub).retrieve(_build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)
    assert exc.value.credits == 2 and KEY not in str(exc.value)
    assert retrieval.credits_used(tmp_path / "t.json") == 2


# --- the prompt ------------------------------------------------------------------------------

def test_the_retrieval_prompt_is_pinned_and_differs_from_the_bare_one_only_where_it_must():
    assert prompt.retrieval_prompt_hash() == prompt.PINNED_RETRIEVAL
    assert prompt.PINNED == {c: prompt.prompt_hash(c) for c in Condition}   # bare unmoved
    assert prompt.retrieval_prompt_hash() not in prompt.PINNED.values()
    assert "no web access and no information beyond" not in prompt.RETRIEVAL_SYSTEM
    assert prompt.SYSTEM[Condition.LINE_AWARE].endswith(
        prompt.RETRIEVAL_SYSTEM.split("Give the probability")[1])


def test_the_retrieval_message_is_the_bare_message_plus_the_news(tmp_path):
    base = _build(Condition.LINE_AWARE)
    rp = _retriever(tmp_path, Stub(PKT_RUN_START)).retrieve(base, snapshot_id=SNAPSHOT)
    msg = prompt.render_retrieved(rp.base, rp.retrieval.items)
    assert msg.startswith(prompt.render(base))
    assert STORY in msg and "LEAK" not in msg
    assert "dropped" not in msg and "credits" not in msg        # bookkeeping is not shown


# --- the config ------------------------------------------------------------------------------

PROVIDERS = ("anthropic", "openai", "google", "google")


def _config(**kw: Any) -> StudyConfig:
    llms = tuple(LlmSpec(name=f"llm{i + 1}", provider=p, model_version=f"{p}-model-{i}",
                         training_cutoff="2025-06", temperature=None, reasoning_effort=None,
                         max_output_tokens=2000, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
                 for i, p in enumerate(PROVIDERS))
    spec = replace(RetrievalSpec(paired_with="llm1", season_credit_cap=1500,
                                 usd_per_credit=0.01), **kw)
    return StudyConfig(llms=llms, season_cap_usd=150.0, home_rate=0.547,
                       home_rate_seasons="2016-2025", est_input_tokens=3000, retrieval=spec,
                       analysis=Analysis(0.02, 0.25, "hub.forecasters.score@abcdef0"))


def test_the_paired_contestant_is_named_after_its_twin():
    cfg = _config()
    assert runner.retrieval_contestant(cfg) == "llm1+retrieval:line_aware"
    assert len(runner.contestants(cfg)) == 12
    assert runner.retrieval_contestant(replace(cfg, retrieval=None)) is None
    runner.require_pinned(cfg)


@pytest.mark.parametrize(("kw", "gap"), [
    ({"paired_with": "llm9"}, "retrieval.paired_with"),
    ({"season_credit_cap": 0}, "retrieval.season_credit_cap"),
    ({"usd_per_credit": -1.0}, "retrieval.usd_per_credit"),
])
def test_an_unpinned_retrieval_arm_is_refused(kw, gap):
    with pytest.raises(runner.NotPinned, match=gap):
        runner.require_pinned(_config(**kw))


def test_the_committed_config_carries_the_arm_and_refuses_it_until_pinned():
    cfg = study_config.load()
    assert cfg.retrieval is not None and cfg.retrieval.paired_with == study_config.UNSET
    with pytest.raises(runner.NotPinned) as exc:
        runner.require_pinned(cfg)
    assert "retrieval.paired_with" in str(exc.value)
    assert "retrieval.usd_per_credit" in str(exc.value)


def test_the_worst_case_prices_the_arm_and_its_searches():
    cfg = _config()
    bare = runner.max_run_cost(cfg, pending_llm_forecasts=8)
    both = runner.max_run_cost(cfg, pending_llm_forecasts=8, pending_retrieval=1)
    calls = cfg.samples * (cfg.max_parse_retries + 1)
    spec = cfg.llms[0]
    one = (calls * ((cfg.est_input_tokens + retrieval.PROMPT_EXTRA_TOKENS) * spec.usd_per_mtok_in
                    + spec.max_output_tokens * spec.usd_per_mtok_out) / 1e6
           + retrieval.SEARCHES_PER_GAME * 0.01)
    assert both - bare == pytest.approx(one)


# --- the day, dry-run end to end on fakes ----------------------------------------------------

class Recorder:
    """A fake model client that keeps every prompt it was handed."""

    def __init__(self) -> None:
        self.users: list[tuple[str, str]] = []

    def complete(self, spec: LlmSpec, system: str, user: str) -> runner.Completion:
        self.users.append((system, user))
        return runner.Completion('{"p_home_win": 0.58, "rationale": "fine"}', 900, 80)


class Odds:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, params: Mapping[str, Any], key: str) -> tuple[Any, Mapping[str, str]]:
        self.calls += 1
        return odds_payload(), {"x-requests-remaining": "450"}


def _qb(run_start: datetime, stamp: SourceStamp) -> QbSource:
    return QbSource(stamp=stamp, updated_at=run_start - timedelta(hours=20),
                    updated_at_basis=qb.QB_TIME_BASIS,
                    starters={t: f"{t} QB" for t in NAMES.values()})


def _deps(tmp_path: Path, *, client: Recorder, stub: Stub | None = None,
          odds_get: Odds | None = None, make_searcher: Any = None) -> pipeline.Deps:
    tick = itertools.count()
    s = stub or Stub()
    return pipeline.Deps(
        schedule_frame=lambda season: schedule_frame(), injuries_payload=injuries_payload,
        qb_source=_qb, make_client=lambda cfg: client,
        make_hub=lambda season, week: sources.RatingsHub(season, week, version="testsha"),
        team_names=lambda games: NAMES, odds_get=odds_get or Odds(), odds_key="test-odds-key",
        clock=lambda: RUN_START + timedelta(seconds=10 * next(tick)), run_id="999",
        last_good_root=tmp_path / "last_good",
        make_searcher=make_searcher or (lambda cfg: tavily.Client(env=ENV, transport=s)))


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(out.rglob("*.jsonl")) if f.name != runner.EXCLUSIONS
            for line in f.read_text().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _hub_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(odds, "STATE", tmp_path / "hub_state" / "odds.json")


def test_the_dry_run_writes_the_paired_arm_for_every_game(tmp_path):
    out, client, stub = tmp_path / "forecasts", Recorder(), Stub()
    o = pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path, client=client,
                                                                stub=stub))
    assert o.status == "done" and o.why is None
    recs = _records(out)
    by = {(r["game_id"], r["contestant"]): r for r in recs}
    assert len(recs) == len(DUE) * 12
    assert len(stub.calls) == len(DUE) * retrieval.SEARCHES_PER_GAME
    assert len(client.users) == len(DUE) * 9 * 5
    for gid in DUE:
        r, twin = by[(gid, "llm1+retrieval:line_aware")], by[(gid, "llm1:line_aware")]
        assert r["paired_with"] == "llm1:line_aware" and r["arm"] == "retrieval"
        assert r["base_packet_hash"] == twin["packet_hash"] != r["packet_hash"]
        assert r["model_version"] == twin["model_version"]
        assert r["settings"] == twin["settings"] and r["condition"] == "line_aware"
        assert r["prompt_hash"] == prompt.PINNED_RETRIEVAL != twin["prompt_hash"]
        assert r["market_snapshot_id"] == twin["market_snapshot_id"] == o.snapshot_id
        assert r["search_credits"] == 4 and r["p"] is not None
        assert r["cost_usd"] == pytest.approx(twin["cost_usd"] + 4 * 0.01)
        block = r["retrieval"]
        assert len(block["items"]) == 4 and len(block["dropped"]) == 12
        assert block["published_before"].startswith("2026-10-18 13:31")
        assert all(i["published_at"] < block["published_before"] for i in block["items"])
        assert "undated" in {d["reason"] for d in block["dropped"]}
    assert retrieval.credits_used(out / pipeline.TAVILY_STATE) == len(DUE) * 4


def test_a_known_pre_kickoff_injury_story_reaches_the_retrieval_arm_and_no_bare_arm(tmp_path):
    client = Recorder()
    pipeline.run_day(DAY, tmp_path / "f", config=_config(), deps=_deps(tmp_path, client=client))
    with_news = [u for s, u in client.users if s == prompt.RETRIEVAL_SYSTEM]
    bare = [u for s, u in client.users if s != prompt.RETRIEVAL_SYSTEM]
    assert len(with_news) == len(DUE) * 5
    assert all(STORY in u and "Kansas City Chiefs" in json.dumps(with_news) for u in with_news)
    assert not any(STORY in u for u in bare) and not any("LEAK" in u for u in with_news)


def test_rerunning_searches_nothing(tmp_path):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path, client=Recorder()))
    stub, again = Stub(), Odds()
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, client=Recorder(), stub=stub, odds_get=again))
    assert o.status == "nothing_left" and stub.calls == [] and again.calls == 0


def test_a_missing_tavily_key_refuses_before_the_snapshot(tmp_path):
    o = Odds()
    with pytest.raises(tavily.MissingKey):
        pipeline.run_day(DAY, tmp_path / "f", config=_config(),
                         deps=_deps(tmp_path, client=Recorder(), odds_get=o,
                                    make_searcher=lambda cfg: tavily.Client(env={})))
    assert o.calls == 0


def test_the_cli_answers_a_missing_tavily_key_with_exit_5(tmp_path):
    lines: list[str] = []
    rc = cli.main(["run", "--out", str(tmp_path / "f"), "--day", DAY.isoformat()],
                  echo=lines.append, config=_config(),
                  deps=_deps(tmp_path, client=Recorder(),
                             make_searcher=lambda cfg: tavily.Client(env={})))
    assert rc == 5 and len(lines) == 1 and lines[0].startswith("refused: TAVILY_API_KEY")


def test_without_the_arm_no_search_client_is_built(tmp_path):
    def never(cfg: StudyConfig) -> tavily.Client:
        raise AssertionError("built a search client for a config without the arm")
    o = pipeline.run_day(DAY, tmp_path / "f", config=replace(_config(), retrieval=None),
                         deps=_deps(tmp_path, client=Recorder(), make_searcher=never))
    assert o.status == "done" and len(_records(tmp_path / "f")) == len(DUE) * 11


def test_the_credit_cap_stops_the_arm_and_the_rest_of_the_day_runs(tmp_path):
    out, stub = tmp_path / "forecasts", Stub()
    cfg = _config(season_credit_cap=9)                          # two games' searches, not three
    o = pipeline.run_day(DAY, out, config=cfg, deps=_deps(tmp_path, client=Recorder(),
                                                          stub=stub))
    names = [r["contestant"] for r in _records(out)]
    assert o.status == "done" and o.why and "credit cap" in o.why
    assert names.count("llm1+retrieval:line_aware") == 2 and len(stub.calls) == 8
    assert names.count("llm1:line_aware") == len(DUE)


def test_a_failed_search_is_a_lost_forecast_with_its_reason(tmp_path):
    out = tmp_path / "forecasts"
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, client=Recorder(), stub=Stub(fail_on=2)))
    r = next(r for r in _records(out) if r["game_id"] == DUE[0]
             and r["contestant"] == "llm1+retrieval:line_aware")
    assert o.status == "done" and r["p"] is None and r["search_credits"] == 1
    assert "search 2 of 4 failed" in r["retrieval_failure"] and KEY not in r["retrieval_failure"]
    assert r["paired_with"] == "llm1:line_aware" and r["packet_hash"] is None


def test_the_spending_guard_counts_the_arm_before_any_search(tmp_path):
    stub = Stub()
    bare = runner.max_run_cost(_config(), pending_llm_forecasts=len(DUE) * 8)
    cfg = replace(_config(), season_cap_usd=bare * 1.0001)     # enough for the bare arms only
    with pytest.raises(runner.BudgetExhausted):
        pipeline.run_day(DAY, tmp_path / "f", config=cfg,
                         deps=_deps(tmp_path, client=Recorder(), stub=stub))
    assert stub.calls == []
