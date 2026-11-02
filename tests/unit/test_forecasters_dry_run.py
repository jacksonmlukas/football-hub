"""`hub.forecasters.dry_run`: one study day through the real adapters, writing only under `out` (#534).

Every transport here is a stub -- the four model providers, Tavily and The Odds API -- so no
call leaves the process. What the ticket asks of the dry run:

* the day runs once and everything it writes lands under `out`, with a summary beside it;
* the ceiling refuses before any call, any key read or any credit spent;
* no key, header or raw paid payload reaches the summary, the records or the printed table.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from forecasters_slate import (
    DAY,
    NAMES,
    RUN_START,
    injuries_payload,
    odds_payload,
    schedule_frame,
)

from hub.fetch import odds, tavily
from hub.forecasters import cli, dry_run, pipeline, providers, retrieval, runner, sources
from hub.forecasters.config import Analysis, LlmSpec, RetrievalSpec, StudyConfig
from hub.forecasters.inputs import QbSource, SourceStamp
from hub.forecasters.qb import QB_TIME_BASIS

KEY = "sk-test-SECRET-0123456789"
ODDS_KEY = "odds-SECRET-0123456789"
PAID = "PAID-PAYLOAD-SENTINEL"
ANSWER = '{"p_home_win": 0.6, "rationale": "home field"}'
ENV = {"ANTHROPIC_API_KEY": KEY, "OPENAI_API_KEY": KEY, "GEMINI_API_KEY": KEY,
       "TAVILY_API_KEY": KEY}


def _config(**kw) -> StudyConfig:
    llms = tuple(LlmSpec(name=f"llm{i + 1}", provider=p, model_version=f"{p}-m{i}",
                         training_cutoff="2025-06", temperature=None, reasoning_effort=None,
                         max_output_tokens=1000, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
                 for i, p in enumerate(("anthropic", "openai", "google", "openai")))
    base = StudyConfig(llms=llms, season_cap_usd=150.0, home_rate=0.547,
                       home_rate_seasons="2016-2025", est_input_tokens=3000,
                       analysis=Analysis(0.02, 0.25, "hub.forecasters.score@abcdef0"),
                       retrieval=RetrievalSpec("llm4", 1500, 0.008))
    return replace(base, **kw)


class Wire:
    """Every transport the run reaches, stubbed and counted. `reply_for` can be overridden."""

    def __init__(self, *, usage: bool = True, out_tokens: int = 120, fail: bool = False) -> None:
        self.usage, self.out_tokens, self.fail = usage, out_tokens, fail
        self.model_calls = 0
        self.search_calls = 0
        self.odds_calls = 0
        self.keys_built = 0
        self.seen: list[tuple[str, Mapping[str, str]]] = []

    def model(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]):
        self.model_calls += 1
        self.seen.append((url, headers))
        if self.fail:
            raise ConnectionError(f"boom {KEY}")
        u = self.out_tokens
        if "anthropic" in url:
            r: dict[str, Any] = {"content": [{"type": "text", "text": ANSWER}], "raw": PAID}
            if self.usage:
                r["usage"] = {"input_tokens": 900, "output_tokens": u}
        elif "openai" in url:
            r = {"output": [{"type": "message", "content": [{"type": "output_text",
                                                             "text": ANSWER}]}], "raw": PAID}
            if self.usage:
                r["usage"] = {"input_tokens": 900, "output_tokens": u}
        else:
            r = {"candidates": [{"content": {"parts": [{"text": ANSWER}]}}], "raw": PAID}
            if self.usage:
                r["usageMetadata"] = {"promptTokenCount": 900, "candidatesTokenCount": u - 40,
                                      "thoughtsTokenCount": 40}
        return r

    def search(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]):
        self.search_calls += 1
        return {"results": [{"url": "https://example.com/a", "title": "t", "content": "news",
                             "published_date": "Sat, 17 Oct 2026 14:00:00 GMT"}],
                "answer": PAID, "usage": {"credits": 1}}

    def odds(self, params: Mapping[str, Any], key: str):
        self.odds_calls += 1
        payload = [{**ev, "paid_extra": PAID} for ev in odds_payload()]
        return payload, {"x-requests-remaining": "450"}


def _qb(run_start, stamp: SourceStamp) -> QbSource:
    return QbSource(stamp=stamp, updated_at=run_start - timedelta(hours=20),
                    updated_at_basis=QB_TIME_BASIS, starters={t: f"{t} QB" for t in NAMES.values()})


def _deps(tmp_path: Path, wire: Wire, *, env: Mapping[str, str] = ENV) -> pipeline.Deps:
    tick = itertools.count()

    def make_client(cfg: StudyConfig) -> runner.Client:
        wire.keys_built += 1
        return providers.Router(cfg, env=env, transport=wire.model)

    injuries = injuries_payload()
    injuries["injuries"][0]["injuries"][0]["headline"] = PAID
    return pipeline.Deps(
        schedule_frame=lambda season: schedule_frame(), injuries_payload=lambda: injuries,
        qb_source=_qb, make_client=make_client,
        make_hub=lambda season, week: sources.RatingsHub(season, week, version="testsha"),
        team_names=lambda games: NAMES, odds_get=wire.odds, odds_key=ODDS_KEY,
        make_searcher=lambda cfg: tavily.Client(env=env, transport=wire.search),
        clock=lambda: RUN_START + timedelta(seconds=10 * next(tick)),
        run_id="999", last_good_root=tmp_path / "elsewhere" / "last_good")


@pytest.fixture(autouse=True)
def _cwd_and_hub_state(tmp_path, monkeypatch) -> Path:
    """Run from an empty directory and point the hub's balance file at a visible spot, so a
    write outside `out` shows up as a file that should not exist."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(odds, "STATE", tmp_path / "hub_state" / "odds.json")
    return cwd


def _go(tmp_path, wire=None, *, ceiling=50.0, config=None, env=ENV, echo=None):
    wire = wire or Wire()
    out = tmp_path / "out"
    lines: list[str] = []
    ticks = itertools.count()
    code = dry_run.run_dry(
        DAY, out, ceiling_usd=ceiling, config=config or _config(),
        deps=_deps(tmp_path, wire, env=env), refusals=(*cli.REFUSALS, (dry_run.CeilingExceeded, 9)),
        echo=echo or lines.append, timer=lambda: next(ticks) * 0.5)
    summary = json.loads((out / dry_run.SUMMARY).read_text())
    return code, summary, out, lines, wire


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# --- the run ------------------------------------------------------------------------------

def test_a_day_runs_once_and_writes_only_under_out(tmp_path, _cwd_and_hub_state):
    code, s, out, _, wire = _go(tmp_path)
    assert code == 0 and s["status"] == "done" and s["refusal"] is None
    assert wire.odds_calls == 1 and wire.model_calls > 0
    assert (out / dry_run.SUMMARY).exists() and list(out.rglob("*.jsonl"))
    assert (out / pipeline.ODDS_STATE).exists() and (out / pipeline.TAVILY_STATE).exists()
    assert list((out / dry_run.LAST_GOOD).glob("*.json"))
    # Nothing outside `out`: the working directory, the hub's balance file, the default root.
    assert _files(_cwd_and_hub_state) == []
    assert not odds.STATE.exists() and not (tmp_path / "elsewhere").exists()


def test_the_summary_reports_each_model_the_credits_and_the_feed_shape(tmp_path):
    _, s, out, _, _ = _go(tmp_path)
    rows = {r["contestant"]: r for r in s["models"]}
    assert set(rows) == {f"llm{i}:{c}" for i in (1, 2, 3, 4) for c in ("blind", "line_aware")} | {
        "llm4+retrieval:line_aware"}
    for r in rows.values():
        assert r["calls"] == r["parsed"] > 0 and r["unparsed"] == 0 and r["provider_errors"] == 0
        assert r["output_tokens_max"] == 120 and r["output_tokens_cap"] == 1000
        assert r["cap_hits"] == 0 and r["usage_missing"] == 0
        assert r["latency_s_mean"] == 0.5 and r["latency_s_max"] == 0.5
    assert rows["llm1:blind"]["cost_usd"] == pytest.approx(
        rows["llm1:blind"]["calls"] * (900 * 1.0 + 120 * 5.0) / 1e6, abs=1e-5)
    assert s["tavily"]["credits"] == s["tavily"]["searches"] == 16 and s["due_games"] == 4
    assert s["odds_api"]["remaining"] == 450
    inj = s["injuries"]
    assert inj["captured"] and inj["items"] == 4 and inj["team_blocks"] == 3
    assert inj["parsed_entries"] + inj["dropped_items"] == 4
    assert {"athlete", "date", "status"} <= set(inj["item_keys"])
    assert "displayName" in inj["athlete_keys"] and "injuries" in inj["top_level_keys"]
    # The summary's dollars are the records' dollars.
    spent = sum(json.loads(line)["cost_usd"] for f in out.rglob("*.jsonl")
                for line in f.read_text().splitlines())
    assert s["spend_usd"]["total"] == pytest.approx(spent, abs=1e-4)


def test_output_tokens_are_read_against_the_cap_and_a_missing_usage_block_is_flagged(tmp_path):
    _, s, *_ = _go(tmp_path, Wire(out_tokens=1000))
    assert all(r["cap_hits"] == r["calls"] for r in s["models"])
    _, s, *_ = _go(tmp_path / "b", Wire(usage=False))
    assert all(r["usage_missing"] == r["calls"] for r in s["models"])


def test_a_failing_provider_is_counted_not_crashed(tmp_path):
    code, s, *_ = _go(tmp_path, Wire(fail=True))
    assert code == 0
    for r in s["models"]:
        assert r["provider_errors"] == r["calls"] > 0 and r["parsed"] == 0
        assert r["usage_missing"] == 0


def test_a_used_out_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.jsonl").write_text("{}\n")
    with pytest.raises(dry_run.OutNotEmpty):
        _go(tmp_path)


# --- the ceiling --------------------------------------------------------------------------

def test_the_ceiling_refuses_before_any_key_credit_or_call(tmp_path):
    wire = Wire()
    code, s, out, lines, wire = _go(tmp_path, wire, ceiling=0.01)
    assert code == dry_run.CEILING_EXIT == 9
    assert s["status"] == "refused" and "ceiling" in s["refusal"]
    assert s["estimate_usd"] > 0.01
    assert (wire.keys_built, wire.model_calls, wire.search_calls, wire.odds_calls) == (0, 0, 0, 0)
    assert [p.name for p in _files(out)] == [dry_run.SUMMARY]
    assert any("refused" in line for line in lines)


def test_the_estimate_is_the_runners_worst_case_and_the_ceiling_is_inclusive(tmp_path):
    cfg = _config()
    worst, n = dry_run.estimate(DAY, cfg, schedule_frame())
    assert n == 4
    assert worst == pytest.approx(runner.max_run_cost(
        cfg, pending_llm_forecasts=4 * 4 * 2, pending_retrieval=4))
    code, s, *_ = _go(tmp_path, ceiling=worst)
    assert code == 0 and s["estimate_usd"] == pytest.approx(worst, abs=1e-4)
    code, s, *_ = _go(tmp_path / "under", ceiling=worst - 0.01)
    assert code == 9


def test_the_runners_guard_is_a_second_fence_at_the_same_ceiling(tmp_path, monkeypatch):
    """The ceiling replaces the season cap handed to the runner."""
    seen = []
    real = runner.run

    def spy(*a, **kw):
        seen.append(kw["config"].season_cap_usd)
        return real(*a, **kw)
    monkeypatch.setattr(runner, "run", spy)
    _go(tmp_path, ceiling=50.0)
    assert seen == [50.0]


def test_a_missing_key_is_a_refusal_in_the_summary_before_the_snapshot(tmp_path):
    code, s, _, _, wire = _go(tmp_path, env={})
    assert code == 5 and s["status"] == "refused" and "API_KEY" in s["refusal"]
    assert wire.odds_calls == 0 and wire.model_calls == 0


def test_a_bad_ceiling_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _go(tmp_path, ceiling=0)


# --- no secret, header or raw payload -----------------------------------------------------

def test_no_key_header_or_raw_payload_reaches_the_artifact_or_the_log(tmp_path, capsys):
    _, _, out, lines, wire = _go(tmp_path)
    blob = "\n".join(lines) + capsys.readouterr().out
    for f in _files(out):
        blob += f.read_text()
    for needle in (KEY, ODDS_KEY, PAID, "SECRET", "x-api-key", "authorization", "Bearer"):
        assert needle not in blob, needle
    assert wire.model_calls > 0 and wire.search_calls > 0       # the sentinels were on the wire


def test_a_failed_call_does_not_leak_the_key_through_the_summary(tmp_path):
    _, _, out, lines, _ = _go(tmp_path, Wire(fail=True))
    blob = "\n".join(lines) + "".join(f.read_text() for f in _files(out))
    assert KEY not in blob


def test_injury_shape_is_counts_and_keys_only():
    payload = injuries_payload()
    payload["injuries"][0]["injuries"][0]["headline"] = PAID
    shape = dry_run.injury_shape(payload, as_of=RUN_START + timedelta(days=1))
    assert PAID not in json.dumps(shape) and "Travis Kelce" not in json.dumps(shape)
    assert shape["items"] == 4 and shape["parsed_entries"] == 4 and shape["parsed_teams"] == 3
    assert "headline" in shape["item_keys"]
    assert shape["status_counts"] == {"Doubtful": 1, "Out": 2, "Questionable": 1}
    assert dry_run.injury_shape(None, as_of=RUN_START) == {"captured": False}


# --- the command --------------------------------------------------------------------------

def test_the_cli_runs_the_dry_run_and_maps_exit_codes(tmp_path):
    lines: list[str] = []
    args = ["dry-run", "--out", str(tmp_path / "o"), "--day", DAY.isoformat(),
            "--ceiling-usd", "50"]
    assert cli.main(args, echo=lines.append, deps=_deps(tmp_path, Wire()),
                    config=_config()) == 0
    assert (tmp_path / "o" / dry_run.SUMMARY).exists() and lines
    args[-1] = "0.01"
    assert cli.main([*args[:2], str(tmp_path / "o2"), *args[3:]], echo=lines.append,
                    deps=_deps(tmp_path, Wire()), config=_config()) == 9
    assert cli.main(args, echo=lines.append, deps=_deps(tmp_path, Wire()),
                    config=_config()) == 2                         # `o` is no longer fresh
    args[-1] = "-1"
    assert cli.main([*args[:2], str(tmp_path / "o3"), *args[3:]], echo=lines.append,
                    deps=_deps(tmp_path, Wire()), config=_config()) == 2


def test_the_cli_refuses_an_unpinned_config_before_anything(tmp_path):
    wire = Wire()
    cfg = _config()
    cfg = replace(cfg, llms=(replace(cfg.llms[0], model_version="TBD"), *cfg.llms[1:]))
    assert cli.main(["dry-run", "--out", str(tmp_path / "o"), "--day", "2026-10-18",
                     "--ceiling-usd", "5"], echo=lambda _: None, deps=_deps(tmp_path, wire),
                    config=cfg) == 3
    assert wire.keys_built == 0 and not (tmp_path / "o").exists()


def test_retrieval_label_and_table_render(tmp_path):
    _, _, _, lines, _ = _go(tmp_path)
    text = "\n".join(lines)
    assert f"llm4+{retrieval.ARM}:line_aware" in text and "out max/cap" in text
    assert "tavily credits: 16" in text and "espn injuries:" in text
