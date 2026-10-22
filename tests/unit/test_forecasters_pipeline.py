"""`hub.forecasters.pipeline`: last weekend's slate, dry-run end to end (#447).

No real LLM call and no Odds API credit: the model client is a fake (or the real provider
adapters on a stubbed transport), the odds pull answers from a synthetic payload, and the
schedule, injury and QB sources are handed in. What the ticket asks of the dry run:

* every due game gets eleven records or a logged exclusion;
* rerunning writes nothing -- and spends nothing, not even the market snapshot;
* every record carries the snapshot id;
* the Odds API balance goes to the study's own state file on the `forecasts` checkout, never
  the hub's `state/odds.json` (#445, adopted (a)).
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from forecasters_slate import (
    DAY,
    DUE,
    NAMES,
    RUN_START,
    injuries_payload,
    odds_payload,
    schedule_frame,
)

from hub.fetch import odds
from hub.forecasters import cli, pipeline, providers, qb, runner, sources
from hub.forecasters.config import LlmSpec, StudyConfig
from hub.forecasters.inputs import QbSource, SourceStamp

PROVIDERS = ("anthropic", "openai", "google", "google")


def _config() -> StudyConfig:
    llms = tuple(LlmSpec(name=f"llm{i + 1}", provider=p, model_version=f"{p}-model-{i}",
                         training_cutoff="2025-06", temperature=None, reasoning_effort=None,
                         max_output_tokens=2000, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
                 for i, p in enumerate(PROVIDERS))
    return StudyConfig(llms=llms, season_cap_usd=150.0, home_rate=0.547,
                       home_rate_seasons="2016-2025", est_input_tokens=3000)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, spec: LlmSpec, system: str, user: str) -> runner.Completion:
        self.calls += 1
        return runner.Completion('{"p_home_win": 0.58, "rationale": "fine"}', 900, 80)


class Odds:
    """The Odds API's `get`, answering from a synthetic payload; counts what it was asked."""

    def __init__(self, payload: list[dict] | None = None, fail: bool = False) -> None:
        self.payload, self.fail, self.calls = payload or odds_payload(), fail, 0

    def __call__(self, params: Mapping[str, Any], key: str) -> tuple[Any, Mapping[str, str]]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("odds down")
        return self.payload, {"x-requests-remaining": "450"}


def _qb(run_start: datetime, stamp: SourceStamp) -> QbSource:
    starters = {t: f"{t} QB" for t in NAMES.values()}
    return QbSource(stamp=stamp, updated_at=run_start - timedelta(hours=20),
                    updated_at_basis=qb.QB_TIME_BASIS, starters=starters)


def _deps(tmp_path: Path, *, client: Any = None, odds_get: Odds | None = None,
          injuries: dict | None = None, make_client: Any = None) -> pipeline.Deps:
    tick = itertools.count()
    fake = client or FakeClient()
    return pipeline.Deps(
        schedule_frame=lambda season: schedule_frame(),
        injuries_payload=lambda: injuries or injuries_payload(),
        qb_source=_qb,
        make_client=make_client or (lambda cfg: fake),
        make_hub=lambda season, week: sources.RatingsHub(season, week, version="testsha"),
        team_names=lambda games: NAMES,
        odds_get=odds_get or Odds(), odds_key="test-odds-key",
        clock=lambda: RUN_START + timedelta(seconds=10 * next(tick)),
        run_id="999", last_good_root=tmp_path / "last_good")


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(out.rglob("*.jsonl")) if f.name != runner.EXCLUSIONS
            for line in f.read_text().splitlines() if line.strip()]


def _files(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != runner.LOCK}


@pytest.fixture(autouse=True)
def _hub_state(tmp_path, monkeypatch) -> Path:
    """The hub's own balance file, redirected so a write to it is visible and harmless."""
    hub = tmp_path / "hub_state" / "odds.json"
    monkeypatch.setattr(odds, "STATE", hub)
    return hub


# --- the dry run --------------------------------------------------------------------------

def test_last_weekends_slate_gets_eleven_records_a_game_with_the_snapshot_id(tmp_path):
    out, client = tmp_path / "forecasts", FakeClient()
    o = pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path, client=client))
    assert o.status == "done" and o.games == DUE and o.excluded == ()
    recs = _records(out)
    names = runner.contestants(_config())
    for gid in DUE:
        assert sorted(r["contestant"] for r in recs if r["game_id"] == gid) == sorted(names)
    assert {r["game_id"] for r in recs} == set(DUE)               # London/Thu/Mon not due
    assert o.snapshot_id and {r["market_snapshot_id"] for r in recs} == {o.snapshot_id}
    assert client.calls == len(DUE) * 8 * 5                         # 4 LLMs x 2 conditions x 5
    assert {r["model_version"] for r in recs if r["contestant"] == "hub"} == {"testsha"}
    assert all(r["excluded"] is None for r in recs)


def test_the_line_aware_packet_carries_this_runs_snapshot_and_the_blind_one_none(tmp_path):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    by = {(r["game_id"], r["contestant"]): r for r in _records(out)}
    blind, aware = by[(DUE[0], "llm1:blind")], by[(DUE[0], "llm1:line_aware")]
    assert blind["packet_hash"] != aware["packet_hash"]
    hub = by[(DUE[0], "hub")]
    assert hub["p"] is not None and 0.5 < hub["p"] < 0.6           # home -1.5 through a normal


def test_rerunning_writes_nothing_and_spends_nothing(tmp_path):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    before = _files(out)
    client, again = FakeClient(), Odds()
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, client=client, odds_get=again))
    assert o.status == "nothing_left"
    assert _files(out) == before and client.calls == 0 and again.calls == 0


def test_a_crash_mid_run_is_filled_without_paying_twice(tmp_path):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    day_file = next(out.rglob("2026-10-18.jsonl"))
    lines = day_file.read_text().splitlines()
    lost = [json.loads(x)["contestant"] for x in lines[-4:]]
    assert lost == ["llm4:line_aware", "hub", "home_constant", "coin_flip"]
    day_file.write_text("\n".join(lines[:-4]) + "\n")              # a crash lost these four
    client = FakeClient()
    o = pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path, client=client))
    assert o.status == "done" and len(_records(out)) == len(lines)
    assert client.calls == 5                                        # one LLM forecast, once


def test_the_balance_goes_to_the_studys_own_state_file_not_the_hubs(tmp_path, _hub_state):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    own = out / pipeline.ODDS_STATE
    assert own == out / "state" / "odds.json"
    assert json.loads(own.read_text())["remaining"] == 450
    assert json.loads(own.read_text())["asked_for"].startswith("study:forecast:")
    assert not _hub_state.exists()


def test_a_day_with_no_games_spends_nothing(tmp_path):
    o, client = Odds(), FakeClient()
    out = tmp_path / "forecasts"
    res = pipeline.run_day(DAY + timedelta(days=2), out, config=_config(),
                           deps=_deps(tmp_path, client=client, odds_get=o))
    assert res.status == "nothing_due" and o.calls == 0 and client.calls == 0
    assert not out.exists()


# --- refusals and degradation -------------------------------------------------------------

def test_a_refused_packet_is_an_exclusion_line_and_no_one_forecasts_it(tmp_path):
    late = injuries_payload(dated=RUN_START + timedelta(hours=31))  # nothing before run start
    out, client = tmp_path / "forecasts", FakeClient()
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, client=client, injuries=late))
    assert o.excluded == DUE and _records(out) == [] and client.calls == 0
    lines = (out / runner.EXCLUSIONS).read_text().splitlines()
    assert sorted(json.loads(x)["game_id"] for x in lines) == sorted(DUE)
    assert all("injury_published" in json.loads(x)["reason"] for x in lines)
    before = _files(out)
    again = Odds()
    o2 = pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path, odds_get=again))
    assert o2.status == "nothing_left" and _files(out) == before and again.calls == 0


def test_no_market_snapshot_runs_the_blind_half_and_reports_partial(tmp_path):
    out = tmp_path / "forecasts"
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, odds_get=Odds(fail=True)))
    assert o.status == "partial" and o.snapshot_id is None
    assert o.why and o.why.startswith("no market snapshot") and "test-odds-key" not in o.why
    names = {r["contestant"] for r in _records(out)}
    assert "llm1:blind" in names and "coin_flip" in names
    assert not any(n.endswith(":line_aware") for n in names) and "hub" not in names
    # The next trigger fills the line-aware half from a fresh snapshot.
    o2 = pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    assert o2.status == "done"
    assert len(_records(out)) == len(DUE) * 11


def test_a_game_with_no_line_still_gets_its_blind_forecasts(tmp_path):
    out = tmp_path / "forecasts"
    payload = odds_payload(DUE[:3])
    o = pipeline.run_day(DAY, out, config=_config(),
                         deps=_deps(tmp_path, odds_get=Odds(payload)))
    assert o.status == "done"
    last = {r["contestant"] for r in _records(out) if r["game_id"] == DUE[3]}
    assert "llm1:blind" in last and "llm1:line_aware" not in last and "hub" not in last


def test_a_failed_source_fetch_serves_the_last_good_copy(tmp_path):
    out = tmp_path / "forecasts"
    pipeline.run_day(DAY, out, config=_config(), deps=_deps(tmp_path))
    deps = _deps(tmp_path)

    def down() -> dict:
        raise ConnectionError("espn down")
    deps.injuries_payload = down
    (out / "2026").joinpath("2026-10-18.jsonl").unlink()            # force a fresh run
    o = pipeline.run_day(DAY, out, config=_config(), deps=deps)
    assert o.status == "done" and len(_records(out)) == len(DUE) * 11


def test_a_missing_key_refuses_before_the_snapshot_spends_a_credit(tmp_path):
    o = Odds()

    def no_keys(cfg: StudyConfig) -> runner.Client:
        return providers.Router(cfg, env={})
    with pytest.raises(providers.MissingKey):
        pipeline.run_day(DAY, tmp_path / "forecasts", config=_config(),
                         deps=_deps(tmp_path, odds_get=o, make_client=no_keys))
    assert o.calls == 0


# --- the real adapters on a stubbed transport ----------------------------------------------

def test_the_provider_adapters_drive_the_whole_day_without_a_network_call(tmp_path):
    answer = '{"p_home_win": 0.57, "rationale": "x"}'
    replies = {
        "api.anthropic.com": {"content": [{"type": "text", "text": answer}],
                              "usage": {"input_tokens": 1000, "output_tokens": 50}},
        "api.openai.com": {"output": [{"type": "message", "content": [
            {"type": "output_text", "text": answer}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 50}},
        "generativelanguage.googleapis.com": {
            "candidates": [{"content": {"parts": [{"text": answer}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 50}},
    }
    hosts: list[str] = []

    def transport(url: str, headers: Mapping[str, str],
                  body: Mapping[str, Any]) -> Mapping[str, Any]:
        host = url.split("/")[2]
        hosts.append(host)
        return replies[host]
    env = dict.fromkeys(providers.KEY_ENV.values(), "k-0123456789")
    deps = _deps(tmp_path, make_client=lambda cfg: providers.Router(cfg, env=env,
                                                                     transport=transport))
    out = tmp_path / "forecasts"
    o = pipeline.run_day(DAY, out, config=_config(), deps=deps)
    assert o.status == "done" and len(_records(out)) == len(DUE) * 11
    assert set(hosts) == set(replies) and len(hosts) == len(DUE) * 8 * 5
    spent = sum(r["cost_usd"] for r in _records(out))
    assert spent == pytest.approx(len(hosts) * (1000 * 1.0 + 50 * 5.0) / 1e6)


# --- the command line ----------------------------------------------------------------------

def test_the_cli_runs_the_day_and_says_what_it_did(tmp_path):
    lines: list[str] = []
    rc = cli.main(["run", "--out", str(tmp_path / "forecasts"), "--day", DAY.isoformat()],
                  echo=lines.append, deps=_deps(tmp_path), config=_config())
    assert rc == 0
    summary = json.loads(lines[-1])
    assert summary["status"] == "done" and summary["games"] == len(DUE)


def test_the_cli_says_partial_with_its_own_exit_code(tmp_path):
    rc = cli.main(["run", "--out", str(tmp_path / "f"), "--day", DAY.isoformat()],
                  echo=lambda _: None, deps=_deps(tmp_path, odds_get=Odds(fail=True)),
                  config=_config())
    assert rc == 6


@pytest.mark.parametrize(("exc", "code"), [
    (providers.MissingKey("OPENAI_API_KEY is not set."), 5),
    (runner.Locked("held"), 7),
    (runner.BudgetExhausted("cap"), 8),
])
def test_the_cli_refuses_in_one_sentence(tmp_path, exc, code):
    def boom(cfg: StudyConfig) -> runner.Client:
        raise exc
    lines: list[str] = []
    rc = cli.main(["run", "--out", str(tmp_path / "f"), "--day", DAY.isoformat()],
                  echo=lines.append, deps=_deps(tmp_path, make_client=boom), config=_config())
    assert rc == code and lines == [f"refused: {exc}"]


def test_the_cli_refuses_when_a_source_has_no_last_good(tmp_path):
    deps = _deps(tmp_path)

    def down() -> dict:
        raise ConnectionError("espn down")
    deps.injuries_payload = down
    rc = cli.main(["run", "--out", str(tmp_path / "f"), "--day", DAY.isoformat()],
                  echo=lambda _: None, deps=deps, config=_config())
    assert rc == 4


def test_the_cli_still_refuses_an_unpinned_config_before_anything(tmp_path):
    o = Odds()
    cfg = replace(_config(), season_cap_usd=0.0)
    rc = cli.main(["run", "--out", str(tmp_path / "f"), "--day", DAY.isoformat()],
                  echo=lambda _: None, deps=_deps(tmp_path, odds_get=o), config=cfg)
    assert rc == 3 and o.calls == 0
