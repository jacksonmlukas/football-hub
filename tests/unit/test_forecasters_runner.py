"""`hub.forecasters.runner`: one forecast run over a day's games, every contestant, once.

The prereg's *Forecast procedure*, rule by rule: pinned settings (temperature only where the
model accepts it), 5 samples, JSON output, a sample that fails to parse retried up to twice,
fewer than 3 valid samples is no forecast, the mean of the valid samples, [0.01, 0.99] clipping
for every contestant, idempotent per (game, contestant), a spending guard before any model
call, a lock before any model call, and the market snapshot id on every record. No network:
the client is a fake that answers from a script and counts its calls.
"""
from __future__ import annotations

import fcntl
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from forecasters_fixtures import KICKOFF, RUN_START, SNAPSHOT, _build

from hub.forecasters import prompt, runner
from hub.forecasters.config import Analysis, LlmSpec, StudyConfig, load
from hub.forecasters.packet import Condition, packet_hash

GID = "2026_06_BUF_KC"


def _spec(name: str = "m1", **kw) -> LlmSpec:
    base = LlmSpec(name=name, provider="fake", model_version=f"{name}-2026-01-01",
                   training_cutoff="2025-06", temperature=0.7, reasoning_effort=None,
                   max_output_tokens=300, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)
    return replace(base, **kw)


def _config(*specs: LlmSpec, cap: float = 100.0) -> StudyConfig:
    return StudyConfig(llms=specs or (_spec(),), season_cap_usd=cap, home_rate=0.55,
                       home_rate_seasons="2015-2025", est_input_tokens=1500,
                       analysis=Analysis(0.02, 0.25, "hub.forecasters.score@abcdef0"))


class FakeClient:
    """Answers each call from `script` in order, then repeats `default`."""

    def __init__(self, script: list[str] | None = None, default: str = '{"p_home_win": 0.6, "rationale": "ok"}'):
        self.script, self.default, self.calls = list(script or []), default, []

    def complete(self, spec: LlmSpec, system: str, user: str) -> runner.Completion:
        self.calls.append((spec.name, system, user))
        text = self.script.pop(0) if self.script else self.default
        return runner.Completion(text=text, input_tokens=1000, output_tokens=100)


class FakeHub:
    version = "deadbeef"

    def predict(self, game_id: str, market) -> float | None:
        return 0.62


def _game(**kw) -> runner.GameInputs:
    base = runner.GameInputs(game_id=GID, kickoff=KICKOFF, blind=_build(Condition.BLIND),
                             line_aware=_build(Condition.LINE_AWARE), refusal=None)
    return replace(base, **kw)


def _run(tmp_path: Path, *, client=None, config=None, games=None, now=None, hub=None):
    out = tmp_path / "forecasts"
    hub = hub or FakeHub()
    clock = iter([RUN_START + timedelta(minutes=i) for i in range(10_000)]) if now is None else now
    return runner.run(games or [_game()], snapshot_id=SNAPSHOT, config=config or _config(),
                      client=client or FakeClient(), hub=hub, out_dir=out, run_start=RUN_START,
                      run_id="12345", clock=lambda: next(clock)), out


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(out.rglob("*.jsonl")) if f.name != "exclusions.jsonl"
            for line in f.read_text().splitlines()]


# --- contestants --------------------------------------------------------------------------

def test_four_llms_give_eleven_contestants():
    cfg = _config(*(_spec(f"m{i}") for i in range(4)))
    names = runner.contestants(cfg)
    assert len(names) == 11
    assert {"hub", "home_constant", "coin_flip"} <= set(names)
    assert {"m0:blind", "m0:line_aware", "m3:blind", "m3:line_aware"} <= set(names)


def test_one_run_writes_every_contestant_once_with_the_snapshot_id(tmp_path):
    _, out = _run(tmp_path)
    recs = _records(out)
    assert sorted(r["contestant"] for r in recs) == sorted(runner.contestants(_config()))
    assert {r["market_snapshot_id"] for r in recs} == {SNAPSHOT}
    assert {r["workflow_run_id"] for r in recs} == {"12345"}


def test_the_baselines_are_the_home_rate_and_a_coin_flip(tmp_path):
    _, out = _run(tmp_path)
    by = {r["contestant"]: r for r in _records(out)}
    assert by["home_constant"]["p"] == 0.55 and by["coin_flip"]["p"] == 0.5


def test_the_hub_record_logs_its_commit(tmp_path):
    _, out = _run(tmp_path)
    hub = next(r for r in _records(out) if r["contestant"] == "hub")
    assert hub["model_version"] == "deadbeef" and hub["p"] == 0.62


def test_a_record_carries_what_the_prereg_lists(tmp_path):
    _, out = _run(tmp_path)
    r = next(r for r in _records(out) if r["contestant"] == "m1:line_aware")
    for k in ("game_id", "contestant", "model_version", "condition", "prompt_hash", "packet_hash",
              "market_snapshot_id", "samples", "mean", "p", "run_start", "finished_at",
              "workflow_run_id", "cost_usd"):
        assert k in r, k
    assert r["packet_hash"] == packet_hash(_build(Condition.LINE_AWARE))
    assert r["condition"] == "line_aware" and len(r["samples"]) == 5


def test_every_record_carries_the_runs_market_probability_unclipped(tmp_path):
    """#448 item 10: the benchmark scoring reads back is the run's de-vigged snapshot value,
    on every record of the game (blind ones too), and is never clipped (item 2). Planted: a
    market at 0.995, outside the contestants' clip."""
    aware = _build(Condition.LINE_AWARE)
    assert aware.market is not None
    hot = replace(aware, market=replace(aware.market, home_win_prob=0.995))
    _, out = _run(tmp_path, games=[_game(line_aware=hot)])
    recs = _records(out)
    assert {r["market_home_win_prob"] for r in recs} == {0.995}
    _, none = _run(tmp_path / "partial", games=[_game(line_aware=None)])
    assert {r["market_home_win_prob"] for r in _records(none)} == {None}


def test_a_pinned_model_config_is_refused_while_delta_or_the_h1_threshold_is_unset():
    """#448 item 8: `runner.require_pinned` refuses a model call while delta or the H1
    threshold is TBD -- each named on its own -- and does not ask for the H2 pin."""
    four = _config(*(_spec(f"m{i}") for i in range(4)))
    for unset, gap in ((replace(four.analysis, delta=None), "analysis.delta"),
                       (replace(four.analysis, h1_min_deviation_sd=None),
                        "analysis.h1_min_deviation_sd")):
        with pytest.raises(runner.NotPinned, match=gap) as exc:
            runner.require_pinned(replace(four, analysis=unset))
        assert "h2_implementation" not in str(exc.value)
    runner.require_pinned(replace(four, analysis=replace(four.analysis,
                                                         h2_implementation=None)))


# --- the LLM call: settings, samples, parsing, retries ------------------------------------

def test_temperature_is_sent_only_where_the_model_accepts_it():
    assert runner.call_settings(_spec(temperature=0.7)) == {"temperature": 0.7, "max_output_tokens": 300}
    reasoning = _spec(temperature=None, reasoning_effort="medium")
    assert runner.call_settings(reasoning) == {"reasoning_effort": "medium", "max_output_tokens": 300}


def test_blind_prompts_never_see_the_line_and_line_aware_ones_do():
    blind = prompt.render(_build(Condition.BLIND))
    aware = prompt.render(_build(Condition.LINE_AWARE))
    assert '"market"' not in blind and "spread" not in blind
    assert '"market"' in aware and '"home_spread"' in aware
    assert SNAPSHOT not in aware                  # the id is the record's, not the model's


def test_each_condition_is_called_with_its_own_packet(tmp_path):
    client = FakeClient()
    _run(tmp_path, client=client)
    users = [u for (_, _, u) in client.calls]
    assert sum('"market"' in u for u in users) == 5 and sum('"market"' not in u for u in users) == 5


@pytest.mark.parametrize(("text", "p"), [
    ('{"p_home_win": 0.61, "rationale": "x"}', 0.61),
    ('```json\n{"p_home_win": 0.3, "rationale": "x"}\n```', 0.3),
    ('{"p_home_win": 1, "rationale": "x"}', 1.0),
])
def test_valid_samples_parse(text, p):
    s = runner.parse_sample(text)
    assert s is not None and s.p == p


@pytest.mark.parametrize("text", [
    "0.6", '{"p": 0.6}', '{"p_home_win": "0.6", "rationale": "x"}', '{"p_home_win": 1.2, "rationale": "x"}',
    '{"p_home_win": -0.1, "rationale": "x"}', '{"p_home_win": NaN, "rationale": "x"}',
    '{"p_home_win": 0.6}', '{"p_home_win": true, "rationale": "x"}', "not json",
])
def test_invalid_samples_do_not_parse(text):
    assert runner.parse_sample(text) is None


def test_a_rationale_over_sixty_words_is_kept_to_sixty_and_flagged():
    s = runner.parse_sample(json.dumps({"p_home_win": 0.5, "rationale": " ".join(["w"] * 80)}))
    assert s is not None and len(s.rationale.split()) == 60 and s.truncated


def test_a_failed_parse_is_retried_up_to_twice_then_the_sample_is_lost(tmp_path):
    bad, good = "garbage", '{"p_home_win": 0.6, "rationale": "ok"}'
    # sample 1: two failures then success (3 calls); sample 2: three failures (lost); 3-5 fine
    client = FakeClient([bad, bad, good, bad, bad, bad, good, good, good])
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert len(client.calls) == 9
    assert rec.valid == 4 and rec.lost == 1 and rec.mean == pytest.approx(0.6)
    assert [s.attempts for s in rec.samples] == [3, 1, 1, 1]     # the lost sample is not kept
    assert len(rec.raw_failures) == 5                            # but every failed answer is


def test_fewer_than_three_valid_samples_is_no_forecast():
    client = FakeClient(["garbage"] * 15)
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert rec.p is None and rec.valid == 0 and len(client.calls) == 15


def test_three_valid_samples_is_a_forecast():
    good = '{"p_home_win": 0.7, "rationale": "ok"}'
    client = FakeClient(["x"] * 6 + [good] * 3)          # two samples lost, three kept
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert rec.valid == 3 and rec.p == pytest.approx(0.7)


@pytest.mark.parametrize(("ps", "mean", "clipped"), [
    ([1.0] * 5, 1.0, 0.99), ([0.0] * 5, 0.0, 0.01), ([0.2, 0.4, 0.6, 0.8, 1.0], 0.6, 0.6)])
def test_the_forecast_is_the_mean_of_valid_samples_clipped(ps, mean, clipped):
    client = FakeClient([json.dumps({"p_home_win": p, "rationale": "r"}) for p in ps])
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert rec.mean == pytest.approx(mean) and rec.p == pytest.approx(clipped)


def test_clipping_applies_to_every_contestant(tmp_path):
    class Sure(FakeHub):
        def predict(self, game_id, market):
            return 1.0
    _, out = _run(tmp_path, hub=Sure())
    assert next(r for r in _records(out) if r["contestant"] == "hub")["p"] == 0.99


def test_cost_is_tokens_times_the_pinned_prices():
    cfg = _config()
    rec = runner.forecast_llm(FakeClient(), cfg.llms[0], _build(Condition.BLIND), cfg)
    # 5 calls x (1000 in x $1/M + 100 out x $5/M) = 5 x 0.0015
    assert rec.cost_usd == pytest.approx(0.0075)


# --- the prompt hash ----------------------------------------------------------------------

def test_the_prompt_hash_is_stable_and_pinned():
    assert prompt.prompt_hash(Condition.BLIND) == prompt.prompt_hash(Condition.BLIND)
    assert prompt.prompt_hash(Condition.BLIND) != prompt.prompt_hash(Condition.LINE_AWARE)
    assert prompt.PINNED == {c: prompt.prompt_hash(c) for c in Condition}


def test_the_prompt_hash_does_not_depend_on_the_game():
    a = prompt.render(_build(Condition.BLIND))
    assert prompt.prompt_hash(Condition.BLIND) == prompt.prompt_hash(Condition.BLIND)
    assert "fetched_at" not in a and "served_from" not in a     # the stamps are not shown


def test_the_rendered_prompt_is_deterministic_for_one_packet():
    assert prompt.render(_build(Condition.LINE_AWARE)) == prompt.render(_build(Condition.LINE_AWARE))


# --- idempotency, the guard, the lock -----------------------------------------------------

def test_a_second_run_writes_nothing_and_calls_no_model(tmp_path):
    _run(tmp_path)
    client = FakeClient()
    _, out = _run(tmp_path, client=client)
    assert client.calls == []
    assert len(_records(out)) == len(runner.contestants(_config()))


def test_a_rerun_after_a_crash_fills_only_the_gaps(tmp_path):
    _, out = _run(tmp_path)
    f = next(out.rglob("*.jsonl"))
    kept = [line for line in f.read_text().splitlines() if '"m1:line_aware"' not in line]
    f.write_text("\n".join(kept) + "\n")
    client = FakeClient()
    _run(tmp_path, client=client)
    assert len(client.calls) == 5
    assert sorted(r["contestant"] for r in _records(out)) == sorted(runner.contestants(_config()))


def test_the_spending_guard_stops_before_any_model_call(tmp_path):
    client = FakeClient()
    cfg = _config(cap=0.001)                    # below one run's worst case
    with pytest.raises(runner.BudgetExhausted):
        _run(tmp_path, client=client, config=cfg)
    assert client.calls == []


def test_the_guard_counts_what_earlier_runs_spent(tmp_path):
    cfg = _config()
    worst = runner.max_run_cost(cfg, pending_llm_forecasts=2)
    _run(tmp_path, config=replace(cfg, season_cap_usd=worst + 0.01))   # spends ~0.015
    client = FakeClient()
    with pytest.raises(runner.BudgetExhausted):
        _run(tmp_path, client=client, config=replace(cfg, season_cap_usd=worst + 0.01),
             games=[_game(game_id="2026_06_X_Y")])
    assert client.calls == []


def test_the_worst_case_assumes_every_sample_retries_at_the_token_ceiling():
    cfg = _config()
    per_call = (1500 * 1.0 + 300 * 5.0) / 1e6
    assert runner.max_run_cost(cfg, pending_llm_forecasts=2) == pytest.approx(2 * 5 * 3 * per_call)


def test_a_held_lock_stops_the_run_before_any_model_call(tmp_path):
    out = tmp_path / "forecasts"
    out.mkdir()
    with open(out / runner.LOCK, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = FakeClient()
        with pytest.raises(runner.Locked):
            _run(tmp_path, client=client)
        assert client.calls == []


# --- exclusions ---------------------------------------------------------------------------

def test_a_refused_packet_is_forecast_by_no_one_and_logged(tmp_path):
    client = FakeClient()
    _, out = _run(tmp_path, client=client, games=[_game(blind=None, line_aware=None,
                                                         refusal="[injury_published] late")])
    assert client.calls == [] and _records(out) == []
    (log,) = list(out.rglob("exclusions.jsonl"))
    assert "injury_published" in log.read_text()


def test_a_refused_game_is_logged_once_and_stays_settled(tmp_path):
    refused = _game(blind=None, line_aware=None, refusal="[injury_published] late")
    _run(tmp_path, games=[refused])
    client = FakeClient()
    _, out = _run(tmp_path, client=client, games=[refused])
    _run(tmp_path, client=client, games=[_game()])                 # packets now build: still out
    assert len((out / runner.EXCLUSIONS).read_text().splitlines()) == 1
    assert runner.excluded(out) == {GID} and client.calls == [] and _records(out) == []


def test_no_line_means_no_line_aware_forecast_but_the_rest_still_run(tmp_path):
    _, out = _run(tmp_path, games=[_game(line_aware=None)])
    names = {r["contestant"] for r in _records(out)}
    assert "m1:blind" in names and "coin_flip" in names
    assert "m1:line_aware" not in names and "hub" not in names


def test_a_forecast_finishing_within_an_hour_of_kickoff_is_written_and_excluded(tmp_path):
    late = iter([KICKOFF - timedelta(minutes=59)] * 10_000)
    _, out = _run(tmp_path, now=late)
    assert all(r["excluded"] == "finished_within_60min_of_kickoff" for r in _records(out))


def test_an_early_finish_is_not_excluded(tmp_path):
    _, out = _run(tmp_path)
    assert all(r["excluded"] is None for r in _records(out))


# --- config -------------------------------------------------------------------------------

def test_the_committed_config_refuses_to_run_until_v0_pins_the_models():
    with pytest.raises(runner.NotPinned):
        runner.require_pinned(load())


def test_a_pinned_config_passes():
    runner.require_pinned(_config(*(_spec(f"m{i}") for i in range(4))))


# --- added after a mutation pass: three rules the tests above could not tell apart -------

def test_two_valid_samples_is_still_no_forecast():
    good = '{"p_home_win": 0.7, "rationale": "ok"}'
    client = FakeClient(["x"] * 9 + [good] * 2)          # three samples lost, two kept
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert rec.valid == 2 and rec.p is None and rec.mean is None


def test_the_forecast_is_the_mean_not_the_median():
    ps = [0.1, 0.1, 0.1, 0.9, 0.9]                       # mean 0.42, median 0.1
    client = FakeClient([json.dumps({"p_home_win": p, "rationale": "r"}) for p in ps])
    cfg = _config()
    rec = runner.forecast_llm(client, cfg.llms[0], _build(Condition.BLIND), cfg)
    assert rec.mean == pytest.approx(0.42)


def test_the_guard_prices_only_what_is_still_unwritten(tmp_path):
    cfg = _config()
    _run(tmp_path, config=cfg)
    _, spent = runner._written(tmp_path / "forecasts")
    # a cap that could not cover a fresh run of this game, but nothing is left to do
    _run(tmp_path, config=replace(cfg, season_cap_usd=spent + 1e-9))
