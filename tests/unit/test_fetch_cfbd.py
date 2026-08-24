"""CFBD fetch layer.

`Makefile:11,18` and `weekly-slate/SKILL.md:17` both invoke this and it did not exist.

The free tier is 1,000 calls a month and `docs/cfbd-quota.md` is unambiguous about how it
dies: one call per team is 136 calls for a week that one bulk call covers. The plan asks
for that to be **impossible by construction, not by convention**, so these tests are mostly
about what the module refuses rather than what it fetches.

Three independent guards, because a comment is not a control:

  * the signature cannot express a team -- there is no parameter for one
  * a params dict carrying a team or game key is rejected by name
  * a per-run call ceiling means a 136-iteration loop cannot finish even if the first two
    were somehow bypassed

Multiple keys and rate-limit circumvention are explicit terms violations that get access
revoked, so the monthly budget is a hard stop rather than a warning.
"""
import json

import polars as pl
import pytest

from hub.fetch import cfbd


@pytest.fixture(autouse=True)
def fresh_run():
    """Each test is its own run. Without this the ceiling leaks between tests, which is
    the same defect a long-lived process would hit."""
    cfbd.reset_run_budget()


@pytest.fixture
def paths(tmp_path):
    return {"cache": tmp_path / "cache", "quota": tmp_path / "quota.json"}


@pytest.fixture
def transport(monkeypatch):
    """Records every call the module would have made."""
    calls = []

    def _install(payload=None):
        def _fake(path, params, key):
            calls.append((path, dict(params)))
            return payload if payload is not None else [{"id": 1, "week": 1}]
        monkeypatch.setattr(cfbd, "_http_get", _fake)
        monkeypatch.setattr(cfbd, "_api_key", lambda: "test-key")
        return calls
    return _install


# --- looping is impossible ------------------------------------------------

def test_bulk_has_no_parameter_for_a_team(transport, paths):
    """The first guard: you cannot ask for one team because there is nowhere to put it."""
    transport()
    with pytest.raises(TypeError):
        cfbd.bulk("games", year=2026, team="Alabama",  # type: ignore[call-arg]
                  cache=paths["cache"], quota_path=paths["quota"])


@pytest.mark.parametrize("bad", ["team", "home", "away", "gameId", "game_id", "conference"])
def test_team_and_game_params_are_rejected_by_name(transport, paths, bad):
    transport()
    with pytest.raises(cfbd.LoopRefused):
        cfbd.bulk("games", year=2026, extra={bad: "x"},
                  cache=paths["cache"], quota_path=paths["quota"])


def test_the_refusal_explains_the_alternative(transport, paths):
    transport()
    with pytest.raises(cfbd.LoopRefused) as e:
        cfbd.bulk("games", year=2026, extra={"team": "Alabama"},
                  cache=paths["cache"], quota_path=paths["quota"])
    assert "filter" in str(e.value).lower()


def test_a_loop_over_teams_cannot_complete(transport, paths):
    """The third guard, and the one that holds if the others are edited away.

    136 FBS teams. The run ceiling stops it long before the monthly budget notices.
    """
    transport()
    with pytest.raises(cfbd.QuotaExceeded):
        for i in range(136):
            cfbd.bulk("games", year=2026, week=i + 1,
                      cache=paths["cache"], quota_path=paths["quota"])


def test_a_normal_week_stays_under_the_ceiling(transport, paths):
    """The documented weekly budget is 5-8 calls, so the ceiling must not bite it."""
    calls = transport()
    cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    assert 0 < len(calls) <= cfbd.MAX_CALLS_PER_RUN


def test_every_call_is_year_or_week_scoped(transport, paths):
    """No request may carry anything narrower than a week."""
    calls = transport()
    cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    for _, params in calls:
        assert set(params) <= {"year", "week", "seasonType", "classification"}


# --- quota accounting -----------------------------------------------------

def test_quota_starts_at_zero(paths):
    assert cfbd.quota_used(paths["quota"]) == 0


def test_each_call_is_recorded(transport, paths):
    transport()
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"])
    cfbd.bulk("lines", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"])
    assert cfbd.quota_used(paths["quota"]) == 2


def test_a_cached_pull_costs_nothing(transport, paths):
    calls = transport()
    for _ in range(3):
        cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                  quota_path=paths["quota"])
    assert len(calls) == 1, "the whole point of caching is quota, not speed"
    assert cfbd.quota_used(paths["quota"]) == 1


def test_the_monthly_budget_is_a_hard_stop(transport, paths):
    transport()
    paths["quota"].write_text(json.dumps({cfbd._month_key(): cfbd.FREE_TIER_MONTHLY}))
    with pytest.raises(cfbd.QuotaExceeded):
        cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                  quota_path=paths["quota"])


def test_usage_is_tracked_per_month(paths):
    paths["quota"].write_text(json.dumps({"2000-01": 900, cfbd._month_key(): 7}))
    assert cfbd.quota_used(paths["quota"]) == 7, "last month's spend is not this month's"


def test_a_corrupt_counter_does_not_crash_the_pipeline(paths):
    paths["quota"].write_text("not json")
    assert cfbd.quota_used(paths["quota"]) == 0


# --- reporting ------------------------------------------------------------

def test_quota_report_names_used_and_remaining(paths, capsys):
    paths["quota"].write_text(json.dumps({cfbd._month_key(): 35}))
    cfbd.quota_report(paths["quota"])
    out = capsys.readouterr().out
    assert "35" in out and "1,000" in out and "965" in out


def test_quota_report_works_without_a_key(paths, monkeypatch, capsys):
    """`make check` has to work on a machine that has never had a CFBD key."""
    monkeypatch.setattr(cfbd, "_api_key", lambda: None)
    assert cfbd.main(["--quota", "--quota-path", str(paths["quota"])]) == 0
    assert "1,000" in capsys.readouterr().out


# --- degradation ----------------------------------------------------------

def test_missing_key_degrades_loudly_rather_than_erroring(paths, monkeypatch, capsys):
    """CLAUDE.md: a module must produce a usable answer with zero attention."""
    monkeypatch.setattr(cfbd, "_api_key", lambda: None)
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    assert "CFBD_API_KEY" in capsys.readouterr().err


def test_unknown_endpoint_names_the_known_ones(transport, paths):
    transport()
    with pytest.raises(cfbd.LoopRefused) as e:
        cfbd.bulk("nope", year=2026, cache=paths["cache"], quota_path=paths["quota"])
    assert "games" in str(e.value)


def test_week_returns_a_frame_per_endpoint(transport, paths):
    transport()
    got = cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    assert set(got) <= set(cfbd.WEEKLY)
    assert all(isinstance(v, pl.DataFrame) for v in got.values())


def test_week_prints_a_summary_not_rows(transport, paths, capsys):
    transport()
    cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    assert len(capsys.readouterr().out.splitlines()) <= 12
