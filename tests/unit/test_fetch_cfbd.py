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
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from hub.fetch import cfbd


@pytest.fixture(autouse=True)
def fresh_run():
    """Each test is its own run. Without this the ceiling leaks between tests, which is
    the same defect a long-lived process would hit."""
    cfbd.reset_run_budget()


@pytest.fixture(autouse=True)
def nothing_real_is_written(tmp_path, monkeypatch):
    """The module's three default write paths, pointed at the test's own directory.

    `main` writes a cache, a quota counter and a run record without being told where, which
    is right for a CLI and would otherwise mean a unit test spending a real month's counter
    and dropping a `site/data/cfbd.json` into the working tree.
    """
    monkeypatch.setattr(cfbd, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cfbd, "QUOTA", tmp_path / "state" / "cfbd-quota.json")
    monkeypatch.setattr(cfbd, "STATUS", tmp_path / "site" / "cfbd.json")


@pytest.fixture
def env(monkeypatch):
    """The module's view of the environment, with the machine's `.env` out of the picture.

    `_env()` folds `.env` into `os.environ`, so a developer who has set a real anchor on
    this machine would otherwise change what these tests mean -- and a test that reads
    differently on two machines is not a test.
    """
    values: dict[str, str] = {}
    monkeypatch.setattr(cfbd, "_env", lambda: values)
    return values


@pytest.fixture
def paths(tmp_path):
    return {"cache": tmp_path / "cache", "quota": tmp_path / "quota.json"}


# The shape `contracts.CFBD_GAMES` and `CFBD_LINES` declare. Note what this fixture is and
# is not: both contracts were written from documentation and have never met a live response,
# so a fixture matching them confirms the *plumbing*, not the declaration. The first real
# response is what decides whether the guess was right, which is why both carry
# `verified_against_live=False` and say so when they fail.
_GAME = {"id": 1, "season": 2026, "week": 1, "homeTeam": "Cal", "awayTeam": "Stanford"}


@pytest.fixture
def transport(monkeypatch):
    """Records every call the module would have made."""
    calls = []

    def _install(payload=None):
        def _fake(path, params, key):
            calls.append((path, dict(params)))
            return payload if payload is not None else [_GAME]
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


def test_a_failed_call_still_spends_the_quota_it_spent(tmp_path, monkeypatch):
    """The counters used to be incremented after `_http_get` returned, so a 429 or a 500 --
    which `raise_for_status` turns into an exception -- spent a real call that neither the
    monthly budget nor the run ceiling ever saw."""
    import hub.fetch.cfbd as C

    C.reset_run_budget()
    q = tmp_path / "quota.json"

    def boom(path, params, key):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(C, "_http_get", boom)
    monkeypatch.setattr(C, "_api_key", lambda: "k")
    for _ in range(3):
        with pytest.raises(RuntimeError):
            C.bulk("lines", 2026, 1, cache=tmp_path / "cache", quota_path=q)

    assert C.quota_used(q) == 3, "a spent call must be counted whether or not it answered"
    assert C._CALLS_THIS_RUN == 3


def test_the_run_ceiling_stops_a_loop_of_failing_calls(tmp_path, monkeypatch):
    """The ceiling exists to catch a per-team loop. A loop that errors every time is still
    a loop, and used to run forever because nothing counted it."""
    import hub.fetch.cfbd as C

    C.reset_run_budget()

    def boom(path, params, key):
        raise RuntimeError("500")

    monkeypatch.setattr(C, "_http_get", boom)
    monkeypatch.setattr(C, "_api_key", lambda: "k")
    attempts = 0
    for i in range(C.MAX_CALLS_PER_RUN + 5):
        try:
            C.bulk("lines", 2026, i + 1, cache=tmp_path / "cache",
                   quota_path=tmp_path / "q.json")
        except C.QuotaExceeded:
            break
        except RuntimeError:
            attempts += 1
    assert attempts == C.MAX_CALLS_PER_RUN, "the ceiling must fire on failing calls too"


# --- where the week comes from --------------------------------------------
#
# The decision issue #56 asks for, asserted rather than described. A college week is not an
# NFL week and cannot be read from the prediction store the way `publish.default_week` reads
# one; a bare week number pinned in configuration is wrong every week after the one it was
# set for, and wrong *silently* -- it would fetch week 3 in November and cache it as though
# that were this week. So the configured fact is the season's start date, which stays true
# from August to January, and the week is counted from it.


def test_an_unset_anchor_is_not_a_guess(env):
    """No anchor, no fetch. The alternative is inventing a week and spending quota on it."""
    got = cfbd.configured_week()
    assert got.week is None
    assert cfbd.CFB_WEEK_ONE_ENV in got.why


def test_the_week_is_counted_from_the_configured_start(env):
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    for day, expected in [("2026-09-01", 1), ("2026-09-05", 1), ("2026-09-08", 2),
                          ("2026-09-09", 2), ("2026-10-06", 6)]:
        got = cfbd.configured_week(now=datetime.fromisoformat(f"{day}T12:00:00+00:00"))
        assert got.week == expected, f"{day} should be week {expected}, got {got.week}"


def test_a_mid_week_start_date_still_counts_from_that_whole_week(env):
    """Week 1's first game is usually a Thursday, and a college week opens on the Tuesday
    before it. Counting plain seven-day blocks from the Thursday would put the Wednesday
    refresh -- 11:00 UTC, the slate's own cron -- one week behind from the second week on:
    it would fetch the week that has just finished instead of the one it is refreshing for.
    """
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-03"                # a Thursday
    for day, expected in [("2026-09-03", 1), ("2026-09-05", 1), ("2026-09-09", 2),
                          ("2026-09-16", 3)]:
        got = cfbd.configured_week(now=datetime.fromisoformat(f"{day}T11:00:00+00:00"))
        assert got.week == expected, f"{day} should be week {expected}, got {got.week}"


def test_before_the_first_game_there_is_no_week_to_fetch(env):
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    got = cfbd.configured_week(now=datetime.fromisoformat("2026-08-20T12:00:00+00:00"))
    assert got.week is None and "not started" in got.why


def test_after_the_regular_season_there_is_no_week_to_fetch(env):
    """Past week 15 the college season is postseason, which is a `seasonType` and not a
    week number -- and `contracts.CFBD_GAMES` only declares weeks 1-20 at all."""
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    got = cfbd.configured_week(now=datetime.fromisoformat("2027-01-05T12:00:00+00:00"))
    assert got.week is None
    assert str(cfbd.REGULAR_SEASON_WEEKS) in got.why


def test_an_unusable_anchor_is_refused_rather_than_guessed_around(env):
    env[cfbd.CFB_WEEK_ONE_ENV] = "week one"
    got = cfbd.configured_week()
    assert got.week is None and "week one" in got.why


# --- the record a scheduled run leaves ------------------------------------


def _record(path):
    return json.loads(path.read_text())


def test_a_run_with_no_week_records_that_it_fetched_nothing(env, transport, paths, capsys):
    """The defect #56 was filed for. `make slate` marks this source optional with a leading
    `-`, so the exit code is swallowed and the stderr line goes into a log nobody reads."""
    calls = transport()
    assert cfbd.main(["--quota-path", str(paths["quota"])]) == 1
    got = _record(cfbd.STATUS)
    assert got["fetched"] is False
    assert got["stale"] is True
    assert cfbd.CFB_WEEK_ONE_ENV in got["reason"]
    assert got["week"] is None
    assert calls == [], "a run that fetched nothing must not have spent a call"
    assert cfbd.quota_used(paths["quota"]) == 0


def test_a_fetched_week_is_recorded_as_fetched(env, transport, paths):
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    transport()
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 0
    got = _record(cfbd.STATUS)
    assert got["fetched"] is True
    assert got["stale"] is False
    assert got["reason"] is None
    assert got["week"] == 3
    assert got["rows_by_endpoint"]["games"] == 1


def test_fetched_and_empty_is_not_the_same_record_as_never_fetched(env, transport, paths):
    """The distinction #27 asked for and `publish.Kept` makes for every other producer: a
    source that answered and had nothing says so in its own words."""
    transport(payload=[])
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 0
    empty = _record(cfbd.STATUS)
    assert empty["fetched"] is True, "the week was fetched; it just held nothing"
    assert empty["stale"] is True, "an empty answer is never reported fresh"
    assert "empty" in empty["reason"]
    assert cfbd.CFB_WEEK_ONE_ENV not in (empty["reason"] or "")


def test_the_week_a_scheduled_run_fetches_comes_from_the_anchor(env, transport, paths):
    """No `--week` on the command line, and a week is still fetched -- which is the whole
    of the defect: the Makefile passed one only when a human had set it."""
    env[cfbd.CFB_WEEK_ONE_ENV] = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
    calls = transport()
    assert cfbd.main(["--quota-path", str(paths["quota"])]) == 0
    assert _record(cfbd.STATUS)["week"] == 2
    assert {params["week"] for _, params in calls} == {2}


def test_a_missing_key_records_not_fetched_rather_than_crashing(env, paths, monkeypatch):
    monkeypatch.setattr(cfbd, "_api_key", lambda: None)
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    got = _record(cfbd.STATUS)
    assert got["fetched"] is False and "CFBD_API_KEY" in got["reason"]


def test_a_failed_fetch_records_the_failure_instead_of_taking_the_slate_down(
        env, paths, monkeypatch):
    """CLAUDE.md's degradation rule. An unreachable optional source must not halt a Sunday,
    and must not report success either."""
    def boom(path, params, key):
        raise RuntimeError("503 Service Unavailable")
    monkeypatch.setattr(cfbd, "_http_get", boom)
    monkeypatch.setattr(cfbd, "_api_key", lambda: "k")
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    got = _record(cfbd.STATUS)
    assert got["fetched"] is False
    assert "503" in got["reason"] and "RuntimeError" in got["reason"]


def test_the_record_carries_counts_and_never_a_cfbd_payload(env, transport, paths):
    """`docs/cfbd-quota.md`: redistributing CFBD data is a terms violation, and the slate
    workflow commits everything under `site/data`. Counts are ours; rows are theirs."""
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    transport()
    cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])])
    text = cfbd.STATUS.read_text()
    assert "Stanford" not in text and "homeTeam" not in text
    assert _record(cfbd.STATUS)["quota"]["limit"] == cfbd.FREE_TIER_MONTHLY


def test_the_record_says_which_season_and_when(env, transport, paths):
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    transport()
    cfbd.main(["--week", "3", "--year", "2026", "--quota-path", str(paths["quota"])])
    got = _record(cfbd.STATUS)
    assert got["season"] == 2026
    assert got["name"] == "cfbd" and got["source"] == "hub.fetch.cfbd"
    assert got["generated_at"].startswith("20")


def test_an_empty_week_is_an_answer_rather_than_a_broken_contract(transport, paths):
    """`CFBD_GAMES` declares `min_rows=1`, so validating an empty frame reports missing
    columns -- which reads as the source having changed shape when it has said 'nothing
    here'. Emptiness is the caller's state to report, not the contract's failure."""
    transport(payload=[])
    got = cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    assert all(df.height == 0 for df in got.values())
