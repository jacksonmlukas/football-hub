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

A fourth guard was added for #68 and is about this file's own kind: `_http_get` refuses to run
under pytest at all, outside `tests/golden/`. Every test here patches the transport, which is
the habit and not the guarantee -- the guarantee has to hold for the test nobody remembered to
patch, and one in `tests/contracts/` was exactly that.
"""
import json
import os
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
    and must not report success either.

    What the record says about the failure is a separate question, and it is asserted by
    `test_a_failed_fetch_records_the_kind_of_failure_and_not_its_words` below: this one used
    to require `"503" in reason`, which is the exception's own words and is the channel #70
    closed.
    """
    def boom(path, params, key):
        raise RuntimeError("503 Service Unavailable")
    monkeypatch.setattr(cfbd, "_http_get", boom)
    monkeypatch.setattr(cfbd, "_api_key", lambda: "k")
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    got = _record(cfbd.STATUS)
    assert got["fetched"] is False and got["stale"] is True
    assert "RuntimeError" in got["reason"]


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


# --- a test cannot spend a live call --------------------------------------
#
# Issue #68. Every test above patches `_http_get`, which is the right habit and is not a
# guarantee: the guarantee has to hold for the test nobody remembered to patch. One in
# `tests/contracts/` was exactly that -- it drove this CLI with neither the environment
# reader nor the cache patched, which was harmless while the CLI had no week to resolve and
# became three live calls a suite run once it counted one from `CFB_WEEK_ONE`. Measured
# 2026-09-05: this machine's `.env` carries neither the anchor nor a key, so nothing was
# spent -- but `.env.example` instructs a developer to set the anchor and the key is already
# a repository secret, so the thing standing between the suite and the account was which
# machine it ran on.
#
# So the refusal lives at the transport, where it holds whatever the environment contains.


@pytest.fixture
def no_network(monkeypatch):
    """`requests.get` replaced by something that fails the test rather than dialling out.

    Present so that these tests still prove something when the guard they are about is
    deleted: without it, the excision harness in `tests/contracts/test_guards_are_load_
    bearing.py` would prove the guard fires by making a real request to CFBD.
    """
    import requests

    def _never(*a, **k):                                     # pragma: no cover - must not run
        raise AssertionError("a test reached the network")

    monkeypatch.setattr(requests, "get", _never)


def test_the_transport_refuses_a_live_call_from_the_default_suite(no_network):
    """The whole of #68, at the one function in this module that touches the network."""
    with pytest.raises(cfbd.LiveCallRefused):
        cfbd._http_get("/games", {"year": 2026, "week": 1}, "a-key")


def test_the_refusal_names_the_test_and_the_rule_it_protects(no_network):
    with pytest.raises(cfbd.LiveCallRefused) as e:
        cfbd._http_get("/games", {"year": 2026}, "a-key")
    said = str(e.value)
    assert "quota" in said.lower(), "a refusal that does not say why teaches nothing"
    assert "test_the_refusal_names_the_test" in said, (
        "the refusal has to name the test that tripped it, or finding it means bisecting")


def test_the_golden_suite_is_the_one_place_a_live_call_is_still_allowed(monkeypatch):
    """`tests/golden/` exists to diff a live response against the frozen fixture, is marked
    `golden`, and is deselected by default (`addopts = "-m 'not golden'"`). Refusing there
    too would turn the only check that meets reality into one that cannot run."""
    sent = []

    class _Response:
        def raise_for_status(self): return None
        def json(self): return [_GAME]

    def _capture(url, **kw):
        sent.append(url)
        return _Response()

    import requests
    monkeypatch.setattr(requests, "get", _capture)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/golden/test_golden.py::t (call)")
    assert cfbd._http_get("/games", {"year": 2025, "week": 1}, "k") == [_GAME]
    assert sent == [f"{cfbd.BASE}/games"]


def test_a_pytest_node_id_is_the_path_the_exemption_matches_on():
    """The premise under `LIVE_TEST_SUITE`. `PYTEST_CURRENT_TEST` is a rootdir-relative node
    id, which is what makes `tests/golden/` a prefix of every node in that suite and of
    nothing else. If a runner ever changes what rootdir is, this says so here -- rather than
    by silently refusing the one suite that is supposed to reach CFBD."""
    assert os.environ[cfbd.PYTEST_NODE_ENV].startswith(
        "tests/unit/test_fetch_cfbd.py::test_a_pytest_node_id_is")


def test_the_network_has_exactly_one_door_in_this_module():
    """What makes the refusal above structural rather than one more patched seam: there is
    one function that can reach CFBD, so guarding it guards everything."""
    import inspect
    src = inspect.getsource(cfbd)
    users = [ln.strip() for ln in src.splitlines()
             if "requests" in ln and not ln.strip().startswith("#")]
    assert users == ["import requests",
                     "r = requests.get(f\"{BASE}{path}\", params=dict(params), timeout=30,"], (
        f"something other than `_http_get` reaches the network now: {users}. The refusal in "
        f"`_http_get` only covers this module while that stays the only door.")


# --- the record carries counts, and the contract still fires on an empty week ---
#
# Issue #70, two defects at the same boundary: what a third party said, and what this repo
# commits about it.


def test_the_status_file_never_quotes_the_values_that_broke_a_contract(env, transport, paths,
                                                                      capsys):
    """`record_run` writes into `site/data/cfbd.json`, which the slate commits to a repo
    intended to go public, and the failure path used to write `str(e)` into it. A contract
    violation quotes the values that broke it. Measured 2026-09-05, the planted frame below
    produces exactly

        cfbd_games: week range [99, 99] outside [1, 20]; homePoints range [131, 131]
        outside [0, 120]

    which is four payload values and a column name. Truncating at 400 characters bounds how
    much of a response escapes; it does not turn rows into counts, and the comment above
    `STATUS` claims counts twice.
    """
    env[cfbd.CFB_WEEK_ONE_ENV] = "2026-09-01"
    transport(payload=[{**_GAME, "week": 99, "homePoints": 131, "awayPoints": 0}])
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1

    text = cfbd.STATUS.read_text()
    for leaked in ("homePoints", "outside", "Stanford", "cfbd_games", "range"):
        assert leaked not in text, f"{leaked!r} came out of the payload and is now committed"
    reason = _record(cfbd.STATUS)["reason"]
    assert "99" not in reason and "131" not in reason, f"payload values in {reason!r}"
    assert "ContractViolation" in reason, (
        "a reader of the status file still has to be able to tell why the fetch failed")
    # Not lost, only not committed: stderr is where the operator reads it.
    assert "week range [99, 99]" in capsys.readouterr().err


def test_a_failed_fetch_records_the_kind_of_failure_and_not_its_words(env, paths, monkeypatch,
                                                                      capsys):
    """The rule the status file can actually keep: the exception's type, never its message.

    A type name is ours -- it comes out of this repo's code or its dependencies' -- while a
    message is an open channel with nothing bounding what a third party can put through it.
    """
    def boom(path, params, key):
        raise RuntimeError("503 Service Unavailable for https://api.collegefootballdata.com")
    monkeypatch.setattr(cfbd, "_http_get", boom)
    monkeypatch.setattr(cfbd, "_api_key", lambda: "k")
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    got = _record(cfbd.STATUS)
    assert got["fetched"] is False
    assert "RuntimeError" in got["reason"]
    assert "Service Unavailable" not in got["reason"] and "api.college" not in got["reason"]
    assert "Service Unavailable" in capsys.readouterr().err


def test_an_http_status_is_a_number_so_it_survives_the_redaction(env, paths, monkeypatch):
    """The diagnostic worth keeping. A status code is a count-shaped fact about the exchange
    -- three digits from the protocol, not a field of anybody's payload -- and it is the
    difference between "the source is down" and "the key is wrong" without opening a log."""
    class _Response:
        status_code = 503

    class _HTTPError(RuntimeError):
        response = _Response()

    def boom(path, params, key):
        raise _HTTPError("503 Server Error: Service Unavailable for url: ...")
    monkeypatch.setattr(cfbd, "_http_get", boom)
    monkeypatch.setattr(cfbd, "_api_key", lambda: "k")
    assert cfbd.main(["--week", "3", "--quota-path", str(paths["quota"])]) == 1
    assert "HTTP 503" in _record(cfbd.STATUS)["reason"]


def test_an_empty_response_carrying_a_shape_is_a_shape_change_not_an_empty_week(transport,
                                                                               paths):
    """The half of #70 that had gone quiet. Validation became conditional on the frame having
    rows, so a response of no rows was not checked at all -- and both CFBD contracts declare
    `min_rows=1`, which made that skip a hole in exactly the pair the provenance work
    constrains.

    An empty week and an empty *shape* are not the same thing. `[]` is the source saying
    "nothing here" and carries no columns to check. A reshaped endpoint answering
    `{"data": []}` is also zero rows -- and is a rename wearing an empty week's clothes.
    """
    from hub.contracts import ContractViolation

    transport(payload={"data": []})
    with pytest.raises(ContractViolation):
        cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])


def test_a_week_with_rows_outside_the_declared_range_is_still_a_violation(transport, paths):
    """The ordinary case, asserted because nothing else here did: the check that fires on a
    populated week has to keep firing while the empty one is being taught to."""
    from hub.contracts import ContractViolation

    transport(payload=[{**_GAME, "week": 99}])
    with pytest.raises(ContractViolation):
        cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])


# --- the cache says when it was captured, and can be refreshed -----------------
#
# #175. The cache was permanent by file existence: if the path was there it was returned,
# with no maximum age, no refresh parameter and nothing recording when the bytes arrived. On
# a metered source that is the expensive half rather than the cheap one -- it is both the
# reason to cache and the reason a stale price could never be corrected.


def _seed(transport, paths, payload=None):
    """One cached week, fetched once through the patched transport."""
    calls = transport(payload)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"])
    return calls


def _entry(paths, endpoint="games", year=2026, week=1):
    return cfbd._cache_path(endpoint, year, week, paths["cache"])


def _forget_the_capture(paths):
    """An entry as it was written before #175: a payload with nothing beside it."""
    cfbd._capture_path(_entry(paths)).unlink()


def _age(paths, hours):
    """Backdate the capture record beside one cache entry."""
    when = datetime.now(UTC) - timedelta(hours=hours)
    cfbd._capture_path(_entry(paths)).write_text(
        json.dumps({"captured_at": when.isoformat()}))
    return when


def _spend_the_month(paths):
    paths["quota"].write_text(
        json.dumps({cfbd._month_key(): cfbd.FREE_TIER_MONTHLY}))


def test_a_fetched_payload_records_when_it_was_captured(transport, paths):
    _seed(transport, paths)
    when = cfbd.captured_at("games", 2026, 1, cache=paths["cache"])
    assert when is not None, "a payload with no capture time is the state #175 is about"
    assert datetime.now(UTC) - when < timedelta(minutes=5)


def test_a_cached_read_is_still_free_and_leaves_the_capture_time_alone(transport, paths):
    """The capture time belongs to the fetch, not to the read. Re-stamping it on every read
    would make a payload from September look like one from this morning."""
    calls = _seed(transport, paths)
    first = cfbd.captured_at("games", 2026, 1, cache=paths["cache"])
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"])
    assert len(calls) == 1
    assert cfbd.captured_at("games", 2026, 1, cache=paths["cache"]) == first


def test_a_forced_refresh_refetches_and_replaces_what_the_cache_holds(transport, paths):
    """Both halves, because either alone passes while the other is broken: the call is made,
    and the payload that comes back is the new one rather than the entry on disk."""
    _seed(transport, paths, [_GAME])
    calls = transport([dict(_GAME, id=2)])
    got = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                    quota_path=paths["quota"], refresh=True)
    assert len(calls) == 2, "the seed call, and this one"
    assert got["id"].to_list() == [2]
    served = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                       quota_path=paths["quota"])
    assert served["id"].to_list() == [2], "the refreshed payload is what the cache now holds"


def test_an_entry_inside_a_stated_age_is_served_without_a_call(transport, paths):
    calls = _seed(transport, paths)
    _age(paths, hours=2)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"],
              max_age=timedelta(hours=6))
    assert len(calls) == 1


def test_an_entry_past_a_stated_age_is_refetched_rather_than_served(transport, paths):
    calls = _seed(transport, paths)
    _age(paths, hours=30)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"],
              max_age=timedelta(hours=6))
    assert len(calls) == 2


def test_no_age_stated_means_the_entry_stands_however_old_it_is(transport, paths):
    """The default every existing caller gets, and the two scheduled runs with it. A default
    bound here would have them re-fetching weeks they already hold, against 1,000 a month."""
    calls = _seed(transport, paths)
    _age(paths, hours=24 * 400)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"])
    assert len(calls) == 1


def test_an_entry_written_before_this_reads_back_unknown_rather_than_invented(transport,
                                                                              paths):
    """#175's fourth criterion. Every entry already on disk has nothing beside it, and the
    answer for those is that nobody knows -- not a number that looks like one."""
    _seed(transport, paths)
    _forget_the_capture(paths)
    assert _entry(paths).exists(), "the payload is still there; only its record is gone"
    assert cfbd.captured_at("games", 2026, 1, cache=paths["cache"]) is None


def test_the_capture_time_is_never_taken_from_the_files_own_mtime(transport, paths):
    """The mtime is a real number about the wrong thing. A clone, a copy, a restore or a
    `touch` rewrites it, so it dates the file and not the fetch -- and it is exactly the
    plausible-looking answer that would make "unknown" quietly disappear."""
    _seed(transport, paths)
    _forget_the_capture(paths)
    os.utime(_entry(paths), None)
    assert cfbd.captured_at("games", 2026, 1, cache=paths["cache"]) is None


def test_an_entry_of_unknown_age_is_still_read_rather_than_erroring(transport, paths):
    calls = _seed(transport, paths)
    _forget_the_capture(paths)
    got = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                    quota_path=paths["quota"])
    assert len(calls) == 1 and got.height == 1


def test_an_unknown_capture_time_cannot_satisfy_a_stated_age(transport, paths):
    """It cannot be shown to be inside the bound, and serving it would answer a question
    about age with a silence that reads as a yes. One call that was not needed is the cheaper
    error than a stale price nothing can correct."""
    calls = _seed(transport, paths)
    _forget_the_capture(paths)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"],
              max_age=timedelta(days=365))
    assert len(calls) == 2


def test_a_refresh_that_would_exceed_the_monthly_budget_serves_the_cache_and_says_so(
        transport, paths, capsys):
    """#175's third criterion. The refusal is the refusal this module always made; what it
    must not do is lose the payload already on disk while making it."""
    calls = _seed(transport, paths)
    _spend_the_month(paths)
    got = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                    quota_path=paths["quota"], refresh=True)
    assert len(calls) == 1, "nothing was spent"
    assert got.height == 1, "and the cached payload came back"
    out = capsys.readouterr().out
    assert "not refreshed" in out and "cached" in out
    assert "budget" in out, "the line says why, not just that"


def test_a_refresh_past_the_run_ceiling_serves_the_cache_and_says_so(transport, paths,
                                                                     monkeypatch, capsys):
    calls = _seed(transport, paths)
    monkeypatch.setattr(cfbd, "_CALLS_THIS_RUN", cfbd.MAX_CALLS_PER_RUN)
    got = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                    quota_path=paths["quota"], refresh=True)
    assert len(calls) == 1 and got.height == 1
    assert "not refreshed" in capsys.readouterr().out


def test_a_refresh_with_no_key_serves_the_cache_rather_than_raising(transport, paths,
                                                                    monkeypatch, capsys):
    """Graceful degradation, and the same branch: a refresh nobody can make must not take
    down a caller that had a perfectly good answer sitting on disk."""
    calls = _seed(transport, paths)
    monkeypatch.setattr(cfbd, "_api_key", lambda: None)
    got = cfbd.bulk("games", year=2026, week=1, cache=paths["cache"],
                    quota_path=paths["quota"], refresh=True)
    assert len(calls) == 1 and got.height == 1
    assert "not refreshed" in capsys.readouterr().out


def test_a_refusal_with_nothing_cached_still_raises(transport, paths):
    """The other half of the same branch. A refusal with no payload behind it has nothing to
    serve, and must not become an empty frame that reads like an answer."""
    transport()
    _spend_the_month(paths)
    with pytest.raises(cfbd.QuotaExceeded):
        cfbd.bulk("games", year=2026, week=2, cache=paths["cache"],
                  quota_path=paths["quota"])


def test_the_served_line_says_when_the_capture_time_is_unknown(transport, paths,
                                                                monkeypatch, capsys):
    """What a refused refresh serves is an entry of *some* age, and the operator reading the
    line has to be told which -- including that nobody knows, which is what every entry
    written before #175 will say."""
    _seed(transport, paths)
    _forget_the_capture(paths)
    monkeypatch.setattr(cfbd, "_CALLS_THIS_RUN", cfbd.MAX_CALLS_PER_RUN)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"],
              refresh=True)
    assert "unknown" in capsys.readouterr().out


def test_the_served_line_names_the_capture_time_when_there_is_one(transport, paths,
                                                                   monkeypatch, capsys):
    _seed(transport, paths)
    when = _age(paths, hours=50)
    monkeypatch.setattr(cfbd, "_CALLS_THIS_RUN", cfbd.MAX_CALLS_PER_RUN)
    cfbd.bulk("games", year=2026, week=1, cache=paths["cache"], quota_path=paths["quota"],
              refresh=True)
    assert when.isoformat() in capsys.readouterr().out


def test_the_weekly_slate_passes_a_refresh_through_to_every_endpoint(transport, paths):
    calls = transport()
    cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"])
    assert len(calls) == 3
    cfbd.week(2026, 3, cache=paths["cache"], quota_path=paths["quota"], refresh=True)
    assert len(calls) == 6, "a refreshed week is three calls again, not zero"


# --- the capture sidecar's own failure modes (issue #175) --------------------
#
# `captured_at` is built on one claim: unknown beats confidently wrong, because unknown means
# "ask again" and a fabricated time means "no need to". Every path that reaches `None` is that
# claim being kept, and the coverage ratchet caught all of them arriving untested.


def test_a_sidecar_that_is_not_an_object_reads_as_unknown(tmp_path):
    """Valid JSON, wrong shape. A list parses and has no capture time in it."""
    entry = tmp_path / "games.json"
    entry.write_text("[]")
    cfbd._capture_path(entry).write_text('["2026-09-07T00:00:00+00:00"]')
    assert cfbd._capture_beside(entry) is None


def test_a_sidecar_whose_timestamp_is_not_a_string_reads_as_unknown(tmp_path):
    """The key is present and the value is a number -- an epoch, plausibly, and not ours."""
    entry = tmp_path / "games.json"
    entry.write_text("[]")
    cfbd._capture_path(entry).write_text(json.dumps({"captured_at": 1757203200}))
    assert cfbd._capture_beside(entry) is None


def test_a_sidecar_whose_timestamp_will_not_parse_reads_as_unknown(tmp_path):
    """A string that is not a timestamp. `fromisoformat` raises and the answer is unknown,
    not today -- the whole point of the field is that it dates the fetch."""
    entry = tmp_path / "games.json"
    entry.write_text("[]")
    cfbd._capture_path(entry).write_text(json.dumps({"captured_at": "last Tuesday"}))
    assert cfbd._capture_beside(entry) is None


def test_a_sidecar_that_cannot_be_written_does_not_take_the_payload_down(tmp_path):
    """A fetched week is worth more than its stamp.

    The call has already been spent against a metered quota by the time the sidecar is
    written, so a failure here must cost the capture time and not the rows. What it leaves
    behind is an entry that reads back unknown, which is exactly the state the reader is
    built for.
    """
    entry = tmp_path / "sub" / "games.json"
    entry.parent.mkdir()
    entry.write_text("[]")
    entry.parent.chmod(0o500)                      # writable no longer
    try:
        cfbd._record_capture(entry)                # must not raise
        assert cfbd._capture_beside(entry) is None
    finally:
        entry.parent.chmod(0o700)                  # so tmp_path can be cleaned up
