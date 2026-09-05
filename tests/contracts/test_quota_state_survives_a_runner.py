"""Two guards that cannot fire, and a fetch that reports success without fetching.

`hub.fetch.odds` refuses a pull when the *stored* balance is below a floor, and
`hub.fetch.cfbd` counts calls against a monthly free tier. Both persisted their state under
`data/raw/`, which `.gitignore` excludes -- so every scheduled run started with no record of
what had been spent, the odds floor could never refuse, and the CFBD counter reset to zero
each month's worth of runs. A guard that cannot fire is the same shape as the contract that
was declared and enforced nowhere: it reads as coverage.

The cadence made it survivable rather than safe -- two pulls a week against 500 credits, and
the balance was 497 the morning this was found. It stops being survivable the moment a
cadence changes or a run loops, which is precisely what the floor exists to catch.

And `make slate` never fetched college football at all. The Makefile passes `--week` only
when `WEEK` is set in the environment and the scheduled run sets none, so the CLI took its
no-week branch, printed a healthy-looking quota report and exited 0. Every scheduled run has
reported success for a fetch that did not happen.

**This file was itself the hazard (#68).** It drives the CFBD CLI, and it did so with neither
the environment reader nor the response cache patched. That was harmless for exactly as long
as the CLI had no week to resolve; the moment it began counting one from `CFB_WEEK_ONE`, a
developer holding a key was making three live calls on every run of the suite. Measured
2026-09-05: this machine's `.env` carries neither the anchor nor a key, so nothing was spent
that day -- but `.env.example` instructs a developer to set the anchor and the key is already
a repository secret, so what stood between the suite and the account was which machine ran
it. The fixture below patches what the sibling unit module patches, and the guarantee sits
one level under it, in `cfbd._http_get`: see `test_an_anchor_and_a_key_still_spend_nothing`.
"""
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hub.fetch import cfbd, odds

ROOT = Path(__file__).resolve().parents[2]

# The module's real environment reader, kept before anything patches it -- the armed test
# below needs the reader that folds `.env` in, because a refusal that only holds when the
# reader is a stub is the conditional guard #68 was filed about.
_REAL_ENV = cfbd._env


@pytest.fixture(autouse=True)
def nothing_real_is_written(tmp_path, monkeypatch):
    """The default write paths and the environment, pointed away from the real ones.

    Deliberately not `QUOTA`: `test_the_cfbd_counter_is_kept_where_a_runner_can_read_it`
    asserts where that file lives, and a patched path would make it assert about `tmp_path`.
    Every test here that drives `main` passes `--quota-path` instead.

    This is hygiene, not the guard. A test added tomorrow that forgets the fixture still
    cannot fetch, because `cfbd._http_get` refuses under pytest.
    """
    monkeypatch.setattr(cfbd, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cfbd, "STATUS", tmp_path / "site" / "cfbd.json")
    monkeypatch.setattr(cfbd, "_env", dict)


def _ignored(rel: str) -> bool:
    """Whether git would refuse to track this path. `check-ignore` exits 1 when not."""
    return subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel],
                          capture_output=True).returncode == 0


# --- the state a guard needs has to reach the next run -----------------------

def test_the_odds_balance_is_kept_where_a_runner_can_read_it():
    """`data/raw/` is gitignored as redistributed third-party data, and a credit balance is
    neither raw nor third-party -- it is our own bookkeeping about our own account. Keeping
    it there made the floor inert on every runner, which is the whole guard."""
    rel = odds.STATE.relative_to(ROOT).as_posix()
    assert not _ignored(rel), f"{rel} is gitignored, so a scheduled run can never see it"
    assert not rel.startswith("data/"), (
        f"{rel} is under data/, whose whole tree is excluded as redistributed payloads")


def test_the_cfbd_counter_is_kept_where_a_runner_can_read_it():
    rel = cfbd.QUOTA.relative_to(ROOT).as_posix()
    assert not _ignored(rel), f"{rel} is gitignored, so a scheduled run can never see it"
    assert not rel.startswith("data/")


def test_the_state_directory_carries_no_third_party_payload():
    """The reason `data/raw/` is excluded still applies to whatever replaces it. Counters
    and balances are ours; a cached response is not."""
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "state"],
                             capture_output=True, text=True).stdout.split()
    assert tracked, "state/ is not tracked, so a runner still starts with no record"
    records = [f for f in tracked if f.endswith(".json")]
    assert set(tracked) - set(records) <= {"state/README.md"}, (
        f"unexpected in state/: {sorted(set(tracked) - set(records))}")
    for f in records:
        body = json.loads((ROOT / f).read_text())
        assert isinstance(body, dict), f"{f} is not a small bookkeeping record"
        assert len(json.dumps(body)) < 4096, f"{f} is too large to be a counter"


def test_the_slate_commits_the_state_it_spent():
    """Reading it on a runner is half of it. A run that spends a credit and does not commit
    the new balance leaves the next run reading the same stale number."""
    wf = (ROOT / ".github" / "workflows" / "slate.yml").read_text()
    add = [ln for ln in wf.splitlines() if ln.strip().startswith("git add")]
    assert add, "the slate no longer commits anything"
    assert any("state" in ln for ln in add), (
        f"the slate commits {add} and not the quota state, so every run starts from the "
        f"balance that was last committed by hand")


# --- the guards themselves, exercised against a store the run can see --------

def test_a_stored_balance_below_the_floor_refuses_the_next_pull(tmp_path):
    """The property the gitignore quietly removed: the refusal depends on state surviving."""
    state = tmp_path / "odds.json"
    state.write_text(json.dumps({"remaining": 10, "checked_at": "2026-09-04T00:00:00"}))
    assert odds.credits_remaining(state) == 10
    try:
        odds.snapshot(season=2026, state_path=state, floor=50)
    except odds.QuotaFloor as e:
        assert "floor" in str(e)
    else:                                                    # pragma: no cover
        raise AssertionError("a balance under the floor must refuse before spending")


def test_the_cfbd_counter_carries_across_runs(tmp_path):
    q = tmp_path / "cfbd-quota.json"
    for _ in range(3):
        cfbd._record_call(q)
    assert cfbd.quota_used(q) == 3, "each run must add to what the last one recorded"


# --- a quota report is not a fetch -------------------------------------------

def test_asking_cfbd_for_no_week_does_not_report_success(capsys, tmp_path):
    """`make slate` runs `hub.fetch.cfbd` with no `--week` unless `WEEK` is set, and the
    scheduled run sets none. Exiting 0 after printing a quota report is how every run has
    reported success for a fetch that did not happen."""
    # `--status-path` as well as `--quota-path`: without it the CLI records its run into
    # the real `site/data/`, so running the suite left an untracked artifact in the working
    # tree. A test that writes into the repo it is testing is one that eventually gets
    # committed by accident.
    #
    # The empty environment comes from the autouse fixture. Without it this test read one
    # way on a machine with `CFB_WEEK_ONE` set and another way on a machine without -- and
    # on the first kind it was not testing the no-week branch at all, it was fetching.
    code = cfbd.main(["--quota-path", "/dev/null",
                      "--status-path", str(tmp_path / "cfbd.json")])
    assert code != 0, "no week means nothing was fetched, and the exit code has to say so"
    assert "nothing was fetched" in capsys.readouterr().err.lower()


def test_an_anchor_and_a_key_still_spend_nothing(tmp_path, monkeypatch, capsys):
    """#68's acceptance criterion, run rather than reasoned about.

    The environment here is the one the setup instructions produce: `.env.example` tells a
    developer to set `CFB_WEEK_ONE`, and `CFBD_API_KEY` is already a repository secret. So
    this test arms both, restores the *real* environment reader -- `.env` folded in and all
    -- and drives the CLI down the path that fetches. Nothing reaches CFBD, because the
    refusal is at the transport rather than in a fixture.

    Verified without spending anything: `LiveCallRefused` is raised above `import requests`
    in `_http_get`, so a run that records it is a run in which nothing left the process.
    """
    monkeypatch.setattr(cfbd, "_env", _REAL_ENV)
    monkeypatch.setenv(cfbd.CFB_WEEK_ONE_ENV,
                       (datetime.now(UTC) - timedelta(days=7)).date().isoformat())
    monkeypatch.setenv("CFBD_API_KEY", "not-a-real-key")

    before = cfbd.QUOTA.read_bytes() if cfbd.QUOTA.exists() else None
    status = tmp_path / "cfbd.json"
    code = cfbd.main(["--quota-path", str(tmp_path / "quota.json"),
                      "--status-path", str(status)])

    got = json.loads(status.read_text())
    assert got["week"] == 2, (
        "the anchor resolved a week and the key was there, so this run did reach the point "
        "of fetching -- which is what makes the refusal below worth asserting")
    assert code == 1 and got["fetched"] is False
    assert cfbd.LiveCallRefused.__name__ in got["reason"], (
        f"the run recorded {got['reason']!r}, which is not the transport refusing")
    assert "would spend a live CFBD call" in capsys.readouterr().err
    # The month's real counter, untouched: every path here writes to `tmp_path`. The refused
    # call is still counted against the *temporary* one, because `bulk` counts in `finally`
    # and erring high is the direction that module argues for.
    assert (cfbd.QUOTA.read_bytes() if cfbd.QUOTA.exists() else None) == before


def test_asking_cfbd_for_the_quota_deliberately_is_still_success(capsys):
    """`make check` exists to print exactly this. Reporting the balance is a legitimate
    thing to ask for; it is only a lie when it stands in for a fetch."""
    assert cfbd.main(["--quota", "--quota-path", "/dev/null"]) == 0
    assert "quota" in capsys.readouterr().out.lower()
