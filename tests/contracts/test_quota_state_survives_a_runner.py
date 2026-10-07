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


# The width ledger is not a counter: since #362 it is append-only, one entry per gate run, so a
# 4,096-byte cap that a counter honours is a cap it crosses on schedule -- it crossed it on
# 2026-10-06, when #343's routed harnesses wrote their first four entries (the file held 2,997
# bytes). It is still *our own* small bookkeeping and still no payload: an entry is a name, two
# digests, an interval and a per-season record. **Sized from the measured entry** (#426, again
# at #434, restated at #439). *Measured* on `state/gate-width.json` as it is on disk (19 rows,
# 23,283 bytes; each row counted as written, indented two levels): the 16 rows without inputs
# average 977 bytes and reach 1,339 (the longest carries five per-season records); the 3 rows
# that carry `inputs` inline (#429) are 1,147, 1,172 and 5,317 bytes, 7,636 between them
# (over all 19 rows the average is 1,224) -- the 5,317 is the draft gate's 36 reads, the
# reason the list is stored once in `state/inputs/` and a new row carries a digest. *Projected*
# (not measured): the two fields since added, `code_digest` (#435) and `inputs_digest` (#434),
# add about 65 bytes a row, so a budget of 1,400 bytes an entry is the measured maximum plus
# those, rounded up. 384 KiB (393,216 bytes) less the 7,636 legacy bytes (append-only, not
# rewritten) is about 275 such entries -- "a few hundred", the size at which someone should
# decide what to do with the history. (The earlier 256 KiB was sized at 800 bytes an entry on
# the compact-JSON measure below, which is not the file's own size: the file is indented, so
# its rows are larger than that arithmetic said.) A counter's cap stays as it was, and this is
# the only file with its own.
#
# The coverage measurement has its own budget since #309. It is a *summary of one run*, rewritten
# in place (not appended), and it grew from 3.2 KB when every group row began carrying its
# `position`, `n`, `n_cal`, deviation, sigma and the named cells that fell back to pooled
# calibration, beside the parametric table it replaced (the prior values, kept in the file).
# **Sized from the measured file, re-measured in #438** (the first figure here, 8,544, was taken
# before the rows gained their verdicts and was stale on arrival): 8,935 bytes serialised the way
# this test measures them, and **10,899 with all three audit looks recorded** -- the file's whole
# life under #310's rule, an `audit` block plus about 145 bytes a look in `audit_looks`. 16 KiB is
# 1.5 times that end-of-life size: room for a sixth season's rows or a longer window, not room
# for a payload. Still no third party's data in it -- coverage rates of this repo's own interval.
LEDGER_CAPS = {"state/gate-width.json": 393216, "state/interval_coverage.json": 16384}


def test_the_coverage_artifacts_cap_and_its_comment_agree_with_the_file():
    """#438: the comment above sized the cap from a figure that was stale on arrival, and
    nothing noticed. The cap must sit within a factor of 2.5 of the file as measured the way
    `test_the_state_directory_carries_no_third_party_payload` measures it -- 1.5x the file at its
    three-look end of life is the intent -- so a cap left 10x too slack, or a file that has
    outgrown its reasoning, fails here rather than in a comment.

    Mutation (observed): `"state/interval_coverage.json": 163840` is red."""
    cap = LEDGER_CAPS["state/interval_coverage.json"]
    size = len(json.dumps(json.loads((ROOT / "state" / "interval_coverage.json").read_text())))
    assert 1.0 < cap / size <= 2.5, f"cap {cap} against a file of {size}: resize it and its comment"


# #432: one capture of the weekly consensus page, a few hundred ranked players, is a record of
# what could have been read before a week's first game and is the only thing that can ever say
# so. Bigger than a counter by design; still one small JSON document a file.
CAPTURE_CAP = 131072

# #434: the content-addressed input sets under `state/inputs/`, one file per distinct set of
# pins a gate run read (identical sets share a file). A file is a small dict of
# `{source, as_of, digest}` reads, 2,178 bytes for the draft gate's 36 -- the largest run
# today -- so 8,192 per file is about 3.7x the measured maximum (~130 reads) and a file at the
# cap means a gate reads an order of magnitude more sources than any does now. A new file
# appears only when a pin moves (a refetch, a new season), not on every run; the directory as a
# whole has its own budget, `INPUTS_DIR_BUDGET`, and its own headroom check below.
INPUTS_PREFIX = "state/inputs/"
INPUTS_FILE_CAP = 8192
INPUTS_DIR_BUDGET = 131072
# A contract that fails at the cap tells you after the fact. These fail at this share of it, so
# the decision (roll the ledger by season, prune the sets no row names) is made while the file
# still works.
HEADROOM = 0.8


def test_the_state_directory_carries_no_third_party_payload():
    """The reason `data/raw/` is excluded still applies to whatever replaces it. Counters
    and balances are ours; a cached response is not.

    One adopted exception (#383, restated 2026-10-06/07): `state/odds/` holds the betting
    market's per-game median spread, derived from The Odds API. Their terms permit "storing
    our data and retaining it indefinitely" and "calculating and displaying values you derive
    from our data", and prohibit redistribution "as a standalone data product"; this repo is
    private. What this test holds is the mechanical line: small JSON records, nothing else.

    A second (#432): `state/consensus/` holds one week's FantasyPros `weekly-op` page, as
    redistributed by DynastyProcess, taken before kickoff because a ranking that is not kept
    cannot be shown to have been read in time. That source's terms have **not** been checked the
    way The Odds API's were; the repo is private, and `state/README.md` says to re-examine this
    directory first if it is ever made public."""
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "state"],
                             capture_output=True, text=True).stdout.split()
    assert tracked, "state/ is not tracked, so a runner still starts with no record"
    records = [f for f in tracked if f.endswith(".json")]
    assert set(tracked) - set(records) <= {"state/README.md"}, (
        f"unexpected in state/: {sorted(set(tracked) - set(records))}")
    for f in records:
        body = json.loads((ROOT / f).read_text())
        assert isinstance(body, dict), f"{f} is not a small bookkeeping record"
        stored_set = f.startswith(INPUTS_PREFIX)
        capture = f.startswith("state/consensus/")
        cap = (INPUTS_FILE_CAP if stored_set else CAPTURE_CAP if capture
               else LEDGER_CAPS.get(f, 4096))
        what = ("a stored input set" if stored_set
                else "one week's captured page" if capture
                else "an append-only ledger's budget" if f in LEDGER_CAPS else "a counter")
        assert len(json.dumps(body)) < cap, f"{f} is over {cap} bytes, too large for {what}"


def _headroom_failure(used: int, cap: int, what: str) -> str | None:
    """The message when `used` has passed `HEADROOM` of `cap`, else `None`."""
    if used <= cap * HEADROOM:
        return None
    return (f"{what} is {used:,} bytes, past {HEADROOM:.0%} of its {cap:,}-byte cap: decide what "
            f"to do with the history (roll by season, prune) before the cap test fails")


def test_the_gate_ledger_and_its_stored_inputs_have_headroom_before_their_caps():
    """#434: the cap test above fails *at* the cap, when the decision is already late -- the
    ledger crossed 4,096 bytes on 2026-10-06 and 256 KiB was within 45 entries on 2026-10-07
    before anyone had asked. This fails at 80% of each cap, on the bytes the file occupies on
    disk (indented, which the compact `json.dumps` above undercounts by about a third)."""
    ledger = ROOT / "state" / "gate-width.json"
    problems = [_headroom_failure(ledger.stat().st_size, LEDGER_CAPS["state/gate-width.json"],
                                  "state/gate-width.json")]
    stored = sorted((ROOT / "state" / "inputs").glob("*.json"))
    problems.append(_headroom_failure(sum(f.stat().st_size for f in stored), INPUTS_DIR_BUDGET,
                                      "state/inputs/"))
    problems += [_headroom_failure(f.stat().st_size, INPUTS_FILE_CAP, f"state/inputs/{f.name}")
                 for f in stored]
    assert not [p for p in problems if p], [p for p in problems if p]


def test_the_headroom_check_fires_where_it_says_it_does():
    """Rule 18: a check that never fails reads as headroom. Planted at the boundary -- one byte
    under 80% passes, one over fails -- and against the real caps, so it is the figures in use
    that are exercised, not a fixture's."""
    cap = LEDGER_CAPS["state/gate-width.json"]
    edge = int(cap * HEADROOM)
    assert _headroom_failure(edge, cap, "x") is None
    over = _headroom_failure(edge + 1, cap, "x")
    assert over is not None and "past 80%" in over
    assert _headroom_failure(INPUTS_FILE_CAP, INPUTS_FILE_CAP, "f") is not None


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
        # `now` pinned to the reading's own month: unpinned, the clock reached October on
        # 2026-10-01 and the September reading became a pre-reset one -- unknown by design
        # (`odds.known_balance`), so the pull went to the wire instead of refusing.
        odds.snapshot(season=2026, state_path=state, floor=50, now=datetime(2026, 9, 10))
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
