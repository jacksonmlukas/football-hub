"""The week-4 instrument check (#389) and its positive controls (rule 18).

Everything here is built from fixtures: a scratch git repository that carries a copy of `src/`
(so the fresh-checkout reader is the clone's own) and hand-written snapshot files, never `data/`
and never the repo's real `state/odds/`. The controls plant the failures the criterion exists to
catch and require it to go red: one capture removed from a committed archive, and one capture
left *untracked* in the working tree -- written by "the runner", never committed, which is the
fault #383 exists to end.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from hub import instrument_check as ic
from hub.ledger import Ledger, WidthEntry

SRC = Path(ic.__file__).resolve().parents[1]
NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
START, UNTIL = datetime(2026, 10, 7, tzinfo=UTC), datetime(2026, 10, 10, tzinfo=UTC)
# Wed 10-07 11:00, Thu 10-08 11:00, Fri 10-09 11:00 -- the first three polls the schedule names.
SLOTS = ("20261007T110004", "20261008T110031", "20261009T110002")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True)


def _snap(repo: Path, week: int, stamp: str) -> Path:
    at = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
    f = repo / "state" / "odds" / "2026" / f"wk{week:02d}" / f"snap-{stamp}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({
        "season": 2026, "week": week, "captured_at": at,
        "rows": [{"game_id": "g1", "close_spread": -3.0, "spread_price": -110.0,
                  "close_total": 44.5, "total_price": -110.0, "captured_at": at,
                  "polls_unmoved": 1, "unmoved_since": at}]}))
    return f


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    """A repository with the real `src/` and a complete archive for the three slots."""
    repo = tmp_path_factory.mktemp("389") / "base"
    repo.mkdir()
    shutil.copytree(SRC, repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for s in SLOTS:
        _snap(repo, 4, s)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "archive")
    return repo


@pytest.fixture
def repo(base, tmp_path) -> Path:
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(base), str(work)], check=True,
                   capture_output=True)
    return work


def _check(repo: Path, **kw) -> ic.Criterion:
    return ic.persistence(repo, 2026, 4, START, UNTIL, now=kw.pop("now", NOW), **kw)


# --- 2. persistence -----------------------------------------------------------------------

def test_a_complete_committed_archive_passes(repo):
    got = _check(repo)
    assert got.status == ic.PASS, got.detail
    assert got.facts["checkable"] == 3 and got.facts["missing"] == []
    assert got.facts["season_rows_read_by_fresh_checkout"] == 3


def test_the_plant_a_removed_capture_turns_it_red(repo):
    """The control the ticket names: a deliberately broken archive with one capture removed.
    A persistence check that passes here is the defect it was written to catch."""
    _git(repo, "rm", "-q", f"state/odds/2026/wk04/snap-{SLOTS[1]}.json")
    _git(repo, "commit", "-q", "-m", "lose one")
    got = _check(repo)
    assert got.status == ic.FAIL
    assert got.facts["missing"] == ["2026-10-08T11:00:00+00:00"]
    assert "10-08 11:00Z" in got.detail


def test_the_plant_a_capture_only_in_the_working_tree_is_not_persisted(repo):
    """Written by the runner and never committed -- present on the machine that wrote it,
    absent from a fresh checkout. The criterion is that second thing."""
    _git(repo, "rm", "-q", f"state/odds/2026/wk04/snap-{SLOTS[2]}.json")
    _git(repo, "commit", "-q", "-m", "lose one")
    _snap(repo, 4, SLOTS[2])                       # back on disk, untracked
    assert (repo / "state/odds/2026/wk04" / f"snap-{SLOTS[2]}.json").exists()
    assert _check(repo).status == ic.FAIL


def test_a_capture_for_another_week_does_not_satisfy_the_week(repo):
    """The poll writes one file per lookahead week; the week asked about is the one read."""
    _git(repo, "rm", "-q", f"state/odds/2026/wk04/snap-{SLOTS[0]}.json")
    _snap(repo, 5, SLOTS[0])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "wrong week")
    assert _check(repo).status == ic.FAIL


def test_a_late_run_inside_the_grace_still_counts_and_one_outside_does_not(repo):
    _git(repo, "rm", "-q", f"state/odds/2026/wk04/snap-{SLOTS[0]}.json")
    _snap(repo, 4, "20261007T130000")              # two hours late: GitHub's measured delay
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "late")
    assert _check(repo).status == ic.PASS
    assert _check(repo, grace=timedelta(minutes=30)).status == ic.FAIL


def test_nothing_checkable_yet_is_not_a_pass(repo):
    """The vacuity control: before any slot's grace has passed there is nothing to have
    persisted, and saying PASS would be the check that cannot fail."""
    got = _check(repo, now=datetime(2026, 10, 7, 12, tzinfo=UTC))
    assert got.status == ic.NOT_YET and got.facts["checkable"] == 0


def test_a_missing_slot_before_the_persistence_date_is_named_as_unrepairable(repo):
    got = ic.persistence(repo, 2026, 4, datetime(2026, 9, 30, tzinfo=UTC),
                         datetime(2026, 10, 2, tzinfo=UTC), now=NOW)
    assert got.status == ic.FAIL
    assert got.facts["missing_before_persistence"] == 2
    assert "cannot be re-captured" in got.detail


def test_the_schedule_expands_to_the_six_polls_in_cron_weekday_order():
    """Sunday is cron 0 and Python 6: the one place this module could be a day off."""
    got = ic.expected_polls(datetime(2026, 10, 5, tzinfo=UTC),
                            datetime(2026, 10, 11, 23, tzinfo=UTC))
    assert [s.strftime("%a %H:%M") for s in got] == [
        "Mon 11:00", "Wed 11:00", "Thu 11:00", "Fri 11:00", "Sat 14:00", "Sun 13:00"]
    assert ic.expected_polls(datetime(2026, 10, 7, 11, 1, tzinfo=UTC),
                             datetime(2026, 10, 7, 12, tzinfo=UTC)) == []


def test_the_backfill_check_sees_a_missing_snapshot(repo, monkeypatch):
    monkeypatch.setattr(ic, "BACKFILLED", SLOTS[:1])
    monkeypatch.setattr(ic, "BACKFILL_WEEKS", 1)
    assert ic.backfill_readable(repo, 2026).status == ic.FAIL     # week 1 holds none
    monkeypatch.setattr(ic, "BACKFILL_WEEKS", 4)
    assert ic.backfill_readable(repo, 2026).status == ic.FAIL     # weeks 1-3 hold none


# --- 1, 3, 4 -------------------------------------------------------------------------------

def _paired(**over) -> pl.DataFrame:
    df = pl.DataFrame({"season": [2025, 2025, 2026, 2026], "roster": [1, 2, 1, 2],
                       "week": [1, 1, 1, 1], "diff": [0.1, 0.2, 0.3, 0.4],
                       "cfg_digest": ["cfg1"] * 4, "data_digest": ["dat1"] * 4})
    return df.with_columns(**{k: pl.lit(v) for k, v in over.items()})


def test_pipeline_passes_on_a_stamped_run_and_fails_each_way_it_can_not_be():
    assert ic.pipeline(_paired(), 2026).status == ic.PASS
    assert ic.pipeline(None, 2026).status == ic.NOT_YET
    assert ic.pipeline(_paired(data_digest=None), 2026).status == ic.FAIL
    assert ic.pipeline(_paired(cfg_digest=""), 2026).status == ic.FAIL
    assert ic.pipeline(_paired().drop("data_digest"), 2026).status == ic.FAIL
    assert ic.pipeline(_paired().filter(pl.col("season") != 2026), 2026).status == ic.FAIL
    assert ic.pipeline(_paired().clear(), 2026).status == ic.FAIL


def test_two_digests_in_one_run_is_not_one_stamp():
    mixed = _paired().with_columns(
        data_digest=pl.Series(["dat1", "dat1", "dat2", "dat2"]))
    assert ic.pipeline(mixed, 2026).status == ic.FAIL


def _entry(cfg="cfg1", data="dat1", width=1.0, ts="2026-10-07T00:00:00", **kw) -> WidthEntry:
    return WidthEntry(name="weekly", config_digest=cfg, data_digest=data, width=width,
                      clusters=5, lo=-1.0, hi=1.0, verdict="SHOW", timestamp=ts, **kw)


def test_the_ledger_criterion_needs_the_run_to_be_in_it(tmp_path):
    path = tmp_path / "gate-width.json"
    led = Ledger(path)
    assert ic.ledger(_paired(), "weekly", path).status == ic.FAIL            # no file at all
    led.record(_entry(cfg="OTHER"))
    assert ic.ledger(_paired(), "weekly", path).status == ic.FAIL, (
        "an entry at another digest pair is not this run being recorded")
    led.record(_entry())
    got = ic.ledger(_paired(), "weekly", path)
    assert got.status == ic.PASS and "nothing was compared" in got.detail


def test_the_ledger_compares_within_the_digest_pair_and_never_across_it(tmp_path):
    """The plant: an earlier entry at *another* pair with a very different width sits between
    the run and its true partner. The partner named must be the same-pair one."""
    path = tmp_path / "gate-width.json"
    led = Ledger(path)
    led.record(_entry(width=1.0, ts="2026-10-01T00:00:00"))
    led.record(_entry(cfg="OTHER", width=9.0, ts="2026-10-02T00:00:00"))
    led.record(_entry(width=0.9, ts="2026-10-03T00:00:00"))
    got = ic.ledger(_paired(), "weekly", path)
    assert got.status == ic.PASS
    assert "2026-10-01T00:00:00" in got.detail, "the partner is the same-pair entry"
    assert "2026-10-02" not in got.detail


def test_the_ledger_on_disk_and_unreadable_is_a_fail_not_a_pass(tmp_path):
    path = tmp_path / "gate-width.json"
    path.write_text("{not json")
    assert ic.ledger(_paired(), "weekly", path).status == ic.FAIL
    assert ic.ledger(None, "weekly", path).status == ic.NOT_YET


def test_k_is_reported_with_rows_per_cluster_and_never_judged():
    got = ic.k_report(_paired(), 2026)
    assert got.status == ic.PASS and got.facts["k"] == 2
    assert "2025: 2 rows x 2 rosters x 1 weeks" in got.detail
    assert ic.k_report(None, 2026).status == ic.NOT_YET


# --- the run, and its exit code ------------------------------------------------------------

@pytest.fixture
def wired(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "BACKFILLED", ())
    monkeypatch.setattr(ic, "WIDTH_STATE", tmp_path / "gate-width.json")
    Ledger(tmp_path / "gate-width.json").record(_entry())
    paired = tmp_path / "paired.parquet"
    _paired().write_parquet(paired)
    args = ["--week", "4", "--from", "2026-10-07T00:00", "--until", "2026-10-10T00:00",
            "--repo", str(repo), "--now", "2026-10-12T12:00"]
    return args, paired


def test_exit_zero_only_when_every_criterion_passes(wired, capsys):
    args, paired = wired
    assert ic.main([*args, "--paired", str(paired)]) == 0
    assert "=> PASS" in capsys.readouterr().out


def test_without_a_paired_run_the_exit_is_not_yet_and_not_zero(wired, capsys):
    args, _ = wired
    assert ic.main(args) == 3
    assert "=> NOT-YET" in capsys.readouterr().out


def test_a_failed_criterion_outranks_a_not_yet(wired, repo):
    args, _ = wired
    _git(repo, "rm", "-q", f"state/odds/2026/wk04/snap-{SLOTS[1]}.json")
    _git(repo, "commit", "-q", "-m", "lose one")
    assert ic.main(args) == 1


def test_the_module_produces_no_verdict_word_and_imports_no_gate():
    """The deliverable says what is not produced. `SHOW`/`ADOPT`/`REMOVE` as output would be
    one; the ledger's recorded verdict is passed through as a fact, never computed here."""
    text = Path(ic.__file__).read_text()
    assert "hub.models.experiment" not in text and "hub.season.weekly_gate" not in text
    assert "import experiment" not in text


@pytest.mark.parametrize("boom", [
    subprocess.TimeoutExpired(["git", "clone"], 120),
    json.JSONDecodeError("Expecting value", "", 0),
], ids=["clone-timeout", "unparseable-reader-output"])
def test_an_unreadable_checkout_is_a_named_refusal_not_a_traceback(wired, monkeypatch, capsys, boom):
    """#433: a stuck clone or a reader that printed non-JSON used to escape `main`'s except
    tuple as a traceback. Each is reported by `hub.cli.unavailable` and exits non-zero."""
    args, _ = wired

    def fail(*_a, **_k):
        raise boom
    monkeypatch.setattr(ic, "read_from_fresh_checkout", fail)
    rc = ic.main(args)
    err = capsys.readouterr().err
    assert rc not in (0, 3) and "hub.instrument_check" in err and "Traceback" not in err


def test_one_clone_serves_both_readings(wired, monkeypatch):
    args, paired = wired
    real, calls = ic.read_from_fresh_checkout, []

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(ic, "read_from_fresh_checkout", counting)
    ic.main([*args, "--paired", str(paired)])
    assert len(calls) == 1


def test_the_clone_has_a_timeout(monkeypatch, repo):
    seen = {}
    real = subprocess.run

    def spy(cmd, *a, **k):
        if cmd[:2] == ["git", "clone"]:
            seen["t"] = k.get("timeout")
        return real(cmd, *a, **k)
    monkeypatch.setattr(ic.subprocess, "run", spy)
    ic.read_from_fresh_checkout(repo, 2026)
    assert seen["t"] == ic.CLONE_TIMEOUT
