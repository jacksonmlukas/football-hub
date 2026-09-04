"""The watchdog's freshness check, which could not pass.

`watchdog.yml` read `.ts` from the published `live.json`. Only `hub.fetch.espn.poll` writes
that field; `hub.publish.live` -- which is what actually writes the file the page reads --
never has. `jq -r '.ts // 0'` then made the age `now - 0`, so every run reported ~56.7 years
of staleness, filed an incident, and never reached the branch that closes one.

The `// 0` is the defect rather than the field name: one sentinel standing for unreachable,
unreadable and stale, which are three failures with three different fixes. This repo has
already split that pattern twice -- `hub.fetch.odds`' `no_game`/`no_line` counters, and
`hub.publish`'s broad `except` around the schedule fetch.

Tested through the script rather than the workflow, the way `test_guard_hook.py` tests the
hook, and over `file://` so nothing here touches the network.
"""
import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "heartbeat.sh"


def _run(path: Path, threshold: int = 600) -> str:
    got = subprocess.run([str(SCRIPT), f"file://{path}", str(threshold)],
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def _artifact(tmp_path: Path, **fields) -> Path:
    p = tmp_path / "live.json"
    p.write_text(json.dumps(fields))
    return p


def _iso(seconds_ago: int) -> str:
    when = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds_ago)
    return when.replace(microsecond=0).isoformat()


def test_a_fresh_artifact_is_healthy(tmp_path):
    assert _run(_artifact(tmp_path, generated_at=_iso(30))).startswith("ok ")


def test_an_old_artifact_is_stale(tmp_path):
    got = _run(_artifact(tmp_path, generated_at=_iso(4000)))
    assert got.startswith("stale ")
    assert 3900 < int(got.split()[1]) < 4100, "the age is the real one, not now-minus-zero"


def test_the_field_it_reads_is_the_one_publish_actually_writes(tmp_path):
    """The regression. An artifact carrying `ts` and no `generated_at` was what the poller
    wrote, while `generated_at` was what the site writer wrote, and the check could read only
    the first. Both go through one envelope now, and this asserts the check reads that."""
    from hub import jsonio
    shape = jsonio.artifact("live", "espn_scoreboard", [], league="nfl")
    assert "generated_at" in shape and "ts" not in shape
    assert _run(_artifact(tmp_path, **{"generated_at": shape["generated_at"]})
                ).startswith("ok ")


def test_an_artifact_with_no_readable_timestamp_says_so_rather_than_reporting_an_age(
        tmp_path):
    """`now - 0` is not an age, it is a missing field wearing one. Reported apart because
    the fix is different: a stale heartbeat means start the poller, an unreadable one means
    the writer and the monitor disagree about the schema."""
    assert _run(_artifact(tmp_path, ts=1788535055, games=[])) == "unreadable"


def test_a_malformed_timestamp_is_unreadable_not_ancient(tmp_path):
    assert _run(_artifact(tmp_path, generated_at="not a date")) == "unreadable"


def test_a_file_that_is_not_there_is_unreachable(tmp_path):
    assert _run(tmp_path / "absent.json") == "unreachable"


def test_the_threshold_is_the_one_it_is_given(tmp_path):
    p = _artifact(tmp_path, generated_at=_iso(300))
    assert _run(p, threshold=600).startswith("ok ")
    assert _run(p, threshold=60).startswith("stale ")


@pytest.mark.parametrize("bad", ["", "not json", "[]"])
def test_junk_is_never_reported_as_an_age(tmp_path, bad):
    p = tmp_path / "live.json"
    p.write_text(bad)
    assert _run(p) in ("unreadable", "unreachable")
