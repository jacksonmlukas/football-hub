"""`hub.forecasters.cli`: the commands the forecast workflow runs.

`should-run` is the gate every trigger passes through: it prints `run=` and `day=` lines for
`$GITHUB_OUTPUT`, and says `run=false` outside 09:30-11:30 ET or once the day's run is marked
done. `mark-done` writes that marker after a run completes; a run that crashes leaves no marker,
so the next trigger retries it and the runner's idempotency fills only the gaps.
"""
from __future__ import annotations

from datetime import date

from hub.forecasters import cli


def _gate(tmp_path, now: str) -> dict[str, str]:
    lines: list[str] = []
    rc = cli.main(["should-run", "--out", str(tmp_path), "--now", now], echo=lines.append)
    assert rc == 0
    return dict(line.split("=", 1) for line in lines)


def test_inside_the_window_with_no_marker_runs(tmp_path):
    assert _gate(tmp_path, "2026-10-10T14:00:00+00:00") == {"run": "true", "day": "2026-10-10"}


def test_outside_the_window_does_not_run(tmp_path):
    assert _gate(tmp_path, "2026-11-07T13:30:00+00:00")["run"] == "false"   # 08:30 EST


def test_a_marked_day_does_not_run_again(tmp_path):
    assert cli.main(["mark-done", "--out", str(tmp_path), "--day", "2026-10-10"]) == 0
    assert cli.done_days(tmp_path) == {date(2026, 10, 10)}
    assert _gate(tmp_path, "2026-10-10T14:00:00+00:00")["run"] == "false"


def test_a_marker_for_another_day_does_not_block_today(tmp_path):
    cli.main(["mark-done", "--out", str(tmp_path), "--day", "2026-10-09"])
    assert _gate(tmp_path, "2026-10-10T14:00:00+00:00")["run"] == "true"


def test_a_naive_now_is_refused(tmp_path):
    assert cli.main(["should-run", "--out", str(tmp_path), "--now", "2026-10-10T14:00:00"],
                    echo=lambda _: None) == 2


def test_run_refuses_while_the_config_is_unpinned(tmp_path):
    """The committed config holds v0 placeholders, so `run` must stop before any model call."""
    assert cli.main(["run", "--out", str(tmp_path), "--day", "2026-10-10"],
                    echo=lambda _: None) == 3
