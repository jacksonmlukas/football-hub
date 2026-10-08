"""#391, the half that makes a silent window visible: a game window in which no watchdog
check ran is reported after the fact.

The replay is 2026-09-28/29, against the real `live.yml` crons, read from the file rather than
copied: PHI @ CHI kicked off at 00:15Z on the Tuesday and was played with no `live` run and no
`watchdog` run, and the first run delivered afterwards was at 06:42Z. Run times and kickoffs
are fixtures; nothing here calls GitHub or ESPN.
"""
import datetime as dt
import json
from pathlib import Path

import pytest

from hub import watchdog_gap as wg

UTC = dt.UTC
LIVE_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "live.yml"
REAL_CRONS = wg.crons_in(LIVE_YML.read_text())


def at(day: int, hh: int, mm: int = 0, month: int = 9) -> dt.datetime:
    return dt.datetime(2026, month, day, hh, mm, tzinfo=UTC)


# The fixture facts of the incident (#391's table and its #390 contrast).
PHI_CHI = at(29, 0, 15)                       # Monday night, 2026-09-28 ET
LAR_DEN = at(28, 0, 20)                       # Sunday night
WATCHDOG_RUNS = [at(28, 4, 57),               # #390's check: inside the Sunday-night window
                 at(28, 11, 37),              # the last run before Monday night's window
                 at(29, 6, 42)]               # the first run after it
REPORTING_RUN = at(29, 6, 42)


def _board(*kicks: dt.datetime):
    return lambda w: list(kicks)


def _replay(kicks, runs=WATCHDOG_RUNS, now=REPORTING_RUN):
    wins = wg.windows(REAL_CRONS, since=at(27, 12), until=now)
    return wins, wg.find_gaps(wins, runs, _board(*kicks))


# --- the windows are the crons, read -------------------------------------------------------

def test_the_scan_finds_the_crons_that_exist():
    """A parser that matched nothing would make every assertion below vacuously true."""
    assert len(REAL_CRONS) == 6


def test_a_sunday_that_runs_past_midnight_is_one_window():
    wins, _ = _replay([])
    sunday = [w for w in wins if w.contains(at(27, 20))]
    assert [(w.start, w.end) for w in sunday] == [(at(27, 16), at(28, 6))], (
        "Sunday daytime and Sunday night are two cron lines and one stretch of games")


def test_monday_night_is_a_window_of_its_own():
    wins, _ = _replay([])
    monday_night = [w for w in wins if w.contains(PHI_CHI)]
    assert [(w.start, w.end) for w in monday_night] == [(at(29, 0), at(29, 6))]


def test_a_window_that_has_not_closed_is_nobodys_gap():
    wins = wg.windows(REAL_CRONS, since=at(28, 12), until=at(29, 3))
    assert wins == [], "00:00-06:00 is still open at 03:00"


def test_a_window_open_at_the_start_of_the_look_back_is_whole_not_clipped():
    wins = wg.windows(REAL_CRONS, since=at(28, 3), until=at(28, 12))
    assert [(w.start, w.end) for w in wins] == [(at(27, 16), at(28, 6))]


def test_a_step_in_a_window_cron_is_refused_rather_than_widened():
    with pytest.raises(ValueError, match="step"):
        wg.windows(["*/10 */2 * * 1"], since=at(27, 0), until=at(30, 0))


def test_a_window_cron_with_a_day_of_month_is_refused():
    with pytest.raises(ValueError, match="days of the week only"):
        wg.windows(["*/10 0-5 1 * 1"], since=at(27, 0), until=at(30, 0))


def test_sunday_written_as_seven_is_sunday():
    sunday7 = wg.windows(["*/10 16-23 * * 7"], since=at(26, 0), until=at(30, 0))
    sunday0 = wg.windows(["*/10 16-23 * * 0"], since=at(26, 0), until=at(30, 0))
    assert sunday7 == sunday0 and sunday7


# --- the replay ----------------------------------------------------------------------------

def test_the_dropped_monday_night_window_is_reported():
    """The incident. No run in 00:00-06:00Z, a kickoff at 00:15Z: reported."""
    _, gaps = _replay([PHI_CHI])
    assert [g.window.id for g in gaps] == ["2026-09-29T00:00Z"]
    assert gaps[0].kickoffs == (PHI_CHI,)


def test_the_run_that_does_the_reporting_is_not_a_check_inside_the_window():
    """06:42Z is after the window closed at 06:00Z. Counting it would make the report
    impossible: the late delivery is always the thing that finds the gap."""
    wins, _ = _replay([PHI_CHI])
    monday_night = next(w for w in wins if w.contains(PHI_CHI))
    assert not monday_night.contains(REPORTING_RUN)


def test_the_sunday_night_window_that_did_have_a_check_is_not_reported():
    """#390's window: the 04:57Z run was inside it, whatever it concluded."""
    _, gaps = _replay([LAR_DEN, PHI_CHI])
    assert all(g.window.id != "2026-09-27T16:00Z" for g in gaps)


def test_a_window_with_a_check_is_never_put_to_espn():
    """The lookup is a request per league per day in production; a week on which the
    scheduler behaved must cost none."""
    asked = []

    def lookup(w):
        asked.append(w.id)
        return [PHI_CHI]
    wins = wg.windows(REAL_CRONS, since=at(27, 12), until=REPORTING_RUN)
    wg.find_gaps(wins, WATCHDOG_RUNS, lookup)
    assert asked == ["2026-09-29T00:00Z"], asked


# --- rule 18: plant the failure, then plant its absence ------------------------------------

def test_planted_no_game_with_no_check_stays_quiet():
    """The same silent window with nothing played in it. A watchdog that reported every
    unchecked window would be reporting the scheduler's weather, not a game."""
    _, gaps = _replay([])
    assert gaps == []


def test_planted_a_kickoff_outside_the_window_does_not_make_it_a_game_window():
    _, gaps = _replay([at(29, 8, 0), at(28, 20, 0)])
    assert gaps == []


def test_planted_a_game_with_a_check_is_quiet():
    _, gaps = _replay([PHI_CHI], runs=[*WATCHDOG_RUNS, at(29, 2, 10)])
    assert gaps == []


def test_a_failed_run_still_counts_as_a_check_that_ran():
    """`check_times` lists runs of any conclusion; a failed job is its own report."""
    _, gaps = _replay([PHI_CHI], runs=[at(29, 5, 59)])
    assert gaps == []


def test_espn_not_answering_is_reported_not_read_as_no_game():
    def down(w):
        raise RuntimeError("ESPN unreachable")
    wins = wg.windows(REAL_CRONS, since=at(28, 12), until=REPORTING_RUN)
    gaps = wg.find_gaps(wins, [], down)
    assert [g.window.id for g in gaps] == ["2026-09-29T00:00Z"]
    assert gaps[0].kickoffs is None
    assert "could not be asked" in wg.render(gaps)


# --- reading ESPN's payload ----------------------------------------------------------------

def test_kickoffs_are_read_from_the_events_date():
    payload = {"events": [{"id": "1", "date": "2026-09-29T00:15Z",
                           "status": {"type": {"name": "STATUS_FINAL"}}},
                          {"id": "2", "date": "2026-09-29T00:20Z"}]}
    assert wg.kickoffs_of(payload) == [at(29, 0, 15), at(29, 0, 20)]


def test_a_postponed_game_was_not_in_progress_in_any_window():
    payload = {"events": [{"id": "1", "date": "2026-09-29T00:15Z",
                           "status": {"type": {"name": "STATUS_POSTPONED"}}},
                          {"id": "2"}]}
    assert wg.kickoffs_of(payload) == []
    assert wg.kickoffs_of({}) == []


def test_espn_kickoffs_asks_both_boards_and_keeps_every_kickoff(monkeypatch):
    from hub.fetch import espn
    asked = []

    def board(league, date=None):
        asked.append((league, date))
        return {"events": [{"id": league + str(date), "date": "2026-09-29T00:15Z"}]}
    monkeypatch.setattr(espn, "scoreboard", board)
    w = wg.Window(at(29, 0), at(29, 6))
    got = wg.espn_kickoffs(w)
    assert {lg for lg, _ in asked} == {"nfl", "cfb"}
    assert {d for _, d in asked} >= {"20260928", "20260929"}, "ET dates around the window"
    assert got and all(k == PHI_CHI for k in got)


# --- the report and its dedupe -------------------------------------------------------------

def test_the_report_names_the_window_and_the_kickoff():
    _, gaps = _replay([PHI_CHI])
    text = wg.render(gaps)
    assert text.startswith(wg.MARKER)
    assert "2026-09-29T00:00Z" in text and "00:15Z" in text


def test_a_gap_the_incident_already_names_is_not_repeated():
    _, gaps = _replay([PHI_CHI])
    assert wg.unseen(gaps, "earlier comment: window 2026-09-29T00:00Z had no check") == []
    assert wg.unseen(gaps, "an unrelated comment") == gaps


# --- the CLI ---------------------------------------------------------------------------------

def _cli(tmp_path, kicks, runs=WATCHDOG_RUNS, known="", extra=()):
    report = tmp_path / "gap.md"
    args = ["--live-yml", str(LIVE_YML), "--repo", "o/r", "--now", REPORTING_RUN.isoformat(),
            "--report", str(report), *extra]
    if known:
        kf = tmp_path / "known.txt"
        kf.write_text(known)
        args += ["--known-file", str(kf)]
    return args, report, _board(*kicks), (lambda workflow, repo: list(runs))


def test_the_cli_reports_the_replayed_window(tmp_path, capsys):
    # The replay is 2026-09-29; a 7-day look-back from it also holds the days before, whose
    # runs this fixture does not carry, so those windows are handed an in-window check.
    earlier = [at(d, h, 30) for d in range(22, 28) for h in (3, 18, 22)]
    args, report, kicks, checks = _cli(tmp_path, [PHI_CHI], runs=WATCHDOG_RUNS + earlier)
    assert wg.main(args, kicks, checks) == 0
    out = capsys.readouterr().out.split()
    assert out == ["gaps=1", "new=1"]
    assert "2026-09-29T00:00Z" in report.read_text()


def test_the_cli_is_quiet_about_a_gap_the_incident_already_holds(tmp_path, capsys):
    earlier = [at(d, h, 30) for d in range(22, 28) for h in (3, 18, 22)]
    args, report, kicks, checks = _cli(
        tmp_path, [PHI_CHI], runs=WATCHDOG_RUNS + earlier,
        known="`2026-09-29T00:00Z` (...) no check ran")
    assert wg.main(args, kicks, checks) == 0
    assert capsys.readouterr().out.split() == ["gaps=1", "new=0"]
    assert not report.exists(), "nothing new to say, so nothing written"


def test_the_cli_with_no_game_reports_nothing(tmp_path, capsys):
    earlier = [at(d, h, 30) for d in range(22, 28) for h in (3, 18, 22)]
    args, report, kicks, checks = _cli(tmp_path, [], runs=WATCHDOG_RUNS + earlier)
    assert wg.main(args, kicks, checks) == 0
    assert capsys.readouterr().out.split() == ["gaps=0", "new=0"]
    assert not report.exists()


def test_the_cli_does_not_look_at_windows_before_the_season(tmp_path, capsys):
    args, _report, kicks, checks = _cli(
        tmp_path, [PHI_CHI], runs=[], extra=["--season-opens", "2026-09-30"])
    assert wg.main(args, kicks, checks) == 0
    assert capsys.readouterr().out.split() == ["gaps=0", "new=0"]


def test_a_run_history_that_cannot_be_read_fails_loudly_rather_than_going_quiet(
        tmp_path, capsys):
    args, _, kicks, _ = _cli(tmp_path, [PHI_CHI])

    def broken(workflow, repo):
        raise OSError("gh: bad credentials")
    assert wg.main(args, kicks, broken) == 2
    assert "could not establish the run history" in capsys.readouterr().err


def test_check_times_reads_gh_json(monkeypatch):
    import subprocess

    class Done:
        stdout = '[{"createdAt": "2026-09-29T06:42:11Z"}]'
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()
    monkeypatch.setattr(subprocess, "run", run)
    assert wg.check_times("watchdog.yml", "o/r") == [dt.datetime(2026, 9, 29, 6, 42, 11,
                                                                 tzinfo=UTC)]
    assert seen["cmd"][:3] == ["gh", "run", "list"] and "watchdog.yml" in seen["cmd"]


# --- #457: a `live` run is a check, and a window with neither is still found -----------------
#
# `live.yml` performs the watchdog check at the end of its loop, so the look-back counts a
# finished `live` run as one. The planted timestamps are the 2026-10-06 window (the second of
# #416's two): one kickoff at 00:15Z, the look-back run delivered at 06:42Z.

OCT6_KICK = dt.datetime(2026, 10, 6, 0, 15, tzinfo=UTC)
OCT6_NOW = dt.datetime(2026, 10, 6, 6, 42, tzinfo=UTC)


def _by_workflow(watchdog=(), live=()):
    def fn(workflow, repo):
        return list({"watchdog.yml": watchdog, "live.yml": live}[workflow])
    return fn


def _lookback(tmp_path, capsys, checks_fn, kicks=(OCT6_KICK,)):
    report = tmp_path / "gap.md"
    args = ["--live-yml", str(LIVE_YML), "--repo", "o/r", "--now", OCT6_NOW.isoformat(),
            "--report", str(report), "--season-opens", "2026-10-06"]
    assert wg.main(args, _board(*kicks), checks_fn) == 0
    return capsys.readouterr().out.split(), report


def test_planted_a_window_with_no_live_run_and_no_watchdog_run_is_still_detected(
        tmp_path, capsys):
    """The Rule-18 control for 'the look-back still catches a window where live did not
    run'. The only evidence anywhere is a live run that ended *before* the window opened."""
    before = dt.datetime(2026, 10, 5, 22, 0, tzinfo=UTC)
    out, report = _lookback(tmp_path, capsys, _by_workflow(live=[before]))
    assert out == ["gaps=1", "new=1"]
    assert "2026-10-06T00:00Z" in report.read_text()


def test_planted_a_live_run_that_ended_inside_the_window_is_a_check(tmp_path, capsys):
    ended = dt.datetime(2026, 10, 6, 2, 5, tzinfo=UTC)
    out, report = _lookback(tmp_path, capsys, _by_workflow(live=[ended]))
    assert out == ["gaps=0", "new=0"]
    assert not report.exists()


def test_a_live_run_that_started_before_the_window_and_ended_in_it_is_a_check(
        tmp_path, capsys):
    """Dated by its end: the check is the loop's last step, so it happened in the window
    even though the run was created before it."""
    out, _ = _lookback(tmp_path, capsys,
                       _by_workflow(live=[dt.datetime(2026, 10, 6, 1, 55, tzinfo=UTC)]))
    assert out == ["gaps=0", "new=0"]


def test_the_incident_text_says_no_refresher_ran_either(tmp_path, capsys):
    _, report = _lookback(tmp_path, capsys, _by_workflow())
    assert "no `live` run" in report.read_text()



def _stub_gh(monkeypatch, runs: dict[int, list[dict]]):
    """Stub `gh`: `run list` returns the ids, `api .../runs/<id>/jobs` returns that run's
    steps. Nothing here calls GitHub."""
    import subprocess

    def run(cmd, **kw):
        class Done:
            stdout = ""
        if cmd[:3] == ["gh", "run", "list"]:
            Done.stdout = json.dumps([{"databaseId": i} for i in runs])
        elif cmd[:2] == ["gh", "api"]:
            rid = int(cmd[2].split("/runs/")[1].split("/")[0])
            Done.stdout = json.dumps({"jobs": [{"steps": runs[rid]}]})
        else:
            raise AssertionError(cmd)
        return Done
    monkeypatch.setattr(subprocess, "run", run)


def _step(conclusion, completed="2026-10-06T02:30:00Z", name="Watchdog check"):
    return {"name": name, "conclusion": conclusion, "completed_at": completed}


@pytest.mark.parametrize("steps, counts", [
    ([_step("success")], True),                     # the step completed
    ([_step("failure")], True),                     # it ran and found a problem
    ([_step("skipped")], False),              # cancelled before it: skipped
    ([_step("cancelled")], False),            # cancelled mid-step
    ([_step(None, None)], False),                   # never started
    ([], False),                                    # no such step at all
    ([_step("success", name="Refresh the overlay through the window")], False),
])
def test_a_live_run_is_a_check_only_if_its_step_completed(monkeypatch, steps, counts):
    """#461. Whatever the run's own conclusion, the step is the evidence. The first rows are
    the controls that flip: a hard-cancelled run (step skipped or missing) is not a check, a
    completed step is, and a `failure` run whose step ran is."""
    _stub_gh(monkeypatch, {1: steps})
    got = wg.check_times("live.yml", "o/r")
    assert got == ([dt.datetime(2026, 10, 6, 2, 30, tzinfo=UTC)] if counts else [])


def test_the_step_is_dated_by_its_own_completion_not_the_runs_update(monkeypatch):
    _stub_gh(monkeypatch, {1: [_step("success", "2026-10-06T01:10:00Z")]})
    assert wg.check_times("live.yml", "o/r") == [dt.datetime(2026, 10, 6, 1, 10, tzinfo=UTC)]


def test_a_hard_cancelled_loop_does_not_hide_a_gap(monkeypatch, tmp_path, capsys):
    """The #416 scenario end to end: the only live run was cancelled inside the window with
    its check skipped. The window has a kickoff and must be reported."""
    _stub_gh(monkeypatch, {1: [_step("skipped")]})
    out, _report = _lookback(tmp_path, capsys, lambda w, r: wg.check_times(w, r)
                            if w == "live.yml" else [])
    assert out == ["gaps=1", "new=1"]


def test_a_watchdog_run_is_still_dated_by_its_creation(monkeypatch):
    import subprocess

    class Done:
        stdout = '[{"createdAt": "2026-10-06T00:30:00Z"}]'
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Done())
    assert wg.check_times("watchdog.yml", "o/r") == [dt.datetime(2026, 10, 6, 0, 30, tzinfo=UTC)]


# --- #461: the season date has one home ----------------------------------------------------

def test_the_season_opening_is_read_from_the_workflow_that_holds_it(tmp_path):
    wf = tmp_path / "w.yml"
    wf.write_text('env:\n  SEASON_OPENS: "2026-10-07"\n')
    assert wg.season_opens_from(wf) == "2026-10-07"
    wf.write_text("env: {}\n")
    with pytest.raises(ValueError):
        wg.season_opens_from(wf)


def test_the_cli_filters_windows_by_the_date_in_the_named_file(tmp_path, capsys):
    """Planted: the same replay with the file's date before and after the window flips the
    result, so the flag is what filters."""
    wf = tmp_path / "w.yml"
    for opens, expected in (("2026-10-07", ["gaps=0", "new=0"]), ("2026-10-06", ["gaps=1", "new=1"])):
        wf.write_text(f'env:\n  SEASON_OPENS: "{opens}"\n')
        args = ["--live-yml", str(LIVE_YML), "--repo", "o/r", "--now", OCT6_NOW.isoformat(),
                "--season-opens-from", str(wf)]
        assert wg.main(args, _board(OCT6_KICK), _by_workflow()) == 0
        assert capsys.readouterr().out.split() == expected
