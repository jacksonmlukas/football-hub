"""`hub.season.weekly_forward`: the 2026 forward measurement admits only what the design admits,
speaks once at the horizon, and can see an effect (#432).

Every check here has its positive control (method rule 18), planted in the same file: a capture
written after kickoff is excluded at the read, a run before the horizon is shown to have loaded
no outcome by an `assemble` that raises if touched, a planted effect is detected in both
directions, and the harness's verdict is held equal to the one `scripts/weekly_forward_power.py`
simulated -- so the power table describes the rule that ships. No test touches the network or
`data/`: the frames are synthetic and `assemble` is a stand-in.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.fetch import consensus
from hub.fetch.consensus import Capture
from hub.models.experiment import Actions, Ceiling, gate
from hub.season import weekly_forward as wf

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "weekly_forward_power.py"
AFTER_HORIZON = date(2026, 12, 20)                  # week 14's last game is 2026-12-14


def _schedule() -> pl.DataFrame:
    rows = []
    for w in range(1, 19):
        thu = date(2026, 9, 10) + timedelta(days=7 * (w - 1))
        for off in (0, 3, 4):
            rows.append({"season": 2026, "week": w, "game_type": "REG",
                         "gameday": (thu + timedelta(days=off)).isoformat()})
    return pl.DataFrame(rows)


SCHEDULE = _schedule()
DAYS = consensus.first_game_days(SCHEDULE)
FLOORS = consensus.scrape_floors(SCHEDULE)


def _known(_path):
    """A first-commit time well before every deadline: the history git would report."""
    return datetime(2026, 9, 1, tzinfo=UTC)


def _read(**kw):
    """`read_forward` with the three things a real run supplies, defaulted to the clean case."""
    kw.setdefault("first_commit", _known)
    kw.setdefault("horizon_data", lambda: True)
    kw.setdefault("arm", dict(wf.PINNED_ARM_MODULES))
    return wf.read_forward(**kw)


def _cap(week: int, *, hours_before: float = 20.0, scrape_days_before: int = 1,
         path: Path | None = Path("state/consensus/2026/wk00/cap-x.json"), players: tuple[str, ...] = ("A", "B")) -> Capture:
    first = DAYS[week]
    return Capture(
        season=2026, week=week, captured_at=consensus.deadline_for(first)
        - timedelta(hours=hours_before), first_game_day=first,
        deadline=consensus.deadline_for(first), scrape_date=first - timedelta(days=scrape_days_before),
        digest="d", rows=pl.DataFrame({"player": list(players), "pos": ["WR"] * len(players),
                                       "team": ["KC"] * len(players),
                                       "ecr": [float(i + 1) for i in range(len(players))]}),
        path=path)


def _full(weeks=range(5, 15)) -> list[Capture]:
    return [_cap(w) for w in weeks]


def _paired(delta: float, *, weeks=range(5, 15), rosters: int = 20, week_sd: float = 1.4,
            noise: float = 7.5, ceiling: float = 10.0, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    wk = list(weeks)
    a = rng.normal(0, week_sd, len(wk))
    d = delta + a[None, :] + rng.normal(0, noise, (rosters, len(wk)))
    return pl.DataFrame({"season": np.full(d.size, 2026), "roster": np.repeat(np.arange(rosters), len(wk)),
                         "week": np.tile(wk, rosters), "diff": d.ravel(),
                         "ceiling_diff": np.full(d.size, ceiling)})


def _assemble(frame: pl.DataFrame, cover: dict | None = None):
    def run(_admitted):
        return frame, cover
    return run


def _run(reading: wf.Reading):
    assert reading.run is not None
    return reading.run


def _boom(_admitted):
    raise AssertionError("an outcome was loaded")


# --- the horizon: it speaks once, and loads nothing before ---------------------------------------

def test_a_run_before_the_horizon_reads_no_outcome():
    """The plant: `assemble` is the only function that reads results, and here it raises if it is
    called. A run on the day week 14's last game is played, and on the day after, is NOT-YET; the
    reading exists on the second day after."""
    for as_of in (date(2026, 10, 7), date(2026, 12, 14), date(2026, 12, 15)):
        r = _read(as_of=as_of, schedule=SCHEDULE, captures=_full(), assemble=_boom)
        assert r.status == "NOT-YET" and r.run is None
        assert "No outcome was loaded" in r.lines[0]
    r = _read(as_of=date(2026, 12, 16), schedule=SCHEDULE, captures=_full(),
                        assemble=_assemble(_paired(0.0)))
    assert r.status != "NOT-YET"


def test_no_schedule_for_the_horizon_week_is_not_yet_not_a_verdict():
    short = SCHEDULE.filter(pl.col("week") < wf.HORIZON_WEEK)
    r = _read(as_of=AFTER_HORIZON, schedule=short, captures=_full(), assemble=_boom)
    assert r.status == "NOT-YET" and "no schedule" in r.lines[0]


def test_dates_past_the_horizon_with_week_14_absent_from_the_data_is_not_yet():
    """Rule 18: the dates say the horizon is past and the data says week 14 is not there -- the
    reading must wait, and must not have loaded an outcome to find out. Flip it: with the rows
    present the same call reads."""
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(), assemble=_boom,
              horizon_data=lambda: False)
    assert r.status == "NOT-YET" and "not in nflverse" in r.lines[0]
    ok = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
               assemble=_assemble(_paired(0.0)), horizon_data=lambda: True)
    assert ok.status != "NOT-YET"
    # and the data is not asked about before the date has passed
    early = _read(as_of=date(2026, 10, 7), schedule=SCHEDULE, captures=_full(), assemble=_boom,
                  horizon_data=lambda: (_ for _ in ()).throw(AssertionError("asked early")))
    assert early.status == "NOT-YET"


def test_a_different_arm_refuses_the_reading_and_loads_nothing():
    """Rule 18: plant an arm that is not the pinned one -- one module edited, one added, one
    gone, and `None`, an arm that cannot be identified -- and the run refuses, names what
    differs, and says a new arm needs a new pre-registration. Flip it: the pinned arm reads."""
    pinned = dict(wf.PINNED_ARM_MODULES)
    edited = {**pinned, "hub.models.panel": "0" * 12}
    added = {**pinned, "hub.models.new_dvp": "1" * 12}
    gone = {k: v for k, v in pinned.items() if k != "hub.models.panel"}
    for arm, said in ((edited, "hub.models.panel changed"),
                      (added, "hub.models.new_dvp is new"),
                      (gone, "hub.models.panel is no longer"),
                      (None, "could not be read")):
        r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(), assemble=_boom,
                  arm=arm)
        assert r.status == "REFUSED" and "new pre-registration" in r.lines[0]
        assert said in r.lines[0]
    ok = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
               assemble=_assemble(_paired(0.0)), arm=pinned)
    assert ok.status != "REFUSED"
    assert f"closure {wf.PINNED_ARM_DIGEST}" in "\n".join(ok.lines)


def test_the_pin_is_recorded_in_the_document():
    """The constants and `docs/weekly-forward.md` agree: the document records the closure's
    digest and every module's digest, so a re-pin of the constant alone fails here."""
    doc = (Path(__file__).resolve().parents[2] / "docs" / "weekly-forward.md").read_text()
    assert wf.PINNED_ARM_DIGEST in doc
    for module, digest in wf.PINNED_ARM_MODULES.items():
        assert f"`{module}` `{digest}`" in doc, module
    assert wf.PINNED_ARM_DIGEST == wf.arm_digest(wf.PINNED_ARM_MODULES)


def _with_source_edited(monkeypatch, module: str):
    """Plant an edit in `module`'s source as `module_digests` reads it, touching no file."""
    import importlib.util

    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    origin = Path(spec.origin)
    real = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes",
        lambda self: real(self) + b"\n# planted\n" if self == origin else real(self))


def test_an_edit_to_the_models_panel_refuses_the_reading_and_an_unchanged_closure_reads(
        monkeypatch):
    """#456's control. `hub.models.panel` is not the file the old pin hashed, and an edit there
    (#315's opponent-adjusted DvP is the example) changes what the arm computes. Plant one in the
    source as it is read: the reading from the live closure refuses and names the module.
    Unplanted, the same call reads."""
    arm = wf.arm_closure()
    assert arm == wf.PINNED_ARM_MODULES, "the closure moved: re-pin per docs/weekly-forward.md"
    ok = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
               assemble=_assemble(_paired(0.0)), arm=arm)
    assert ok.status != "REFUSED"
    _with_source_edited(monkeypatch, "hub.models.panel")
    planted = wf.arm_closure()
    assert planted is not None and planted != arm
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(), assemble=_boom,
              arm=planted)
    assert r.status == "REFUSED" and "hub.models.panel changed" in r.lines[0]
    assert wf.arm_difference(planted) == ["hub.models.panel changed"]


def test_an_edit_to_an_exempt_module_does_not_refuse_and_the_exemption_holds(monkeypatch):
    """The exemptions are not in the closure -- not hashed, not walked into -- so an edit there
    leaves the reading alone. The reason holds only if the module really is outside what the arm
    computes: `hub.paths` is constants and `hub.config` is hashed by `config_digest`; neither is
    reached except as an exempt import. Plant an edit in each and the closure is unchanged."""
    from hub.closure import CLOSURE_EXEMPT, closure_exempt

    for module in ("hub.paths", "hub.config", "hub.jsonio"):
        assert closure_exempt(module) and module in CLOSURE_EXEMPT and CLOSURE_EXEMPT[module]
        assert module not in wf.PINNED_ARM_MODULES
    with monkeypatch.context() as m:
        for module in ("hub.paths", "hub.config", "hub.jsonio"):
            _with_source_edited(m, module)
        assert wf.arm_closure() == wf.PINNED_ARM_MODULES
        r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                  assemble=_assemble(_paired(0.0)), arm=wf.arm_closure())
        assert r.status != "REFUSED"


def test_the_pinned_closure_is_the_roots_import_closure_and_holds_both_roots():
    """The pin follows the imports: the roots are in it, the walk is the ledger's, and the
    forward harness itself (which reads the verdict, not the arm) is not."""
    from hub.closure import import_closure

    assert set(wf.PINNED_ARM_MODULES) == import_closure(wf.ARM_ROOTS)
    assert set(wf.ARM_ROOTS) <= set(wf.PINNED_ARM_MODULES)
    assert "hub.models.panel" in wf.PINNED_ARM_MODULES
    assert "hub.season.weekly_forward" not in wf.PINNED_ARM_MODULES


def test_the_horizon_is_the_day_after_week_14s_last_game():
    last = date(2026, 12, 14)
    assert not wf.horizon_reached(last + timedelta(days=1), last)
    assert wf.horizon_reached(last + timedelta(days=2), last)
    assert not wf.horizon_reached(AFTER_HORIZON, None)


# --- admission: only the weeks the design admits -------------------------------------------------

def test_a_capture_after_kickoff_is_excluded_at_the_read_and_named():
    """Rule 18, the read half: the write refuses a late capture, but a file placed in the tree by
    any other route is excluded here, with its week and the reason, and the week is not scored."""
    late = _cap(9, hours_before=-3)                          # written three hours into the day
    adm = wf.admit([*_full(range(5, 9)), late, *_full(range(10, 15))], DAYS, FLOORS, first_commit=_known)
    assert 9 not in adm.admitted and "at or after the deadline" in adm.refused[9]
    assert sorted(adm.admitted) == [5, 6, 7, 8, 10, 11, 12, 13, 14]
    # and on the right side of the same boundary the same week is admitted
    assert 9 in wf.admit([_cap(9, hours_before=0.01)], DAYS, FLOORS, first_commit=_known).admitted


def test_a_stale_page_is_refused_and_the_latest_valid_capture_is_the_one_used():
    stale = _cap(6, scrape_days_before=9)                    # nine days before: week 5's page
    assert "not week 6's page" in wf.admit([stale], DAYS, FLOORS, first_commit=_known).refused[6]
    # week 5's Wednesday page filed under week 6 is stale however recent it is
    wrong = dataclasses.replace(_cap(6), scrape_date=date(2026, 10, 7))
    assert "not week 6's page" in wf.admit([wrong], DAYS, FLOORS, first_commit=_known).refused[6]
    early, later = _cap(7, hours_before=60), _cap(7, hours_before=5)
    assert wf.admit([early, later], DAYS, FLOORS, first_commit=_known).admitted[7][0].captured_at == later.captured_at
    # one bad capture does not veto a good one of the same week
    adm = wf.admit([_cap(7, scrape_days_before=9), later], DAYS, FLOORS, first_commit=_known)
    assert 7 in adm.admitted and 7 not in adm.refused


def test_a_week_with_no_capture_is_named_not_scored():
    adm = wf.admit(_full(range(8, 15)), DAYS, FLOORS, first_commit=_known)
    assert adm.uncaptured == (1, 2, 3, 4, 5, 6, 7) and adm.refused == {}
    assert sorted(adm.admitted) == list(range(8, 15))
    assert wf.admit([_cap(15)], DAYS, FLOORS, first_commit=_known).admitted == {}        # 15-17 are never read


def test_a_capture_first_committed_after_the_deadline_is_not_admitted():
    """The file says it was captured in time; git says it arrived late. The commit is the
    timestamp. Where git cannot say (None) the capture is not admitted (the
    shallow-clone plant, at the end of this test)."""
    p = Path("state/consensus/2026/wk05/cap-x.json")
    cap = _cap(5, path=p)
    late = consensus.deadline_for(DAYS[5]) + timedelta(hours=2)
    assert "after the deadline" in wf.admit([cap], DAYS, FLOORS,
                                            first_commit=lambda _p: late).refused[5]
    early = consensus.deadline_for(DAYS[5]) - timedelta(hours=2)
    assert 5 in wf.admit([cap], DAYS, FLOORS, first_commit=lambda _p: early).admitted
    # Rule 18, the shallow-clone plant: git cannot say when it was committed, and the file's own
    # `captured_at` (well before the deadline) must not stand in for it. Flip it: a known time
    # admits the identical capture.
    unknown = wf.admit([cap], DAYS, FLOORS, first_commit=lambda _p: None)
    assert 5 not in unknown.admitted
    assert "first-commit time unknown" in unknown.refused[5] and "cap-x.json" in unknown.refused[5]
    assert "fetch-depth: 0" in unknown.refused[5]
    assert 5 in wf.admit([cap], DAYS, FLOORS, first_commit=_known).admitted
    no_path = dataclasses.replace(cap, path=None)
    assert 5 not in wf.admit([no_path], DAYS, FLOORS, first_commit=_known).admitted


def test_first_commit_at_reads_git_and_says_none_where_it_cannot(tmp_path):
    f = tmp_path / "cap.json"
    f.write_text("{}")
    assert wf.first_commit_at(f) is None                    # not a repository
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CONFIG_SYSTEM": "/dev/null"}

    def git(*args, when=None):
        e = {**env, **({"GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when} if when else {})}
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", **e})

    git("init", "-q")
    git("add", "cap.json")
    git("commit", "-q", "-m", "x", when="2026-10-07T12:00:00+00:00")
    assert wf.first_commit_at(f) == datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    f.write_text('{"a": 1}')
    git("commit", "-q", "-am", "y", when="2026-10-09T12:00:00+00:00")
    assert wf.first_commit_at(f) == datetime(2026, 10, 7, 12, 0, tzinfo=UTC)   # the first add
    (tmp_path / "new.json").write_text("{}")
    assert wf.first_commit_at(tmp_path / "new.json") is None                    # untracked


# --- in time per player (ADR-0027, #515) ---------------------------------------------------------
#
# A schedule with teams and kickoffs, Eastern as nflverse spells them: Thursday DAL@PHI 20:15,
# Sunday KC@BUF 13:00 and SF@SEA 16:25 (SF and SEA on a bye in week 7), Monday NYJ@MIA 20:15.

def _timed_schedule() -> pl.DataFrame:
    rows = []
    for w in range(1, 19):
        thu = date(2026, 9, 10) + timedelta(days=7 * (w - 1))
        games = [(0, "DAL", "PHI", "20:15"), (3, "KC", "BUF", "13:00"),
                 (4, "NYJ", "MIA", "20:15")]
        if w != 7:
            games.append((3, "SF", "SEA", "16:25"))
        for off, away, home, t in games:
            rows.append({"season": 2026, "week": w, "game_type": "REG", "away_team": away,
                         "home_team": home, "gameday": (thu + timedelta(days=off)).isoformat(),
                         "gametime": t})
    return pl.DataFrame(rows)


TIMED = _timed_schedule()
KICKS = consensus.kickoffs(TIMED)
TEAMS = {"Thu Guy": "PHI", "Sun Guy": "KC", "Mon Guy": "MIA", "Bye Guy": "SF", "Nobody": "FA"}


def _friday(week: int, *, hour_utc: int = 15, players: dict[str, str] = TEAMS,
            name: str = "lineup-fri.json") -> Capture:
    """The Friday capture of `week`'s page, as `read_every_capture` hands it over."""
    thu = DAYS[week]
    fri = thu + timedelta(days=1)
    return Capture(
        season=2026, week=week, captured_at=datetime(fri.year, fri.month, fri.day, hour_utc,
                                                     tzinfo=UTC),
        first_game_day=thu, deadline=consensus.lineup_deadline_for(thu + timedelta(days=3)),
        scrape_date=fri, digest="d", purpose=consensus.LINEUP,
        rows=pl.DataFrame({"player": list(players), "pos": ["WR"] * len(players),
                           "team": list(players.values()),
                           "ecr": [float(i + 1) for i in range(len(players))]}),
        path=Path(f"state/consensus/2026/lineup/wk{week:02d}/{name}"))


def _players(caps) -> set[str]:
    return {p for c in caps for p in c.rows["player"].to_list()}


def test_a_friday_capture_is_in_time_for_sunday_players_and_not_for_thursdays():
    """ADR-0027: in time for a player means taken before *his* kickoff. Friday's page counts for
    Sunday's and Monday's players and for a player on a bye (no game to have seen), and not for
    Thursday's, nor for one the schedule cannot place."""
    adm = wf.admit([_friday(7)], DAYS, FLOORS, first_commit=_known, kickoffs=KICKS)
    assert 7 in adm.admitted and 7 not in adm.refused
    assert _players(adm.admitted[7]) == {"Sun Guy", "Mon Guy", "Bye Guy"}
    assert adm.excluded == {7: 2}                                   # Thu Guy and Nobody
    assert any("week 7: 2 ranked player(s) left out" in ln for ln in wf._admission_lines(adm))
    # a capture in time for none of the players it ranks admits nothing and says why
    only_thursday = _friday(7, players={"Thu Guy": "PHI", "Nobody": "FA"})
    none = wf.admit([only_thursday], DAYS, FLOORS, first_commit=_known, kickoffs=KICKS)
    assert 7 not in none.admitted and "2 ranked, none in time" in none.refused[7]


def test_the_comparison_leaves_thursdays_players_out_of_the_week():
    """The frame the gate reads has no row for a Thursday player under a Friday capture, so he is
    outside `priced_by_both` for both arms that week. With a week capture taken on Wednesday too,
    each player is read from the newest capture in time for him: Thursday's from Wednesday's
    page, everyone else from Friday's."""
    from hub.names import player_key

    frame = wf.consensus_frame(
        wf.admit([_friday(7)], DAYS, FLOORS, first_commit=_known, kickoffs=KICKS).admitted)
    assert player_key("Thu Guy") not in frame["key"].to_list()
    assert player_key("Sun Guy") in frame["key"].to_list()
    wed = dataclasses.replace(_cap(7, players=("Thu Guy", "Sun Guy")),
                              rows=_friday(7, players={"Thu Guy": "PHI", "Sun Guy": "KC"}).rows)
    adm = wf.admit([wed, _friday(7)], DAYS, FLOORS, first_commit=_known, kickoffs=KICKS)
    fri_cut, wed_cut = adm.admitted[7]
    assert fri_cut.purpose == consensus.LINEUP and wed_cut.captured_at == wed.captured_at
    assert set(wed_cut.rows["player"]) == {"Thu Guy"}
    assert "Sun Guy" in set(fri_cut.rows["player"]) and adm.excluded == {7: 1}   # Nobody


def test_a_capture_committed_after_sundays_kickoff_is_in_time_only_for_mondays_players():
    """The commit is the timestamp: a Friday file first committed at Sunday 18:00 UTC (after the
    13:00 ET kickoff, 17:00 UTC) is not in time for Sunday's players whatever it says of itself."""
    sunday_late = datetime(2026, 10, 25, 18, tzinfo=UTC)
    adm = wf.admit([_friday(7)], DAYS, FLOORS, first_commit=lambda _p: sunday_late,
                   kickoffs=KICKS)
    assert _players(adm.admitted[7]) == {"Mon Guy", "Bye Guy"}
    after_monday = datetime(2026, 10, 27, 4, tzinfo=UTC)          # 00:00 ET Tuesday
    gone = wf.admit([_friday(7)], DAYS, FLOORS, first_commit=lambda _p: after_monday,
                    kickoffs=KICKS)
    assert 7 not in gone.admitted and "in time for no player" in gone.refused[7]


def test_weeks_before_the_adoption_are_replayed_by_the_rule_they_were_registered_under():
    """Replay invariance: weeks 5 and 6 were played before ADR-0027 was adopted (2026-10-20), so
    the per-player rule does not reach back into them. Their Friday captures are refused exactly
    as the per-week rule refused them, with or without kickoff times; week 7 is the first read
    per player."""
    caps = [_friday(5), _friday(6), _friday(7)]
    timed = wf.admit(caps, DAYS, FLOORS, first_commit=_known, kickoffs=KICKS)
    untimed = wf.admit(caps, DAYS, FLOORS, first_commit=_known)
    assert wf.PER_PLAYER_FROM_WEEK == 7
    for week in (5, 6):
        assert week not in timed.admitted and week not in untimed.admitted
        assert timed.refused[week] == untimed.refused[week]
        assert "at or after the deadline" in timed.refused[week]
    assert 7 in timed.admitted and 7 not in untimed.admitted
    # and a week read by the per-week rule is read off its newest capture alone, as it was:
    # an older capture's extra players are not merged in
    early = _cap(6, hours_before=60, players=("A", "B", "Only Early"))
    late = _cap(6, hours_before=5, players=("A", "B"))
    adm = wf.admit([early, late], DAYS, FLOORS, first_commit=_known, kickoffs=KICKS)
    assert len(adm.admitted[6]) == 1 and adm.admitted[6][0].captured_at == late.captured_at
    assert adm.admitted[6][0].rows.equals(late.rows) and adm.excluded == {}


def test_a_forward_reading_on_friday_captures_runs_without_thursdays_players():
    """End to end through `read_forward`: eight Friday captures (weeks 7-14) are enough weeks to
    read, and what `assemble` is handed holds no Thursday player in any week."""
    seen: dict = {}

    def assemble(admitted):
        seen.update(admitted)
        return _paired(0.0, weeks=range(7, 15)), None

    r = _read(as_of=AFTER_HORIZON, schedule=TIMED,
              captures=[_friday(w) for w in range(7, 15)], assemble=assemble)
    assert r.status not in {"NOT-RUNNABLE", "NOT-YET", "REFUSED"}
    assert sorted(seen) == list(range(7, 15))
    assert all("Thu Guy" not in _players(caps) for caps in seen.values())
    assert any("ranked player(s) left out" in ln for ln in r.lines)


# --- too few weeks: the exemption is named -------------------------------------------------------

def test_fewer_than_the_pre_registered_weeks_is_not_runnable_and_loads_nothing():
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(range(10, 15)),
                        assemble=_boom)
    assert r.status == "NOT-RUNNABLE" and "5 admitted week(s), fewer than the 6" in r.lines[0]
    assert wf.MIN_WEEKS == 6
    r6 = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(range(9, 15)),
                         assemble=_assemble(_paired(0.0, weeks=range(9, 15))))
    assert r6.status != "NOT-RUNNABLE"


# --- the rule: a planted effect is detected, in both directions ----------------------------------

def test_a_planted_effect_is_detected_in_both_directions():
    """The positive control the design asks for. An effect a hundred points wide is ADOPT, its
    negative REMOVE, and the same frame with no effect is neither -- so a harness that returned
    SHOW for everything, or ADOPT for everything, would fail here."""
    up = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                         assemble=_assemble(_paired(100.0)))
    down = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                           assemble=_assemble(_paired(-100.0)))
    null = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                           assemble=_assemble(_paired(0.0)))
    assert (up.status, down.status, null.status) == ("ADOPT", "REMOVE", "SHOW")
    assert _run(up).resolved == 1 and _run(up).abstained == 0
    assert _run(up).summary["clusters"] == 10                  # the week is the cluster
    assert any("weeks admitted (10)" in line for line in up.lines)
    assert any("caveat" in line for line in up.lines)


def test_the_week_is_the_cluster_not_the_roster_and_it_widens_the_interval():
    """Rosters share the week's slate, so a week effect common to all of them is one reading
    and not twenty. The same frame clustered on the roster gives a narrower interval than the
    one this reads -- the difference the design's unit choice makes, seen rather than asserted."""
    frame = _paired(0.5, week_sd=3.0, noise=2.0)
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                        assemble=_assemble(frame))
    by_week = _run(r).summary
    by_roster = gate(frame.drop("ceiling_diff"), cluster=("roster",), within=("roster",),
                     ceiling=Ceiling("x", np.array([10.0])),
                     actions=Actions(adopt="A", remove="R", show="S")).summary
    assert by_week["se"] > 2 * by_roster["se"]
    assert by_week["clusters"] == 10 and by_roster["clusters"] == 20


def test_a_ceiling_below_the_mde_is_not_runnable_in_the_forward_gate_too():
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                        assemble=_assemble(_paired(0.0, ceiling=0.05)))
    assert r.status == "NOT-RUNNABLE" and "ceiling" in _run(r).verdict[1]


def test_a_join_failure_voids_the_forward_gate_as_it_voids_the_weekly_gate():
    cover = {"cells": 100.0, "join_failure": 0.5}
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                        assemble=_assemble(_paired(100.0), cover))
    assert r.status == "VOID"


def test_a_frame_with_no_ceiling_column_is_not_runnable_not_adopted():
    r = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                        assemble=_assemble(_paired(100.0).drop("ceiling_diff")))
    assert r.status == "NOT-RUNNABLE"


# --- the power table describes the rule that ships -----------------------------------------------

@pytest.fixture(scope="module")
def power():
    spec = importlib.util.spec_from_file_location("weekly_forward_power", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["weekly_forward_power"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_simulated_rule_is_the_shipped_rule(power):
    """`scripts/weekly_forward_power.py` simulates `experiment.gate` with the week as the cluster
    and the roster as the within-season unit. Planted on one frame, the script's reading and the
    harness's must be the same verdict -- and the call's two units are the module's own."""
    assert power.ROSTERS == 20 and wf.CLUSTER_WEEK == ("week",)
    for delta in (100.0, -100.0, 0.0):
        frame = _paired(delta)
        shipped = _read(as_of=AFTER_HORIZON, schedule=SCHEDULE, captures=_full(),
                                  assemble=_assemble(frame)).status
        assert power.read(frame.drop("ceiling_diff"), 0, 4000) == shipped


def test_the_power_script_sees_a_planted_effect_and_a_null_is_not_degenerate(power):
    cells = [power.Cell(10, 100.0), power.Cell(10, -100.0), power.Cell(10, 0.0)]
    res = power.run_cells(cells, trials=200, workers=1)
    assert res[0]["verdict"].get("ADOPT", 0) == 200
    assert res[1]["verdict"].get("REMOVE", 0) == 200
    null = res[2]["verdict"]
    assert null.get("ADOPT", 0) < 40 and null.get("REMOVE", 0) < 40       # near 5% two-sided
    assert null.get("SHOW", 0) > 150


def test_the_week_component_is_the_disattenuated_fit(power):
    assert power.WEEK_VAR == pytest.approx(1.8814 ** 2 - 61.5 / 40)
    assert power.baseline_row_var() == pytest.approx(62.564, abs=1e-3)   # #388's, not re-derived


# --- the captures become the frame the gate reads ------------------------------------------------

def test_the_captures_become_the_consensus_frame_the_gate_reads():
    frame = wf.consensus_frame({5: (_cap(5, players=("Patrick Mahomes II", "Star One")),),
                                6: (_cap(6, players=("Star One",)),)})
    assert set(frame.columns) >= {"season", "week", "key", "ecr", "lead_days"}
    assert sorted(frame["week"].unique().to_list()) == [5, 6]
    assert frame.filter(pl.col("week") == 5).height == 2
    assert frame["key"].str.contains("mahomes").any()                  # the repo's own player key


def test_the_pinned_arm_is_the_closure_the_arm_runs():
    """#430, widened by #456: the pin was stale for a day before anyone noticed once (#309 edited
    a diagnostic's prose in the pinned file), and the first anyone would have heard of it was a
    REFUSED verdict in November. This holds the constant to what the tree's closure is, so an
    edit to any module the arm imports -- a comment included -- fails here, naming it, and not
    at the reading."""
    arm = wf.arm_closure()
    assert arm is not None
    assert not wf.arm_difference(arm), (
        f"{wf.arm_difference(arm)}. If the edit changed what the projection computes it is a new "
        f"arm and needs a new pre-registration; if it did not, prove that with "
        f"tests/unit/test_weekly_projection_move.py, re-pin PINNED_ARM_MODULES, and amend "
        f"docs/weekly-forward.md, dated, before any 2026 outcome is read.")
