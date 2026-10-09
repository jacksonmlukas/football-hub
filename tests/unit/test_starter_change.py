"""The quarterback adjustment's gate (#291) over the starter-change events and the CLI that runs
it beside the study (#346). The gate is held to its pre-registration in `docs/gate-power.md` --
the frozen price is the comparator, the shipped estimator is the arm, fewer than three
event-seasons is not-runnable. The events are held in `test_starter_events.py`, the study in
`test_starter_study.py`.

Every fixture is synthetic and every test runs with no `data/` and no network.
"""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest
from starter_world import (
    ARCHIVE,
    PRIOR,
    SEASON,
    played_rows,
    polls,
    schedule_for,
    serve_world,
    source_rows,
    utc,
)

from hub.fetch import nfeloqb, nflverse, odds, replay
from hub.ledger import Ledger, WidthEntry
from hub.models import experiment, quarterback
from hub.models import starter_change as sc
from hub.models import starter_events as se
from hub.models import starter_study as ss
from hub.models.market import MARGIN_SD, normal_cdf


@pytest.fixture
def rows():
    return source_rows(PRIOR + SEASON)


@pytest.fixture(autouse=True)
def _the_events_source_is_served_offline(rows):
    """Every test here that drives `main` reads its events through `nflverse.load`, which on
    the network adapter would leave the machine; the offline suite fails a test that tries.
    The default world is the fixture's own rows read as play-by-play."""
    serve_world(rows)




def test_the_arm_is_the_shipped_seam_and_prices_the_departing_starter(rows):
    """The gate's arm is `nfeloqb.state(rows, as_of=<the week's first game day>)` handed to
    `quarterback.apply` -- what `ratings._rated_by_week` does for a played week. Rows
    strictly before 2026-09-27 hold KC's week-2 row (Mahomes, +42) and LA's last 2025 row
    (Stafford, +10), so the frozen +3 moves by (42 - 10) / 25: the *departing* starter's
    adjustment, because the file carries Gabbert on no row before his first start. Both
    arms score the home result by `MarketBaseline`'s conversion; the difference is
    unadjusted minus adjusted, positive when the adjustment helped."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    paired = sc.gate_rows(ARCHIVE, games, tg, rows)
    assert paired["game_id"].to_list() == ["2026_03_LA_KC"]      # week 4 has no result yet
    row = paired.row(0, named=True)
    shipped = nfeloqb.state(rows, as_of=dt.date(2026, 9, 27))
    assert shipped.filter(pl.col("team") == "KC")["qb"][0] == "Mahomes"
    assert row["adjusted_adjustment"] == pytest.approx((42.0 - 10.0) / quarterback.ELO_PER_POINT)
    assert row["adjusted"] == pytest.approx(3.0 + row["adjusted_adjustment"])
    # KC lost 10-20 at home: y = 0.
    ll = lambda s: -math.log(1.0 - normal_cdf(s / MARGIN_SD))  # noqa: E731
    assert row["diff"] == pytest.approx(ll(3.0) - ll(row["adjusted"]))
    assert row["ceiling"] == pytest.approx(ll(3.0) - ll(-2.0))
    assert row["season"] == 2026


def test_the_oracle_arm_knows_the_arriving_starter_and_is_reported_beside_the_gate(rows):
    """The diagnostic: the same estimator with the week's own rows as the state, so the
    frozen +3 moves by (-110 - (-90)) / 25 -- Gabbert against Bethard. Its difference is a
    separate column the rule never reads."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    row = sc.gate_rows(ARCHIVE, games, tg, rows).row(0, named=True)
    assert row["oracle_adjustment"] == pytest.approx((-110.0 + 90.0) / quarterback.ELO_PER_POINT)
    assert row["oracle"] == pytest.approx(3.0 + row["oracle_adjustment"])
    ll = lambda s: -math.log(1.0 - normal_cdf(s / MARGIN_SD))  # noqa: E731
    assert row["oracle_diff"] == pytest.approx(ll(3.0) - ll(row["oracle"]))
    assert row["oracle_diff"] != pytest.approx(row["diff"])


def test_a_game_whose_opponent_has_no_prior_state_is_excluded_not_zeroed():
    """The gate guards on `adjusted_by`, the adjustment's own marker, never on the moved
    spread column: `quarterback.apply` leaves that column at the frozen price -- unmoved,
    and therefore never null -- on a game it did not touch. A's week-2 change is a genuine
    in-season event, but its opponent Z has no row anywhere before that week, so the shipped
    state has nothing to price Z with and `apply` needs both sides: the whole game goes
    untouched. The old not-null filter on the spread column let a game like this through at
    a manufactured zero difference and inflated n; guarding on the marker excludes it."""
    rows_ = source_rows([
        ("2026-09-06", 2026, "1.0", "A", "B", "a1", "b1", 100.0, 90.0, 5.0, 3.0,
         20, 10, .55, .58),
        ("2026-09-13", 2026, "2.0", "A", "Z", "a2", "z1", 130.0, 95.0, 40.0, 4.0,
         17, 24, .60, .70),
    ])
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games.height == 1                                   # A's week-2 change: the only event
    store_archive = polls([
        ("2026_02_Z_A", 2.0, utc(5), 2),      # before frozen_before (09-06): the frozen price
        ("2026_02_Z_A", -1.0, utc(10), 2),    # before the game day (09-13): the close
    ])
    paired = sc.gate_rows(store_archive, games, tg, rows_)
    assert paired.is_empty()


def test_the_rule_reads_the_shipped_arm_and_never_the_oracle(tmp_path):
    """Three seasons where the oracle would ADOPT and the shipped arm loses everywhere: the
    verdict is the shipped arm's."""
    paired = _paired([2026, 2027, 2028], diff=-0.05).with_columns(
        pl.lit(0.05).alias("oracle_diff"))
    run = sc.run(paired, needed=3, ledger=Ledger(path=None))
    assert run.verdict[0] == "REMOVE"


def test_the_pilot_reads_the_sources_own_two_columns_on_event_games_only(rows):
    """The power input: the source's base and quarterback-adjusted probabilities, scored on
    the same event games and nowhere else, per season; the target is the absolute mean of the
    season means and `s` their spread. One season has a mean and no spread."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    pilot = sc.pilot(tg, games)
    assert pilot["seasons"] == 1 and pilot["n"] == 1        # week 4 is unplayed
    ll = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))  # noqa: E731
    assert pilot["target"] == pytest.approx(abs(ll(0.65, 0.0) - ll(0.52, 0.0)))
    assert math.isnan(pilot["season_sd"])


NEGATIVE_PILOT_GAME = [
    ("2026-09-06", 2026, "1.0", "A", "X", "a1", "x1", 100.0, 90.0, 5.0, 3.0, 20, 10, .30, .20),
    ("2026-09-13", 2026, "2.0", "A", "Y", "a2", "y1", 120.0, 95.0, 6.0, 3.5, 24, 17, .30, .20),
]


def test_the_pilot_target_is_the_absolute_mean_not_the_signed_one():
    """#304 (rule 15): the fixture above has one event game and its diff is already positive,
    so `abs()` is the identity there and its deletion is invisible. Here the source's
    quarterback-adjusted column (0.20) is the *worse* predictor of the home win the home team
    goes on to get (y=1) than the base column (0.30) -- a genuine loss, so the mean of season
    means is negative -- and `target` must still be its absolute value, not the signed mean
    itself."""
    rows_ = source_rows(NEGATIVE_PILOT_GAME)
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games.height == 1
    pilot = sc.pilot(tg, games)
    ll = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))  # noqa: E731
    signed = ll(0.30, 1.0) - ll(0.20, 1.0)
    assert signed < 0
    assert pilot["per_season"][0]["mean"] == pytest.approx(signed)
    assert pilot["target"] == pytest.approx(abs(signed))


def test_event_seasons_needed_is_the_smallest_k_whose_mde_clears_the_target():
    """On the t reference: at s = 1 the MDE is 9.58 s at two seasons, 2.97 at three, 2.01 at
    four -- the table in the pre-registration -- so a target of 2.5 needs four and a target
    of 10 needs two. A target no cap reaches is None, not a number."""
    def mde_at(k):
        return experiment.minimum_detectable_effect(1.0 / math.sqrt(k), k)

    assert mde_at(2) == pytest.approx(9.58, abs=0.01)
    assert mde_at(3) == pytest.approx(2.97, abs=0.01)
    assert mde_at(4) == pytest.approx(2.01, abs=0.01)
    assert sc.event_seasons_needed(2.5, 1.0) == 4
    assert sc.event_seasons_needed(10.0, 1.0) == 2
    assert sc.event_seasons_needed(0.0001, 1.0, cap=50) is None


def _paired(seasons, n=6, diff=0.05):
    return pl.DataFrame({"season": [s for s in seasons for _ in range(n)],
                         "game_id": [f"g{s}_{i}" for s in seasons for i in range(n)],
                         "diff": [diff] * (n * len(seasons)),
                         "ceiling": [0.5] * (n * len(seasons))})


def test_fewer_than_three_event_seasons_is_not_runnable_and_names_the_count_needed(tmp_path):
    """The first pre-registered precondition, since #300 an exemption from this diagnostic
    firing rather than a bar to the module's ADOPT condition (#221's coefficient). The house
    rule is never read: a frame that would ADOPT at three seasons is NOT-RUNNABLE at two, and
    the sentence carries the event-season count the pilot says is needed."""
    run = sc.run(_paired([2026, 2027]), needed=31, ledger=Ledger(path=None))
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert "2 event-season" in run.verdict[1] and "31" in run.verdict[1]
    assert "exemption" in run.verdict[1] and "#221" in run.verdict[1]
    enough = sc.run(_paired([2026, 2027, 2028]), needed=3, ledger=Ledger(path=None))
    assert enough.verdict[0] == "ADOPT"


def test_the_verdict_names_itself_a_diagnostic_on_every_branch(tmp_path):
    """Amended 2026-09-17 (#300): none of the gate's own branches license ADOPT or pull the
    module -- #221's line-move coefficient does, and this gate is read beside it. The three
    sentences say so, on ADOPT as much as on SHOW or REMOVE."""
    mixed = pl.concat([_paired([2026, 2027], diff=0.05), _paired([2028], diff=-0.05)])
    show = sc.run(mixed, needed=3, ledger=Ledger(path=None))
    assert show.verdict[0] == "SHOW"
    assert "Diagnostic only" in show.verdict[1] and "#221" in show.verdict[1]

    adopt = sc.run(_paired([2026, 2027, 2028]), needed=3, ledger=Ledger(path=None))
    assert adopt.verdict[0] == "ADOPT"
    assert "Diagnostic only" in adopt.verdict[1] and "#221" in adopt.verdict[1]

    remove = sc.run(_paired([2026, 2027, 2028], diff=-0.05), needed=3,
                    ledger=Ledger(path=None))
    assert remove.verdict[0] == "REMOVE"
    assert "Diagnostic only" in remove.verdict[1] and "#221" in remove.verdict[1]


def test_no_rows_is_not_runnable_with_zero_event_seasons(tmp_path):
    run = sc.run(pl.DataFrame(schema={"season": pl.Int64, "diff": pl.Float64}), needed=None,
                 ledger=Ledger(path=None))
    assert run.verdict[0] == "NOT-RUNNABLE" and "0 event-season" in run.verdict[1]


def test_not_runnable_publishes_no_interval_and_records_no_width():
    """The three-season floor is applied before the summary, the house verdict, the width
    history and the rendered lines -- not applied to the house verdict alone after they have
    already run. An underpowered run's `lines` are empty (no CI, no MDE, no ceiling check,
    no stamp) and the ledger is never touched -- a run that publishes no interval has no width
    to keep. Checked on the ledger itself, not a file: a comparable entry recorded afterward
    sees no history, so nothing was written."""
    ledger = Ledger(path=None)
    run = sc.run(_paired([2026, 2027]), needed=31, ledger=ledger)
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert run.lines == []
    probe = WidthEntry(name="quarterback_gate", config_digest="cfg", data_digest="dat",
                       width=1.0, clusters=4.0, lo=-0.5, hi=0.5, verdict="SHOW")
    assert ledger.record(probe).previous is None


def test_the_ceiling_arm_can_make_stage_2_fire(tmp_path):
    """#304 (rule 15): every `_paired` fixture in this file gives every row of a season the
    same `diff` and the same `ceiling`, so the bootstrap has zero width, the MDE is ~0, and
    `experiment.gate`'s own NOT-RUNNABLE branch -- an MDE exceeding the measured ceiling,
    `docs/gate-power.md`'s stage 2 -- can never fire in any of them, guard deleted or not.
    Three seasons (past this module's own `EVENT_SEASONS_MINIMUM` guard) with a `diff` that
    swings widely within and across seasons and a `ceiling` pinned near zero: the bootstrap
    interval is wide, the measured ceiling is tiny, and the MDE exceeds it -- stage 2's own
    sentence, not the count-of-event-seasons one the guard above prints."""
    paired = pl.DataFrame({
        "season": [2026, 2026, 2027, 2027, 2028, 2028],
        "game_id": ["a", "b", "c", "d", "e", "f"],
        "diff": [0.30, 0.28, -0.25, -0.30, 0.05, -0.05],
        "ceiling": [0.001] * 6,
    })
    run = sc.run(paired, needed=3, ledger=Ledger(path=None))
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert "stage 2" in run.verdict[1]
    assert "event-season" not in run.verdict[1]              # not this module's own guard


def test_season_event_summaries_returns_one_typed_record_per_season(rows):
    """KC's two in-season changes (Gabbert in, Mahomes back) and LA's one offseason change
    (flagged, not an event) are all dated season 2026 -- the fixture `SEASON`'s own comment
    names the Rams' change as the offseason one. Cross-checked against `_event_lines`'
    sentence, built from the same inputs, so a divergence between the typed record and the
    printed line cannot land unnoticed."""
    tg = se.team_games(rows)
    ev = se.events(tg)
    games = se.event_games(se.in_season_events(ev))
    got = sc.season_event_summaries(ev, games)
    assert [s.season for s in got] == [2026]
    s = got[0]
    assert isinstance(s, sc.SeasonEventSummary)
    assert s.changes == 2 and s.games == 2 and s.n_gaps == 2 and s.offseason == 1
    line = f"    {s.season}: {s.changes} changes on {s.games} event games, gap sd {s.gap_sd:.1f} " \
           f"value units (n={s.n_gaps}); {s.offseason} offseason change(s) not events"
    assert line in sc._event_lines(ev, games, since=2022)


# --- the entry point ---------------------------------------------------------------------


def test_the_cli_reads_the_cache_and_the_store_and_reports_the_counts(tmp_path, capsys,
                                                                       monkeypatch, rows):
    """Driven end to end on the fixture: the nfeloqb cache under `--cache`, an empty store
    under `--store`. The events are counted per season, the gate reports zero event-seasons
    and NOT-RUNNABLE, and the study reports the censored count -- with n on every line."""
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    monkeypatch.setattr(experiment, "WIDTH_STATE", tmp_path / "w.json")
    code = sc.main(["--events", "--gate", "--study", "--cache", str(cache),
                    "--store", str(tmp_path / "store")])
    out = capsys.readouterr().out
    assert code == 0
    assert "2026: 2 changes on 2 event games" in out
    assert "NOT-RUNNABLE" in out and "0 event-season" in out
    assert "no snapshot archive" in out
    # #329: the study's own verdict, wired into the NOT-RUNNABLE path when it has no rows --
    # `study_fit` on an empty frame hands `verdict` an MDE that is not finite, which is
    # `verdict`'s own NOT-RUNNABLE condition rather than a case special-cased around it.
    assert "coefficient: not established" in out
    assert "NOT-RUNNABLE: NOT RUNNABLE:" in out
    assert "cannot be stated from these inputs" in out


def test_the_cli_ceiling_flag_defaults_on_and_no_ceiling_turns_it_off(tmp_path, monkeypatch,
                                                                        rows):
    """`run()`'s own keyword default is `ceiling=True` -- stage 2 on -- but until #302 the
    CLI's own `--ceiling` was `store_true` and defaulted to False, so the documented
    invocation (`--events --gate --study`, with no `--ceiling`) shipped with stage 2 off no
    matter what `run` itself defaulted to: the tested configuration was never the shipped
    one. `--ceiling` is a `BooleanOptionalAction` now, on unless `--no-ceiling` is given, so
    the flag that reaches `run` is what the usage line's absence of `--ceiling` implies."""
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    seen: list[bool] = []

    def fake_run(paired, *, needed, ceiling=True, ledger=None):
        seen.append(ceiling)
        return experiment.GateRun({}, pl.DataFrame(), ("SHOW", "stub"), [], pl.DataFrame())

    monkeypatch.setattr(sc, "run", fake_run)
    store_path = str(tmp_path / "store")
    assert sc.main(["--gate", "--cache", str(cache), "--store", store_path]) == 0
    assert sc.main(["--gate", "--no-ceiling", "--cache", str(cache), "--store", store_path]) == 0
    assert seen == [True, False]


def test_the_cli_without_a_cache_is_a_sentence_not_a_traceback(tmp_path, capsys):
    code = sc.main(["--gate", "--cache", str(tmp_path / "none"), "--store", str(tmp_path)])
    assert code != 0
    assert "nfeloqb" in capsys.readouterr().err


def test_no_event_at_all_yields_empty_frames_with_the_schema(rows):
    """A season with no change: the readers get the empty frame in the declared shape, not
    a traceback from a group_by over nothing."""
    tg = se.team_games(rows).filter(pl.col("team") == "DEN")
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games.is_empty() and list(games.columns) == list(se.EVENT_GAME_SCHEMA)
    assert sc.gate_rows(ARCHIVE, games, tg, rows).is_empty()
    assert ss.study_rows(ARCHIVE, games, tg).is_empty()
    assert math.isnan(
        ss.study_fit(ss.study_rows(ARCHIVE, games, tg), floor_per_root_day=0.4)["beta"])
    assert math.isnan(ss.study_mde(n=1, sd_gap=70.0, window_days=7.0, floor_per_root_day=0.4))


def test_the_cli_reads_a_store_with_an_archive_and_reports_the_study(tmp_path, capsys,
                                                                       monkeypatch, rows):
    """Driven with the fixture archive written into a store: the archive line, the noise
    floor off it, the pilot, the gate's zero and the study's coefficient with its n. Week 4
    (KC-DEN) has no result yet in this fixture, so #303's fix excludes it from the study --
    only the played KC-LA event reaches `study_rows`, one game rather than two."""
    from hub import store

    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    base = tmp_path / "store"
    lines = ARCHIVE.with_columns(pl.lit("nfl").alias("league"), pl.lit(2026).alias("season"))
    for week, part in lines.group_by("week"):
        store.write(part.drop("week"), "lines", "nfl", 2026, int(week[0]), name="t",
                    base=base)
    monkeypatch.setattr(experiment, "WIDTH_STATE", tmp_path / "w.json")

    # #330: --study reaches for the depth chart the same-quarterback floor conditions on;
    # this fixture has no chart to give it, so it is stubbed exactly the way the unit suite
    # stubs `odds.noise_floor_report`'s own fetch (tests/unit/test_fetch_odds.py) rather than
    # let it reach the network.
    def _no_chart(season):
        raise ConnectionError("no network")
    monkeypatch.setattr(odds, "load_qb_starters", _no_chart)

    # --ceiling is on by default since #302; passed explicitly here only because this test
    # also pins --since to collapse the assembled range to the one season the fixture has.
    code = sc.main(["--gate", "--ceiling", "--study", "--since", "2026", "--cache", str(cache),
                    "--store", str(base)])
    out = capsys.readouterr().out
    assert code == 0
    assert "archive: 3 games" in out
    assert "2026: 2 event games; 0 censored" in out
    assert "gate: 1 scored event games over 1 event-season(s)" in out
    assert "diagnostic, not the gate -- the oracle arm" in out
    assert "NOT-RUNNABLE" in out
    assert "same-quarterback NOT applied for 2026" in out and "ConnectionError" in out
    assert "all-games floor, SAME-QUARTERBACK NOT APPLIED" in out
    assert "1 uncensored, priced event game(s) excluded as unplayed" in out
    assert "coefficient:" in out and "n=1 event games" in out
    # #329: the study's verdict, wired in after the coefficient line -- two rows have almost
    # no power, so the MDE exceeds delta and the branch is NOT-RUNNABLE, never the interval.
    assert "0.132:" in out
    assert "NOT-RUNNABLE: NOT RUNNABLE:" in out and "No branch below is read" in out
    assert "change-point: seen on" in out


def test_the_cli_assembles_every_season_from_since_through_season(tmp_path, capsys,
                                                                    monkeypatch, rows):
    """Until #302 `main` read a single season's archive (`--season` alone) and filtered
    `games` to it, so the gate could never see more than one event-season no matter how many
    the caches held (`docs/gate-power.md`'s three-season floor was unreachable through the
    CLI). `--since` through `--season` is now the range `main` assembles: a second season's
    archive, disjoint from the fixture's 2026 one, folds into the same run's polls."""
    from hub import store

    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    base = tmp_path / "store"
    lines = ARCHIVE.with_columns(pl.lit("nfl").alias("league"), pl.lit(2026).alias("season"))
    for week, part in lines.group_by("week"):
        store.write(part.drop("week"), "lines", "nfl", 2026, int(week[0]), name="t", base=base)
    extra = pl.DataFrame({
        "game_id": ["2025_01_BB_AA"], "close_spread": [1.0],
        "captured_at": [dt.datetime(2025, 9, 1, 16)], "week": [1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64}
    ).with_columns(pl.lit("nfl").alias("league"), pl.lit(2025).alias("season"))
    store.write(extra.drop("week"), "lines", "nfl", 2025, 1, name="t", base=base)
    monkeypatch.setattr(experiment, "WIDTH_STATE", tmp_path / "w.json")
    code = sc.main(["--gate", "--since", "2025", "--season", "2026", "--cache", str(cache),
                    "--store", str(base)])
    out = capsys.readouterr().out
    assert code == 0
    # 3 game ids from the 2026 archive plus the one from 2025's: both seasons folded in.
    assert "archive: 4 games" in out
    assert "2025" in out and "2026" in out


def test_the_cli_study_line_names_the_season_whose_chart_is_not_applied(tmp_path, capsys,
                                                                          monkeypatch, rows):
    """#331: one depth-chart fetch per season in the assembled range, not one for the last
    season handed to every season's polls -- `odds._starter_at`'s as-of join means a chart
    fetched for 2026 cannot condition a 2025 poll, so the old single fetch left 2025's
    intervals with `same_qb` null and silently dropped them as unknown. Two seasons' worth
    of archive, one poll pair each; the chart succeeds for 2026 and fails for 2025, and the
    run line names 2025, not 2026, as the one not applied."""
    from hub import store

    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    base = tmp_path / "store"
    lines = ARCHIVE.with_columns(pl.lit("nfl").alias("league"), pl.lit(2026).alias("season"))
    for week, part in lines.group_by("week"):
        store.write(part.drop("week"), "lines", "nfl", 2026, int(week[0]), name="t", base=base)
    extra = pl.DataFrame({
        "game_id": ["2025_01_BB_AA", "2025_01_BB_AA"], "close_spread": [1.0, 1.5],
        "captured_at": [dt.datetime(2025, 8, 25, 16), dt.datetime(2025, 8, 28, 16)],
        "week": [1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64}
    ).with_columns(pl.lit("nfl").alias("league"), pl.lit(2025).alias("season"))
    store.write(extra.drop("week"), "lines", "nfl", 2025, 1, name="t", base=base)

    def _chart(season):
        if season == 2026:
            return pl.DataFrame({"dt": [dt.datetime(2026, 8, 1)], "team": ["KC"], "qb": ["x"]})
        raise ConnectionError("no chart for 2025")
    monkeypatch.setattr(odds, "load_qb_starters", _chart)
    monkeypatch.setattr(experiment, "WIDTH_STATE", tmp_path / "w.json")

    code = sc.main(["--study", "--since", "2025", "--season", "2026", "--cache", str(cache),
                    "--store", str(base)])
    out = capsys.readouterr().out
    assert code == 0
    assert "same-quarterback NOT applied for 2025" in out
    assert "NOT applied for 2026" not in out
    assert "ConnectionError" in out
    assert "same-quarterback floor, PARTIAL" in out


def test_the_cli_refuses_when_season_is_before_since(tmp_path, capsys, rows):
    """A range with nothing in it -- `--season` behind `--since` -- is refused with the
    reason stated, not silently run on an empty or inverted range."""
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    code = sc.main(["--gate", "--since", "2027", "--season", "2026", "--cache", str(cache),
                    "--store", str(tmp_path / "store")])
    assert code != 0
    err = capsys.readouterr().err
    assert "--since" in err and "--season" in err


def test_the_cli_with_no_reader_asked_for_prints_usage(capsys):
    assert sc.main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_an_event_with_no_lookahead_price_is_censored_and_counted_not_dropped(
        tmp_path, capsys, rows):
    """Control 4: both of the fixture's event games are priced by no poll at all -- the state of
    a 2026 week the back-filled lookahead never reached -- and the run line counts both as
    censored and never polled, rather than printing two event games and no mention of what
    could not be priced. Mutation: `priced` marking `censored` false for an unpolled game
    (or `main` filtering censored games out before counting) leaves this red."""
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    code = sc.main(["--events", "--since", "2026", "--cache", str(cache),
                    "--store", str(tmp_path / "store")])
    out = capsys.readouterr().out
    assert code == 0
    assert "2026: 2 event games; 2 censored (no snapshot before the change could be known: " \
           "2 never polled at all, 0 polled only after the change)" in out
    assert "reconciliation against the pinned file" in out          # the pbp path, not the file's


def test_a_season_played_past_the_source_is_not_established_and_counts_once_it_is(
        tmp_path, capsys):
    """#420's positive control (rule 18): week 4 is played and play-by-play has no pass for
    either side yet. The season's count is printed as NOT ESTABLISHED past the last game day
    with a starter, with the one change seen through it beside it -- not as "1 changes on 1
    event games" as though it were the season's. The same fixture with the horizon advanced
    (week 4's passes present) prints the season's two changes and no such line. Mutation:
    `beyond_event_horizon` returning `{}` leaves the first block red; ignoring `qb` nulls in
    it leaves the second red."""
    played = played_rows(source_rows(PRIOR + SEASON))
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    played.write_csv(cache / nfeloqb.FILE)
    argv = ["--events", "--since", "2026", "--cache", str(cache),
            "--store", str(tmp_path / "store")]

    serve_world(played, drop=[("2026_04_DEN_KC", "KC"), ("2026_04_DEN_KC", "DEN")])
    assert sc.main(argv) == 0
    out = capsys.readouterr().out
    assert "2026: NOT ESTABLISHED past 2026-09-27 -- 2 played team-game(s) after it have no " \
           "observed starter; through it: 1 changes on 1 event games" in out
    assert "horizons 2026: played through 2026-10-04 (week 4); starters observed through " \
           "2026-09-27" in out

    serve_world(played)
    assert sc.main(argv) == 0
    out = capsys.readouterr().out
    assert "2026: 2 changes on 2 event games" in out and "NOT ESTABLISHED" not in out
    assert "starters observed through 2026-10-04" in out


def test_the_study_says_so_when_it_can_only_read_the_pinned_file(tmp_path, capsys, rows):
    """With play-by-play unreachable the study reads the pinned file and says it: the season
    in progress is NOT ESTABLISHED past the file's last game, because a pinned file cannot hold
    a change made after its pin and a count from it is a fact about the file (V2). Mutation:
    removing the `beyond` the fallback returns prints the count with no such sentence."""
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    nflverse.select(replay.Replay({"schedules": schedule_for(rows)}))     # no pbp recorded
    assert sc.main(["--events", "--since", "2026", "--cache", str(cache),
                    "--store", str(tmp_path / "store")]) == 0
    out = capsys.readouterr().out
    assert "play-by-play unavailable (NotRecorded" in out
    assert "2026: NOT ESTABLISHED past 2026-10-04 -- games after it are not in the source" in out
    assert "reconciliation against" not in out


def test_prices_past_the_archives_last_poll_are_not_established_and_are_once_it_reaches(rows):
    """The price side of the same horizon: the fixture's polls end 2026-09-30, and a played
    game on 2026-10-04 cannot have a frozen price from them. Advanced past the last game, no
    season is reported. Mutation: comparing `date` to the *first* poll day instead of the last
    leaves the second assertion red."""
    played_tg = se.team_games(played_rows(rows))
    got = sc.beyond_price_horizon(ARCHIVE, played_tg)
    assert got == {2026: sc.Beyond("2026-09-30", 2)}
    ahead = polls([("2026_04_DEN_KC", 1.0, dt.datetime(2026, 10, 5, 16), 4)])
    assert sc.beyond_price_horizon(ahead, played_tg) == {}
    assert sc.beyond_price_horizon(polls([]), played_tg) == {}


def test_the_study_reads_the_committed_snapshots_and_prices_an_event_off_them(
        tmp_path, capsys, monkeypatch, rows):
    """#383's tree under `state/odds/` is read by the study through `store.lines` with no
    `--store`: two committed captures, one before KC's week-2 game day and one before its
    week-3 game, price the week-3 event off a frozen and a pre-game snapshot -- the run line
    counts the files it read and shows one game priced and one (week 4, no capture) censored.
    Mutation: pointing `archive` at a store with no committed tree leaves "0 file(s)" and the
    event game censored, and this red."""
    from hub import store

    monkeypatch.setattr(store, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(store, "DATA", tmp_path / "processed")
    monkeypatch.setattr(store, "CATALOG", tmp_path / "processed" / "hub.duckdb")
    for when, spread in ((dt.datetime(2026, 9, 17, 12), 3.0), (dt.datetime(2026, 9, 24, 12), -1.0)):
        frame = pl.DataFrame({
            "game_id": ["2026_03_LA_KC"], "close_spread": [spread], "spread_price": [None],
            "close_total": [None], "total_price": [None], "captured_at": [when],
            "polls_unmoved": [0], "unmoved_since": [None]},
            schema={"game_id": pl.Utf8, "close_spread": pl.Float64, "spread_price": pl.Float64,
                    "close_total": pl.Float64, "total_price": pl.Float64,
                    "captured_at": pl.Datetime("us"), "polls_unmoved": pl.Int64,
                    "unmoved_since": pl.Datetime("us")})
        store.write_snapshot(frame, 2026, 3, when)
    cache = tmp_path / "nfeloqb"
    cache.mkdir()
    rows.write_csv(cache / nfeloqb.FILE)
    assert sc.main(["--events", "--since", "2026", "--cache", str(cache)]) == 0
    out = capsys.readouterr().out
    assert "committed snapshots read: 2 file(s)" in out
    assert "2026: 2 event games; 1 censored (no snapshot before the change could be known: " \
           "1 never polled at all, 0 polled only after the change), 1 with a frozen price" in out
