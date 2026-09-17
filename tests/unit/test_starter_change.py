"""Starter-change events and their two readers (#221, #291): the gate on the quarterback
adjustment over a frozen line, and the line-move study.

The event construction is held first, because both readers stand on it: an event is a team
whose starter on one game differs from its starter on its previous game of the same season,
observed off the source's own starter column or the first pass attempt in play-by-play, and
never off the injury report. The gate is held to its pre-registration in
`docs/gate-power.md` -- the frozen price is the comparator, the shipped estimator is the arm,
fewer than three event-seasons is not-runnable -- and the study to its: the week's mean move
is subtracted, the regressor is the net ex-ante gap, the floor sets the standard error.

Every fixture is synthetic and every test runs with no `data/` and no network.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics

import polars as pl
import pytest

from hub.fetch import nfeloqb, odds
from hub.models import experiment, quarterback
from hub.models import starter_change as sc
from hub.models.market import MARGIN_SD, normal_cdf

# --- fixtures ------------------------------------------------------------------------------


def source_rows(games):
    """nfeloqb-shaped rows. Each game: (date, season, week, home, away, home_qb, away_qb,
    home_value, away_value, home_adj, away_adj, home_score | None, away_score | None,
    elo_prob1 | None, qbelo_prob1 | None). Home is team1, the source's convention."""
    cols = ("date", "season", "week", "team1", "team2", "qb1", "qb2", "qb1_value_pre",
            "qb2_value_pre", "qb1_adj", "qb2_adj", "score1", "score2", "elo_prob1",
            "qbelo_prob1")
    frame = pl.DataFrame({c: [g[i] for g in games] for i, c in enumerate(cols)},
                         schema={"date": pl.Utf8, "season": pl.Int64, "week": pl.Utf8,
                                 "team1": pl.Utf8, "team2": pl.Utf8, "qb1": pl.Utf8,
                                 "qb2": pl.Utf8, "qb1_value_pre": pl.Float64,
                                 "qb2_value_pre": pl.Float64, "qb1_adj": pl.Float64,
                                 "qb2_adj": pl.Float64, "score1": pl.Int64,
                                 "score2": pl.Int64, "elo_prob1": pl.Float64,
                                 "qbelo_prob1": pl.Float64})
    return frame.with_columns(pl.lit("REG").alias("game_type"))


def polls(rows):
    """The archive as `hub.fetch.odds._archive` returns it: (game_id, close_spread,
    captured_at, week). `captured_at` is naive UTC."""
    return pl.DataFrame({"game_id": [r[0] for r in rows],
                         "close_spread": [float(r[1]) for r in rows],
                         "captured_at": [r[2] for r in rows],
                         "week": [r[3] for r in rows]},
                        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                "captured_at": pl.Datetime("us"), "week": pl.Int64})


def utc(day, hour=16):
    return dt.datetime(2026, 9, day, hour)


# A season for one team, KC: Mahomes starts weeks 1-2, a backup takes week 3, Mahomes is
# back for week 4. The opponent DEN keeps one starter throughout; the Rams, spelled the
# source's way, arrive in week 3 with a starter who differs from their last 2025 start --
# an offseason change, flagged and not an event.
SEASON = [
    ("2026-09-13", 2026, "1.0", "KC", "DEN", "Mahomes", "Nix", 200.0, 120.0, 40.0, 2.0,
     27, 20, 0.70, 0.75),
    ("2026-09-20", 2026, "2.0", "DEN", "KC", "Nix", "Mahomes", 121.0, 205.0, 2.5, 42.0,
     17, 24, 0.40, 0.35),
    ("2026-09-27", 2026, "3.0", "KC", "LAR", "Gabbert", "Bethard", 60.0, 55.0, -110.0, -90.0,
     10, 20, 0.65, 0.52),
    ("2026-10-04", 2026, "4.0", "KC", "DEN", "Mahomes", "Nix", 199.0, 122.0, 38.0, 3.0,
     None, None, 0.71, 0.76),
]
PRIOR = [("2025-12-28", 2025, "17.0", "KC", "DEN", "Mahomes", "Nix", 210.0, 100.0, 45.0,
          -5.0, 30, 10, 0.8, 0.85),
         ("2025-12-28", 2025, "17.0", "LAR", "SF", "Stafford", "Purdy", 150.0, 160.0, 10.0,
          12.0, 21, 24, 0.5, 0.49)]


@pytest.fixture
def rows():
    return source_rows(PRIOR + SEASON)


# --- the event construction ----------------------------------------------------------------


def test_team_games_key_every_side_by_nflverse_id_and_spelling(rows):
    """One row per (team, game), the id rebuilt in nflverse's spelling from the source's own
    columns -- the source spells the Rams LAR where the archive says LA -- with the side and
    the ex-ante value and adjustment on it."""
    tg = sc.team_games(rows)
    week3 = tg.filter(pl.col("game_id") == "2026_03_LA_KC").sort("team")
    assert week3["team"].to_list() == ["KC", "LA"]
    assert week3["home"].to_list() == [True, False]
    assert week3["qb"].to_list() == ["Gabbert", "Bethard"]
    assert week3["value"].to_list() == [60.0, 55.0]
    assert week3["adj"].to_list() == [-110.0, -90.0]
    assert tg.filter(pl.col("team") == "LAR").is_empty()


# KC's week-2 game (at DEN) is blank on KC's own side only -- the source has no starter for
# them that week, DEN's side is fully populated. Week 1 (Mahomes) and week 3 (Gabbert, a
# genuine change) are both readable, so week 2's hole sits between two known rows.
ONE_SIDED_BLANK = [
    ("2026-09-13", 2026, "1.0", "KC", "DEN", "Mahomes", "Nix", 200.0, 120.0, 40.0, 2.0,
     27, 20, 0.70, 0.75),
    ("2026-09-20", 2026, "2.0", "DEN", "KC", "Nix", None, 121.0, None, 2.5, None,
     17, 24, 0.40, 0.35),
    ("2026-09-27", 2026, "3.0", "KC", "LAR", "Gabbert", "Bethard", 60.0, 55.0, -110.0, -90.0,
     10, 20, 0.65, 0.52),
]


def test_a_one_sided_blank_does_not_move_the_previous_game_link():
    """The previous-game link is built off the full schedule, not off whichever row survives
    the blank: KC's week-3 change reads its ancestor as week 2 (the game a one-sided blank
    dropped), not week 1 -- two games and fourteen days too early, the bug this test used to
    catch when the link was a shift over the filtered rows."""
    rows = source_rows(ONE_SIDED_BLANK)
    tg = sc.team_games(rows)
    assert tg.filter(pl.col("team") == "KC")["week"].to_list() == [1, 3]     # week 2 dropped
    ev = sc.events(tg)
    kc = ev.filter(pl.col("team") == "KC")
    assert kc.height == 1
    row = kc.row(0, named=True)
    assert row["departing"] == "Mahomes" and row["arriving"] == "Gabbert"    # last known starter
    assert row["prev_game_id"] == "2026_02_KC_DEN"                          # week 2, not week 1
    assert row["prev_date"] == "2026-09-20"                                  # not "2026-09-13"
    games = sc.event_games(sc.in_season_events(ev))
    assert games.row(0, named=True)["frozen_before"] == "2026-09-20"
    assert sc.unreadable_games(rows) == 1                                    # KC's week-2 side


# KC 2025 week 18 (Mahomes, known); 2026 week 1 (at DEN) is blank on KC's own side; 2026
# week 2 (Gabbert) is readable. The hole spans the season boundary: #301's link correctly
# names week 1 as the ancestor, but the departing starter's own last known game is week 18
# of 2025, not week 1 -- so this change is unattributable to a season, not an in-season one.
OFFSEASON_BEHIND_BLANK = [
    ("2025-12-28", 2025, "18.0", "KC", "DEN", "Mahomes", "Nix", 210.0, 100.0, 45.0, -5.0,
     30, 10, 0.8, 0.85),
    ("2026-09-06", 2026, "1.0", "DEN", "KC", "Nix", None, 121.0, None, 2.5, None,
     17, 24, 0.40, 0.35),
    ("2026-09-13", 2026, "2.0", "KC", "LAR", "Gabbert", "Bethard", 60.0, 55.0, -110.0, -90.0,
     10, 20, 0.65, 0.52),
]


def test_a_hole_spanning_the_season_boundary_is_not_an_in_season_event():
    """#328: `in_season` used to read the *link's* season (2026, week 1's -- correct as an
    ancestor) instead of the departing starter's own last known game's season (2025, off
    Mahomes' week-18 start), manufacturing an in-season event across the boundary the blank
    week spans. `departing_game_id`/`departing_season` carry that game explicitly, `in_season`
    reads off them, and #301's link (`prev_game_id`) is unchanged."""
    rows = source_rows(OFFSEASON_BEHIND_BLANK)
    tg = sc.team_games(rows)
    ev = sc.events(tg)
    kc = ev.filter(pl.col("team") == "KC")
    assert kc.height == 1
    row = kc.row(0, named=True)
    assert row["departing"] == "Mahomes" and row["arriving"] == "Gabbert"
    assert row["prev_game_id"] == "2026_01_KC_DEN"           # #301's link: week 1, unchanged
    assert row["departing_game_id"] == "2025_18_DEN_KC"      # Mahomes' own last known game
    assert row["departing_season"] == 2025
    assert row["in_season"] is False                         # not week 1's season, 2026
    assert sc.in_season_events(ev).filter(pl.col("team") == "KC").is_empty()


def test_events_are_starter_changes_between_consecutive_games_of_one_season(rows):
    """KC changes twice in 2026 (Gabbert in, Mahomes back); LA once (Bethard, against its last
    2025 start -- an offseason change, flagged and not an event); DEN never. A team's first
    row has nothing to differ from."""
    ev = sc.events(sc.team_games(rows))
    kc = ev.filter(pl.col("team") == "KC").sort("week")
    assert kc["week"].to_list() == [3, 4]
    assert kc["departing"].to_list() == ["Mahomes", "Gabbert"]
    assert kc["arriving"].to_list() == ["Gabbert", "Mahomes"]
    assert kc["prev_game_id"].to_list() == ["2026_02_KC_DEN", "2026_03_LA_KC"]
    assert kc["in_season"].all()
    la = ev.filter(pl.col("team") == "LA")
    assert la.height == 1 and not la["in_season"][0]
    assert ev.filter(pl.col("team") == "DEN").is_empty()
    # The 2025 -> 2026 Mahomes rows are the same starter: no event across the offseason.
    assert ev.filter((pl.col("team") == "KC") & (pl.col("week") == 1)).is_empty()


def test_values_are_ex_ante_for_both_quarterbacks(rows):
    """The departing starter's value is read off his last start, the arriving starter's off
    the event row -- both `qb_value_pre`, neither a post-game figure -- and the gap is the
    arriving minus the departing. The arriving starter's adjustment rides on the row."""
    ev = sc.in_season_events(sc.events(sc.team_games(rows)))
    kc3 = ev.filter((pl.col("team") == "KC") & (pl.col("week") == 3)).row(0, named=True)
    assert kc3["departing_value"] == 205.0        # Mahomes on the week-2 row
    assert kc3["arriving_value"] == 60.0          # Gabbert on the week-3 row
    assert kc3["gap"] == 60.0 - 205.0
    assert kc3["arriving_adj"] == -110.0
    assert kc3["prev_date"] == "2026-09-20" and kc3["date"] == "2026-09-27"


def test_a_game_where_both_sides_changed_is_one_event_game_with_a_net_gap(rows):
    """LV's change would make week 3 a two-change game; here it is LA's offseason change, so
    week 3 is one change and week 4 one change. Built on a frame where both sides change in
    one game, the game is one row and the net gap is home minus away."""
    both = source_rows([
        ("2026-09-13", 2026, "1.0", "A", "B", "a1", "b1", 100.0, 100.0, 0.0, 0.0, 20, 10, .5, .5),
        ("2026-09-20", 2026, "2.0", "A", "B", "a2", "b2", 40.0, 70.0, -60.0, -30.0, 10, 20,
         .5, .5)])
    games = sc.event_games(sc.in_season_events(sc.events(sc.team_games(both))))
    assert games.height == 1
    row = games.row(0, named=True)
    assert row["changes"] == 2
    assert row["home_gap"] == -60.0 and row["away_gap"] == -30.0
    assert row["net_gap"] == -30.0
    assert row["frozen_before"] == "2026-09-13"


def test_the_first_pass_attempt_names_the_starter_in_play_by_play():
    """The passer on the earliest pass play of each (game, team) by play id -- a run-first
    drive does not name him and a later relief appearance does not replace him. Without the
    passer column the reader refuses by name rather than guessing off a column it has."""
    pbp = pl.DataFrame({
        "game_id": ["g1"] * 5, "season": [2026] * 5, "week": [1] * 5,
        "play_id": [10, 20, 30, 40, 50],
        "posteam": ["A", "A", "B", "A", "B"],
        "play_type": ["run", "pass", "pass", "pass", "run"],
        "passer_player_id": [None, "a-starter", "b-starter", "a-backup", None]})
    got = sc.starters_from_pbp(pbp).sort("team")
    assert got["qb"].to_list() == ["a-starter", "b-starter"]
    assert got["game_id"].to_list() == ["g1", "g1"]
    with pytest.raises(ValueError, match="passer_player_id"):
        sc.starters_from_pbp(pbp.drop("passer_player_id"))


# --- the gate ----------------------------------------------------------------------------


ARCHIVE = polls([
    # KC-LA (week 3): a lookahead frozen at +3 through the week-2 game day, then repriced
    # after Gabbert is named. Game day 2026-09-27; the previous KC game day 2026-09-20.
    ("2026_03_LA_KC", 3.0, utc(15), 3), ("2026_03_LA_KC", 3.0, utc(19), 3),
    ("2026_03_LA_KC", 3.0, utc(20, 12), 3),    # still the week-2 game day: not before it
    ("2026_03_LA_KC", -1.0, utc(23), 3), ("2026_03_LA_KC", -2.0, utc(26), 3),
    # KC-DEN (week 4): Mahomes back; frozen before 2026-09-27, close before 2026-10-04.
    ("2026_04_DEN_KC", -3.0, utc(24), 4), ("2026_04_DEN_KC", 4.0, utc(30), 4),
    # Another week-3 game, no change: the week's mean move between the same two poll days.
    ("2026_03_SF_DEN", 1.0, utc(19), 3), ("2026_03_SF_DEN", 2.0, utc(26), 3),
])


def test_the_frozen_price_predates_the_previous_game_day_and_the_close_the_game_day(rows):
    """Strictly before, both: a poll on the previous game day could already carry an
    in-game injury, and a poll on the game day is not a lookahead at all."""
    games = sc.event_games(sc.in_season_events(sc.events(sc.team_games(rows))))
    priced = sc.priced(ARCHIVE, games).sort("week")
    assert priced["frozen"].to_list() == [3.0, -3.0]
    assert priced["frozen_at"].to_list() == [utc(19), utc(24)]
    assert priced["close"].to_list() == [-2.0, 4.0]


def test_an_event_with_no_snapshot_before_its_previous_game_day_is_censored(rows):
    """Seen only after it moved is not seen at all: the row is kept, marked and counted, so
    the run says how many events the archive could not have caught."""
    late = ARCHIVE.filter(pl.col("captured_at") >= utc(23))
    games = sc.event_games(sc.in_season_events(sc.events(sc.team_games(rows))))
    priced = sc.priced(late, games).sort("week")
    assert priced["censored"].to_list() == [True, False]
    assert priced.filter(pl.col("censored"))["frozen"].is_null().all()


def test_the_arm_is_the_shipped_seam_and_prices_the_departing_starter(rows):
    """The gate's arm is `nfeloqb.state(rows, as_of=<the week's first game day>)` handed to
    `quarterback.apply` -- what `ratings._rated_by_week` does for a played week. Rows
    strictly before 2026-09-27 hold KC's week-2 row (Mahomes, +42) and LA's last 2025 row
    (Stafford, +10), so the frozen +3 moves by (42 - 10) / 25: the *departing* starter's
    adjustment, because the file carries Gabbert on no row before his first start. Both
    arms score the home result by `MarketBaseline`'s conversion; the difference is
    unadjusted minus adjusted, positive when the adjustment helped."""
    tg = sc.team_games(rows)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
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
    tg = sc.team_games(rows)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
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
    tg = sc.team_games(rows_)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
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
    run = sc.run(paired, needed=3, width_path=tmp_path / "w.json")
    assert run.verdict[0] == "REMOVE"


def test_the_pilot_reads_the_sources_own_two_columns_on_event_games_only(rows):
    """The power input: the source's base and quarterback-adjusted probabilities, scored on
    the same event games and nowhere else, per season; the target is the absolute mean of the
    season means and `s` their spread. One season has a mean and no spread."""
    tg = sc.team_games(rows)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
    pilot = sc.pilot(tg, games)
    assert pilot["seasons"] == 1 and pilot["n"] == 1        # week 4 is unplayed
    ll = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))  # noqa: E731
    assert pilot["target"] == pytest.approx(abs(ll(0.65, 0.0) - ll(0.52, 0.0)))
    assert math.isnan(pilot["season_sd"])


def test_event_seasons_needed_is_the_smallest_k_whose_mde_clears_the_target():
    """On the t reference: at s = 1 the MDE is 9.58 s at two seasons, 2.97 at three, 2.01 at
    four -- the table in the pre-registration -- so a target of 2.5 needs four and a target
    of 10 needs two. A target no cap reaches is None, not a number."""
    assert sc.mde_at(2, 1.0) == pytest.approx(9.58, abs=0.01)
    assert sc.mde_at(3, 1.0) == pytest.approx(2.97, abs=0.01)
    assert sc.mde_at(4, 1.0) == pytest.approx(2.01, abs=0.01)
    assert sc.event_seasons_needed(2.5, 1.0) == 4
    assert sc.event_seasons_needed(10.0, 1.0) == 2
    assert sc.event_seasons_needed(0.0001, 1.0, cap=50) is None
    assert sc.mde_at(4, 1.0) == pytest.approx(
        experiment.minimum_detectable_effect(1.0 / math.sqrt(4), 4))


def _paired(seasons, n=6, diff=0.05):
    return pl.DataFrame({"season": [s for s in seasons for _ in range(n)],
                         "diff": [diff] * (n * len(seasons)),
                         "ceiling": [0.5] * (n * len(seasons))})


def test_fewer_than_three_event_seasons_is_not_runnable_and_names_the_count_needed(tmp_path):
    """The first pre-registered precondition, since #300 an exemption from this diagnostic
    firing rather than a bar to the module's ADOPT condition (#221's coefficient). The house
    rule is never read: a frame that would ADOPT at three seasons is NOT-RUNNABLE at two, and
    the sentence carries the event-season count the pilot says is needed."""
    run = sc.run(_paired([2026, 2027]), needed=31, width_path=tmp_path / "w.json")
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert "2 event-season" in run.verdict[1] and "31" in run.verdict[1]
    assert "exemption" in run.verdict[1] and "#221" in run.verdict[1]
    enough = sc.run(_paired([2026, 2027, 2028]), needed=3, width_path=tmp_path / "w.json")
    assert enough.verdict[0] == "ADOPT"


def test_the_verdict_names_itself_a_diagnostic_on_every_branch(tmp_path):
    """Amended 2026-09-17 (#300): none of the gate's own branches license ADOPT or pull the
    module -- #221's line-move coefficient does, and this gate is read beside it. The three
    sentences say so, on ADOPT as much as on SHOW or REMOVE."""
    mixed = pl.concat([_paired([2026, 2027], diff=0.05), _paired([2028], diff=-0.05)])
    show = sc.run(mixed, needed=3, width_path=tmp_path / "w.json")
    assert show.verdict[0] == "SHOW"
    assert "Diagnostic only" in show.verdict[1] and "#221" in show.verdict[1]

    adopt = sc.run(_paired([2026, 2027, 2028]), needed=3, width_path=tmp_path / "w.json")
    assert adopt.verdict[0] == "ADOPT"
    assert "Diagnostic only" in adopt.verdict[1] and "#221" in adopt.verdict[1]

    remove = sc.run(_paired([2026, 2027, 2028], diff=-0.05), needed=3,
                    width_path=tmp_path / "w.json")
    assert remove.verdict[0] == "REMOVE"
    assert "Diagnostic only" in remove.verdict[1] and "#221" in remove.verdict[1]


def test_no_rows_is_not_runnable_with_zero_event_seasons(tmp_path):
    run = sc.run(pl.DataFrame(schema={"season": pl.Int64, "diff": pl.Float64}), needed=None,
                 width_path=tmp_path / "w.json")
    assert run.verdict[0] == "NOT-RUNNABLE" and "0 event-season" in run.verdict[1]


def test_not_runnable_publishes_no_interval_and_records_no_width(tmp_path):
    """The three-season floor is applied before the summary, the house verdict, the width
    history and the rendered lines -- not applied to the house verdict alone after they have
    already run. An underpowered run's `lines` are empty (no CI, no MDE, no ceiling check,
    no stamp) and nothing is written to the width file even though this call asks to record
    one, because a run that publishes no interval has no width to keep."""
    width_path = tmp_path / "w.json"
    run = sc.run(_paired([2026, 2027]), needed=31, width_path=width_path, record_width=True)
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert run.lines == []
    assert not width_path.exists()


# --- the study ---------------------------------------------------------------------------


def test_the_study_subtracts_the_weeks_mean_move_and_regresses_on_the_net_gap(rows):
    """Week 3: KC-LA moved 3 -> -2 (-5), the other week-3 game moved 1 -> 2 (+1) over the
    same two poll days, so the week-adjusted move is -6 on a net gap of -145 (KC's change
    alone; LA's is offseason and not an event). Week 4: -3 -> +4 with no other game to
    subtract, on a net gap of +139."""
    tg = sc.team_games(rows)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
    study = sc.study_rows(ARCHIVE, games).sort("week")
    assert study["move"].to_list() == [-5.0, 7.0]
    assert study["week_mean"].to_list() == [1.0, 0.0]
    assert study["adjusted_move"].to_list() == [-6.0, 7.0]
    assert study["net_gap"].to_list() == [-145.0, 139.0]


def test_the_change_point_is_the_first_poll_day_past_the_floor(rows):
    """Days from the previous game day to the first poll whose move from the frozen price
    clears `floor_per_root_day * sqrt(days since the frozen poll)`. KC-LA's first poll after
    the frozen one (09-19) is 09-23 at -1: a move of 4 over 4 days clears 0.4 * 2 = 0.8, so
    the change is seen 3 days after the 09-20 game day. A game whose polls never clear the
    floor has no change-point."""
    tg = sc.team_games(rows)
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
    study = sc.study_rows(ARCHIVE, games, floor_per_root_day=0.4).sort("week")
    assert study["days_to_change"].to_list() == [3.0, 3.0]
    quiet = sc.study_rows(ARCHIVE, games, floor_per_root_day=5.0).sort("week")
    assert quiet["days_to_change"].is_null().all()


def test_the_fit_recovers_the_slope_and_takes_its_error_from_the_floor():
    """Moves manufactured at 0.132 points per unit of gap plus a week effect; after the
    week's mean is subtracted the slope is the benchmark, and the standard error is the
    floor per window over the gap's spread and root n, not the residual's."""
    gaps = [-150.0, -80.0, -20.0, 30.0, 90.0, 140.0]
    rows_ = pl.DataFrame({
        "season": [2026] * 6, "week": [1, 1, 2, 2, 3, 3], "game_id": [f"g{i}" for i in range(6)],
        "net_gap": gaps, "adjusted_move": [0.132 * g for g in gaps],
        "window_days": [7.0] * 6})
    fit = sc.study_fit(rows_, floor_per_root_day=0.4)
    assert fit["beta"] == pytest.approx(0.132)
    assert fit["n"] == 6
    floor_window = 0.4 * math.sqrt(7.0)
    sd_gap = statistics.stdev(gaps)
    assert fit["se"] == pytest.approx(floor_window / (sd_gap * math.sqrt(6)))
    assert fit["mde"] == pytest.approx(
        (experiment.t_quantile(0.975, 5) + 0.8416) * fit["se"], abs=1e-3)
    assert fit["benchmark"] == sc.BENCHMARK == pytest.approx(3.3 / 25)
    assert fit["t_vs_benchmark"] == pytest.approx(0.0)


def test_the_study_mde_before_the_run_is_stated_from_the_events_gap_spread():
    """With no archived event the MDE line is still stated: the pinned file's gap spread,
    the noise floor per window, and the event count a season carries."""
    line = sc.study_mde(n=53, sd_gap=70.0, window_days=7.0, floor_per_root_day=0.4)
    assert line == pytest.approx((experiment.t_quantile(0.975, 52) + 0.8416)
                                 * 0.4 * math.sqrt(7.0) / (70.0 * math.sqrt(53)), abs=1e-4)


def test_the_noise_floor_excludes_a_change_the_chart_dates_between_the_polls():
    """#330: `starters` is the depth-chart frame `odds._qb_starters` returns -- `dt` the
    chart's own timestamp, published day by day through the week -- and never this module's
    own team-game rows, whose `date` is *kickoff*. A real archive's polls of a game all
    precede that game's own kickoff, so a change dated at kickoff is invisible to every one
    of them and a starters frame built off kickoff dates can never exclude the interval it
    exists to exclude -- which is exactly the shape the superseded fixture missed, dating a
    team's *other* game between two polls of this one, a shape no real archive has.

    Same three pre-kickoff polls of DAL-PHI, KC-LAC and SF-SEA throughout: a chart entry
    dated *between* the DAL-PHI polls excludes that interval (KC-LAC and SF-SEA are the two
    left); the identical chart dating the same change *at* kickoff -- after both polls --
    does not, because neither poll's as-of lookup ever sees it."""
    g1, g2, g3 = "2026_01_DAL_PHI", "2026_01_KC_LAC", "2026_01_SF_SEA"
    tue = dt.datetime(2026, 8, 25, 4, 58)
    thu = dt.datetime(2026, 8, 28, 2, 1)
    kickoff = dt.datetime(2026, 9, 13, 17, 0)                    # well after both polls
    polls_ = pl.DataFrame({
        "game_id": [g1, g1, g2, g2, g3, g3],
        "close_spread": [-3.0, -3.5, -3.0, -3.5, 1.0, 1.5],
        "captured_at": [tue, thu, tue, thu, tue, thu],
        "week": [1, 1, 1, 1, 1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    base = [(dt.datetime(2026, 8, 1), "PHI", "qb-phi"), (dt.datetime(2026, 8, 1), "DAL", "qb-dal-1"),
            (dt.datetime(2026, 8, 1), "KC", "qb-kc"), (dt.datetime(2026, 8, 1), "LAC", "qb-lac"),
            (dt.datetime(2026, 8, 1), "SF", "qb-sf"), (dt.datetime(2026, 8, 1), "SEA", "qb-sea")]

    def _chart(extra):
        rows_ = [*base, extra]
        return pl.DataFrame({"dt": [r[0] for r in rows_], "team": [r[1] for r in rows_],
                             "qb": [r[2] for r in rows_]})

    mid_week = _chart((dt.datetime(2026, 8, 26), "DAL", "qb-dal-2"))
    floor, floor_games, not_applied = sc.noise_floor_per_root_day([(2026, polls_, mid_week)])
    assert not_applied == ()
    assert floor_games == 2 and math.isfinite(floor)      # KC-LAC and SF-SEA survive

    at_kickoff = _chart((kickoff, "DAL", "qb-dal-2"))
    floor2, floor_games2, not_applied2 = sc.noise_floor_per_root_day([(2026, polls_, at_kickoff)])
    assert not_applied2 == ()
    assert floor_games2 == 3 and math.isfinite(floor2)    # neither poll sees the change


def test_the_noise_floor_with_no_starters_names_the_season_not_applied():
    """The degradation `noise_floor_per_root_day` shares with `noise_floor_report`: with no
    chart for a season, that season's live intervals are still counted -- not dropped as
    unknown -- and the season is named in the returned `not_applied` tuple rather than
    letting a caller print an unconditioned number under the same-quarterback label."""
    polls_ = pl.DataFrame({
        "game_id": ["2026_01_DAL_PHI", "2026_01_DAL_PHI"],
        "close_spread": [-3.0, -3.5],
        "captured_at": [dt.datetime(2026, 8, 25, 4, 58), dt.datetime(2026, 8, 28, 2, 1)],
        "week": [1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    _floor, floor_games, not_applied = sc.noise_floor_per_root_day([(2026, polls_, None)])
    assert not_applied == (2026,)
    assert floor_games == 1


def test_the_noise_floor_covers_every_season_with_its_own_chart():
    """#331: `odds._starter_at` is a backward as-of join, so a chart fetched for one season
    cannot condition another season's polls -- fetching once for the last season only left
    every earlier season's intervals with `same_qb` null, dropped as unknown, and the run
    line still called the result same-quarterback. One `(season, polls, starters)` triple
    per season: with a chart for both 2025 and 2026, both seasons' games reach `floor_games`
    -- neither silently drops out for lacking the other season's chart; with a chart for
    2026 only, 2025's game is still counted (unconditioned rather than dropped) and 2025 is
    the one named in `not_applied`."""
    g2025, g2026_stable, g2026_change = "2025_01_AA_BB", "2026_01_KC_LAC", "2026_01_DAL_PHI"
    tue, thu = dt.datetime(2026, 8, 25, 4, 58), dt.datetime(2026, 8, 28, 2, 1)
    polls_2025 = pl.DataFrame({
        "game_id": [g2025, g2025], "close_spread": [1.0, 1.5],
        "captured_at": [dt.datetime(2025, 8, 25, 4, 58), dt.datetime(2025, 8, 28, 2, 1)],
        "week": [1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    polls_2026 = pl.DataFrame({
        "game_id": [g2026_stable, g2026_stable, g2026_change, g2026_change],
        "close_spread": [-3.0, -3.5, 2.0, 2.5], "captured_at": [tue, thu, tue, thu],
        "week": [1, 1, 1, 1]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime("us"), "week": pl.Int64})
    chart_2025 = pl.DataFrame({
        "dt": [dt.datetime(2025, 8, 1), dt.datetime(2025, 8, 1)], "team": ["AA", "BB"],
        "qb": ["qb-aa", "qb-bb"]})
    chart_2026 = pl.DataFrame({
        "dt": [dt.datetime(2026, 8, 1)] * 4 + [dt.datetime(2026, 8, 26)],
        "team": ["KC", "LAC", "PHI", "DAL", "DAL"],
        "qb": ["qb-kc", "qb-lac", "qb-phi", "qb-dal-1", "qb-dal-2"]})

    both = sc.noise_floor_per_root_day(
        [(2025, polls_2025, chart_2025), (2026, polls_2026, chart_2026)])
    floor, floor_games, not_applied = both
    assert not_applied == ()
    assert floor_games == 2 and math.isfinite(floor)      # 2025's game and KC-LAC both count

    one = sc.noise_floor_per_root_day(
        [(2025, polls_2025, None), (2026, polls_2026, chart_2026)])
    floor2, floor_games2, not_applied2 = one
    assert not_applied2 == (2025,)
    assert floor_games2 == 2 and math.isfinite(floor2)    # 2025's game still counted


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

    def fake_run(paired, *, needed, ceiling=True, width_path=None, record_width=False):
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


def test_events_found_in_play_by_play_price_off_the_pinned_file(rows):
    """The other starter source: a change identified by passer id joins the source's values
    by (team, game) and by (team, previous game), so the gap is the pinned file's whichever
    source named the change."""
    tg = sc.team_games(rows)
    starters = tg.select("game_id", "season", "week", "team").with_columns(
        pl.when((pl.col("team") == "KC") & (pl.col("week") == 3)).then(pl.lit("00-gabbert"))
          .when(pl.col("team") == "KC").then(pl.lit("00-mahomes"))
          .otherwise(pl.lit("00-other")).alias("qb"))
    ev = sc.with_values(sc.in_season_events(sc.events(starters)), tg)
    kc = ev.sort("week")
    assert kc["departing"].to_list() == ["00-mahomes", "00-gabbert"]
    assert kc["gap"].to_list() == [60.0 - 205.0, 199.0 - 60.0]
    assert kc["prev_date"].to_list() == ["2026-09-20", "2026-09-27"]


def test_no_event_at_all_yields_empty_frames_with_the_schema(rows):
    """A season with no change: the readers get the empty frame in the declared shape, not
    a traceback from a group_by over nothing."""
    tg = sc.team_games(rows).filter(pl.col("team") == "DEN")
    games = sc.event_games(sc.in_season_events(sc.events(tg)))
    assert games.is_empty() and list(games.columns) == list(sc.EVENT_GAME_SCHEMA)
    assert sc.gate_rows(ARCHIVE, games, tg, rows).is_empty()
    assert sc.study_rows(ARCHIVE, games).is_empty()
    assert math.isnan(sc.study_fit(sc.study_rows(ARCHIVE, games), floor_per_root_day=0.4)["beta"])
    assert math.isnan(sc.study_mde(n=1, sd_gap=70.0, window_days=7.0, floor_per_root_day=0.4))


def test_rows_without_a_week_are_refused_by_name():
    with pytest.raises(ValueError, match="week"):
        sc.team_games(source_rows(SEASON).drop("week"))


def test_the_cli_reads_a_store_with_an_archive_and_reports_the_study(tmp_path, capsys,
                                                                       monkeypatch, rows):
    """Driven with the fixture archive written into a store: the archive line, the noise
    floor off it, the pilot, the gate's zero and the study's coefficient with its n."""
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
    monkeypatch.setattr(odds, "_qb_starters", _no_chart)

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
    assert "coefficient:" in out and "n=2 event games" in out
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
    monkeypatch.setattr(odds, "_qb_starters", _chart)
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
