"""Starter-change events (#221, #291, #346): the one event set the gate and the study both read.

An event is a team whose starter on one game differs from its starter on its previous game of
the same season, observed off the source's own starter column or the first pass attempt in
play-by-play, and never off the injury report. This file holds the construction, the
play-by-play source and its reconciliation against the pinned file, and the two archive reads
(`priced`, `results`) both readers take. The gate's tests are in `test_starter_change.py`, the
study's in `test_starter_study.py`.

Every fixture is synthetic and every test runs with no `data/` and no network.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from starter_world import (
    ARCHIVE,
    PRIOR,
    SEASON,
    pbp_events,
    pbp_for,
    polls,
    schedule_for,
    serve_world,
    source_rows,
    utc,
)

from hub.fetch import nfeloqb, replay
from hub.models import starter_events as se


@pytest.fixture
def rows():
    return source_rows(PRIOR + SEASON)


@pytest.fixture(autouse=True)
def _the_events_source_is_served_offline(rows):
    """Every test here that drives `main` reads its events through `nflverse.load`, which on
    the network adapter would leave the machine; the offline suite fails a test that tries.
    The default world is the fixture's own rows read as play-by-play."""
    serve_world(rows)




# --- the event construction ----------------------------------------------------------------


def test_team_games_key_every_side_by_nflverse_id_and_spelling(rows):
    """One row per (team, game), the id rebuilt in nflverse's spelling from the source's own
    columns -- the source spells the Rams LAR where the archive says LA -- with the side and
    the ex-ante value and adjustment on it."""
    tg = se.team_games(rows)
    week3 = tg.filter(pl.col("game_id") == "2026_03_LA_KC").sort("team")
    assert week3["team"].to_list() == ["KC", "LA"]
    assert week3["home"].to_list() == [True, False]
    assert week3["qb"].to_list() == ["Gabbert", "Bethard"]
    assert week3["value"].to_list() == [60.0, 55.0]
    assert week3["adj"].to_list() == [-110.0, -90.0]
    assert tg.filter(pl.col("team") == "LAR").is_empty()


POSTSEASON_AND_REGULAR = [
    ("2026-09-06", 2026, "1.0", "A", "B", "a1", "b1", 100.0, 90.0, 5.0, 3.0, 20, 10, .55, .58),
    ("2027-01-10", 2026, "19.0", "A", "B", "a2", "b2", 120.0, 95.0, 6.0, 3.5, 24, 17, .60, .62),
]


def test_team_games_drops_postseason_rows_from_itself_and_the_schedule():
    """#304 (rule 15): `source_rows` hard-codes `game_type='REG'` on every row, so every other
    fixture in this file has nothing to drop and the postseason filter -- in `se.team_games`,
    which reads `nfeloqb.team_games(..., regular_season_only=True)` (#374), and in
    `nfeloqb.schedule`, the full row set the previous-game link is built off -- could be
    deleted with nothing to notice. Week 19 here is marked 'POST'; both readers must drop it
    entirely, not merely fail to count it as an event."""
    rows_ = source_rows(POSTSEASON_AND_REGULAR).with_columns(
        pl.when(pl.col("week") == "19.0").then(pl.lit("POST")).otherwise(pl.lit("REG"))
          .alias("game_type"))
    tg = se.team_games(rows_)
    assert tg["week"].to_list() == [1, 1]                       # both sides of week 1 only
    assert 19 not in tg["week"].to_list()
    sched = nfeloqb.schedule(rows_, regular_season_only=True)
    assert sched["week"].to_list() == [1, 1]
    assert 19 not in sched["week"].to_list()


# Same starter, week 1 (REG) and week 19 (POST): the two readers of #374's shared frame
# legitimately disagree about whether the second row exists at all.
PLAYOFF_TENURE = [
    ("2026-09-06", 2026, "1.0", "A", "B", "a1", "b1", 100.0, 90.0, 5.0, 3.0, 20, 10, .55, .58),
    ("2027-01-10", 2026, "19.0", "A", "B", "a1", "b1", 102.0, 92.0, 6.0, 3.5, 24, 17, .60, .62),
]


def test_tenure_counts_a_playoff_start_the_event_construction_excludes():
    """#374: `nfeloqb.state` does not filter to the regular season -- a team's tenure run
    reaches back across the postseason boundary, because a Super Bowl participant's latest
    row is its playoff game and not its last regular-season one -- while this module's event
    construction reads `nfeloqb.team_games(..., regular_season_only=True)`, because the
    line-move study is regular-season by pre-registration. Same starter both games, so the
    only thing that can be disagreeing is whether the postseason row is read at all: it is,
    for tenure, and it is not, for the event construction, which never sees a second row for
    A to differ against."""
    rows_ = source_rows(PLAYOFF_TENURE).with_columns(
        pl.when(pl.col("week") == "19.0").then(pl.lit("POST")).otherwise(pl.lit("REG"))
          .alias("game_type"))

    st = nfeloqb.state(rows_)
    a = st.filter(pl.col("team") == "A").row(0, named=True)
    assert a["qb"] == "a1" and a["tenure"] == 2, "both starts, the playoff one included"
    assert a["as_of"] == "2027-01-10", "the playoff row is the latest"

    tg = se.team_games(rows_)
    assert tg.filter(pl.col("team") == "A")["week"].to_list() == [1], "week 19 never arrives"
    assert se.events(tg).filter(pl.col("team") == "A").is_empty(), "nothing to diff it against"


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
    tg = se.team_games(rows)
    assert tg.filter(pl.col("team") == "KC")["week"].to_list() == [1, 3]     # week 2 dropped
    ev = se.events(tg)
    kc = ev.filter(pl.col("team") == "KC")
    assert kc.height == 1
    row = kc.row(0, named=True)
    assert row["departing"] == "Mahomes" and row["arriving"] == "Gabbert"    # last known starter
    assert row["prev_game_id"] == "2026_02_KC_DEN"                          # week 2, not week 1
    assert row["prev_date"] == "2026-09-20"                                  # not "2026-09-13"
    games = se.event_games(se.in_season_events(ev))
    assert games.row(0, named=True)["frozen_before"] == "2026-09-20"
    assert se.unreadable_games(rows) == 1                                    # KC's week-2 side


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
    tg = se.team_games(rows)
    ev = se.events(tg)
    kc = ev.filter(pl.col("team") == "KC")
    assert kc.height == 1
    row = kc.row(0, named=True)
    assert row["departing"] == "Mahomes" and row["arriving"] == "Gabbert"
    assert row["prev_game_id"] == "2026_01_KC_DEN"           # #301's link: week 1, unchanged
    assert row["departing_game_id"] == "2025_18_DEN_KC"      # Mahomes' own last known game
    assert row["departing_season"] == 2025
    assert row["in_season"] is False                         # not week 1's season, 2026
    assert se.in_season_events(ev).filter(pl.col("team") == "KC").is_empty()


def test_events_are_starter_changes_between_consecutive_games_of_one_season(rows):
    """KC changes twice in 2026 (Gabbert in, Mahomes back); LA once (Bethard, against its last
    2025 start -- an offseason change, flagged and not an event); DEN never. A team's first
    row has nothing to differ from."""
    ev = se.events(se.team_games(rows))
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


def test_events_sorts_before_shifting_so_row_order_never_matters():
    """#304 (rule 15): every caller in this file feeds `events()` rows that already end
    sorted (off `team_games`, itself sorted, or a fixture built in order), so deleting the
    function's own `.sort("team", "season", "week")` is invisible to the whole suite. The
    same three games fed out of order must still find the one real change -- game 3, where
    the starter differs from the team's second game -- and not a spurious one manufactured
    from whichever row happened to arrive first."""
    ordered = pl.DataFrame({
        "game_id": ["g1", "g2", "g3"], "season": [2026, 2026, 2026], "week": [1, 2, 3],
        "team": ["T", "T", "T"], "qb": ["q1", "q1", "q2"]})
    shuffled = pl.DataFrame({
        "game_id": ["g3", "g1", "g2"], "season": [2026, 2026, 2026], "week": [3, 1, 2],
        "team": ["T", "T", "T"], "qb": ["q2", "q1", "q1"]})
    got = se.events(shuffled)
    assert got.height == 1
    row = got.row(0, named=True)
    assert row["game_id"] == "g3"
    assert row["departing"] == "q1" and row["arriving"] == "q2"
    assert got.equals(se.events(ordered))


def test_values_are_ex_ante_for_both_quarterbacks(rows):
    """The departing starter's value is read off his last start, the arriving starter's off
    the event row -- both `qb_value_pre`, neither a post-game figure -- and the gap is the
    arriving minus the departing. The arriving starter's adjustment rides on the row."""
    ev = se.in_season_events(se.events(se.team_games(rows)))
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
    games = se.event_games(se.in_season_events(se.events(se.team_games(both))))
    assert games.height == 1
    row = games.row(0, named=True)
    assert row["changes"] == 2
    assert row["home_gap"] == -60.0 and row["away_gap"] == -30.0
    assert row["net_gap"] == -30.0
    assert row["frozen_before"] == "2026-09-13"


TWO_SIDED_CHANGE_DIFFERENT_PREVIOUS_GAMES = [
    ("2026-09-06", 2026, "1.0", "A", "X", "a1", "x1", 100.0, 90.0, 5.0, 3.0, 20, 10, .55, .58),
    ("2026-09-08", 2026, "1.0", "B", "Y", "b1", "y1", 100.0, 90.0, 5.0, 3.0, 20, 10, .55, .58),
    ("2026-09-13", 2026, "2.0", "A", "B", "a2", "b2", 150.0, 160.0, 40.0, 45.0, 17, 24, .60, .65),
]


def test_frozen_before_is_the_earlier_of_two_different_previous_game_days():
    """#304 (rule 15): the suite's only other two-changed-side fixture, above, has both sides
    sharing the *same* previous game (A and B played each other in week 1), so `.min()` and
    `.max()` over `prev_date` agree there and a swap to `.max()` is invisible. Here A's
    previous game (against X, 09-06) is two days earlier than B's (against Y, 09-08), so
    `frozen_before` -- the earliest previous game day, before which no change could have been
    known -- must be A's 09-06, not B's 09-08."""
    rows_ = source_rows(TWO_SIDED_CHANGE_DIFFERENT_PREVIOUS_GAMES)
    tg = se.team_games(rows_)
    games = se.event_games(se.in_season_events(se.events(tg)))
    assert games.height == 1
    row = games.row(0, named=True)
    assert row["changes"] == 2
    assert row["frozen_before"] == "2026-09-06"                 # A's, the earlier of the two


def test_the_first_pass_attempt_names_the_starter_in_play_by_play():
    """The passer on the earliest pass play of each (game, team) by play id -- a run-first
    drive does not name him and a later relief appearance does not replace him. Without the
    passer column the reader refuses by name rather than guessing off a column it has."""
    pbp = pl.DataFrame({
        "game_id": ["g1"] * 5 + ["g1p"], "season": [2026] * 6, "week": [1] * 5 + [19],
        "season_type": ["REG"] * 5 + ["POST"],
        "play_id": [10, 20, 30, 40, 50, 60],
        "posteam": ["A", "A", "B", "A", "B", "A"],
        "play_type": ["run", "pass", "pass", "pass", "run", "pass"],
        "passer_player_id": [None, "a-starter", "b-starter", "a-backup", None, "a-playoff"]})
    got = se.starters_from_pbp(pbp).sort("team")
    # the postseason game names no starter: this is a regular-season question (#420)
    assert got["qb"].to_list() == ["a-starter", "b-starter"]
    assert got["game_id"].to_list() == ["g1", "g1"]
    with pytest.raises(ValueError, match="passer_player_id"):
        se.starters_from_pbp(pbp.drop("passer_player_id"))
    with pytest.raises(ValueError, match="season_type"):
        se.starters_from_pbp(pbp.drop("season_type"))


def test_poll_day_converts_the_utc_capture_to_its_eastern_calendar_date():
    """#304 (rule 15): `_last_before` and `_change_points` both read `_poll_day` through
    several layers of joins and filters, so a test built through them can kill the
    Eastern-conversion mutant without ever proving the function itself is what does it -- and
    the archive fixtures above all capture in the afternoon UTC, where the Eastern date
    already happens to agree with the UTC one. A capture at 2026-01-01T03:00 UTC is
    2025-12-31, 22:00 Eastern (EST is UTC-5) -- the previous calendar date in the zone every
    date comparison in this module goes through -- so the poll day is '2025-12-31', never the
    UTC date '2026-01-01'."""
    frame = pl.DataFrame({"captured_at": [dt.datetime(2026, 1, 1, 3, 0)]},
                         schema={"captured_at": pl.Datetime("us")})
    got = frame.select(se.poll_day().alias("poll_day"))
    assert got["poll_day"].to_list() == ["2025-12-31"]


def test_the_frozen_price_predates_the_previous_game_day_and_the_close_the_game_day(rows):
    """Strictly before, both: a poll on the previous game day could already carry an
    in-game injury, and a poll on the game day is not a lookahead at all."""
    games = se.event_games(se.in_season_events(se.events(se.team_games(rows))))
    priced = se.priced(ARCHIVE, games).sort("week")
    assert priced["frozen"].to_list() == [3.0, -3.0]
    assert priced["frozen_at"].to_list() == [utc(19), utc(24)]
    assert priced["close"].to_list() == [-2.0, 4.0]


def test_an_event_with_no_snapshot_before_its_previous_game_day_is_censored(rows):
    """Seen only after it moved is not seen at all: the row is kept, marked and counted, so
    the run says how many events the archive could not have caught."""
    late = ARCHIVE.filter(pl.col("captured_at") >= utc(23))
    games = se.event_games(se.in_season_events(se.events(se.team_games(rows))))
    priced = se.priced(late, games).sort("week")
    assert priced["censored"].to_list() == [True, False]
    assert priced.filter(pl.col("censored"))["frozen"].is_null().all()


def test_censored_separates_never_polled_from_polled_after_the_change(rows):
    """#303, defect 5: `censored` alone cannot tell "the archive holds nothing for this game"
    from "the archive polled it, just not before the change" -- two different facts a run
    reporting one number conflates. One event game the archive never polled at all, one it
    polled only the day after the change (both censored), and one properly frozen."""
    tg = se.team_games(rows)
    games = se.event_games(se.in_season_events(se.events(tg)))
    kc_la = games.filter(pl.col("game_id") == "2026_03_LA_KC")
    never = kc_la.with_columns(pl.lit("2099_01_ZZ_ZZ").alias("game_id"))
    late = kc_la.with_columns(pl.lit("2099_02_YY_YY").alias("game_id"))
    both = pl.concat([kc_la, never, late])
    archive = pl.concat([
        ARCHIVE,
        # "2099_02_YY_YY": polled, but only after the change (frozen_before is 09-20).
        polls([("2099_02_YY_YY", 1.0, utc(21), 3), ("2099_02_YY_YY", 1.5, utc(26), 3)]),
        # "2099_01_ZZ_ZZ" is never polled at all.
    ])
    priced = se.priced(archive, both).sort("game_id")
    never_row = priced.filter(pl.col("game_id") == "2099_01_ZZ_ZZ").row(0, named=True)
    late_row = priced.filter(pl.col("game_id") == "2099_02_YY_YY").row(0, named=True)
    kc_row = priced.filter(pl.col("game_id") == "2026_03_LA_KC").row(0, named=True)
    assert never_row["censored"] and never_row["never_polled"]
    assert late_row["censored"] and not late_row["never_polled"]
    assert not kc_row["censored"] and not kc_row["never_polled"]


def test_events_found_in_play_by_play_price_off_the_pinned_file(rows):
    """The other starter source: a change identified by passer id joins the source's values
    by (team, game) and by (team, previous game), so the gap is the pinned file's whichever
    source named the change."""
    tg = se.team_games(rows)
    starters = tg.select("game_id", "season", "week", "team").with_columns(
        pl.when((pl.col("team") == "KC") & (pl.col("week") == 3)).then(pl.lit("00-gabbert"))
          .when(pl.col("team") == "KC").then(pl.lit("00-mahomes"))
          .otherwise(pl.lit("00-other")).alias("qb"))
    ev = se.with_values(se.in_season_events(se.events(starters)), tg)
    kc = ev.sort("week")
    assert kc["departing"].to_list() == ["00-mahomes", "00-gabbert"]
    assert kc["gap"].to_list() == [60.0 - 205.0, 199.0 - 60.0]
    assert kc["prev_date"].to_list() == ["2026-09-20", "2026-09-27"]


def test_rows_without_a_week_are_refused_by_name():
    with pytest.raises(ValueError, match="week"):
        se.team_games(source_rows(SEASON).drop("week"))


def test_a_starter_change_in_play_by_play_is_detected_as_an_event(rows):
    """#420's control 1: KC's first pass goes from Mahomes to Gabbert in week 3 and back in
    week 4, and both are events, dated off the schedule with the previous-game link. The Rams'
    change across the offseason is flagged, not counted. Mutation: `events` filtering on
    `arriving == departing` (or `starters_from_pbp` reading the last passer, which is the
    backup here) leaves this red."""
    _, ev = pbp_events(rows)
    got = ev.sort("team", "week").select("team", "week", "departing", "arriving", "prev_date")
    assert got.rows() == [("KC", 3, "Mahomes", "Gabbert", "2026-09-20"),
                          ("KC", 4, "Gabbert", "Mahomes", "2026-09-27")]


def test_the_same_starter_across_games_is_no_event(rows):
    """Control 2: with Gabbert's game given to Mahomes, nobody changes in-season -- the Rams'
    offseason change is still flagged and still not counted. Mutation: dropping the
    `in_season` filter from `in_season_events` turns the Rams' offseason change into an event
    and this red."""
    same = rows.with_columns(pl.col("qb1").str.replace("Gabbert", "Mahomes"))
    tg, ev = pbp_events(same)
    assert ev.is_empty()
    assert se.events(se.observed(tg)).filter(~pl.col("in_season")).height == 1


def test_a_later_trick_play_is_not_a_change_and_a_first_one_is_the_rules_counted_cost(rows):
    """Control 3. The rule is the passer of the *first* pass play, so a wide receiver's pass
    after it changes nothing; and a wide receiver's pass that *opens* a game names him the
    starter, which reads as a change out and a change back. The second half is what the rule
    does, not what a position-aware rule would do, and it is pinned so that adding a guard is
    a decision with this test's name on it: the reconciliation counts the exposure at six of
    2,174 team-games with a first passer who threw two or fewer, and a guard chosen after
    that count would be a second definition of "starter" (method rule 1). Mutation: sorting
    `starters_from_pbp` by passer id, not play id, turns the first assertion red."""
    base = pbp_for(rows)
    kc2 = (pl.col("game_id") == "2026_02_KC_DEN") & (pl.col("posteam") == "KC")
    later = pl.concat([base, base.filter(kc2 & (pl.col("play_id") == 2.0)).with_columns(
        pl.lit(2.5).alias("play_id"), pl.lit("00-aaa-wr").alias("passer_player_id"))])
    _, ev = pbp_events(rows, later)
    assert ev.sort("week")["arriving"].to_list() == ["Gabbert", "Mahomes"]      # unchanged

    opened = base.with_columns(
        pl.when(kc2 & (pl.col("play_id") == 2.0)).then(pl.lit("00-wr"))
          .otherwise(pl.col("passer_player_id")).alias("passer_player_id"))
    _, ev = pbp_events(rows, opened)
    assert ev.sort("week")["arriving"].to_list() == ["00-wr", "Gabbert", "Mahomes"]


def test_with_values_keeps_the_dates_events_already_carry(rows):
    """Events built off the schedule carry a date for every game; the pinned file has dates
    for only the games it holds. `with_values` joins values and leaves dates alone -- so a
    game the file does not hold (here, every week after week 3) keeps its date rather than
    losing it to a null join. Mutation: re-deriving the dates from `tg` (the old behaviour)
    nulls `prev_date` for week 4 and this goes red."""
    _, ev = pbp_events(rows)
    only_early = se.team_games(rows).filter(pl.col("week") <= 3)
    got = se.with_values(ev, only_early).sort("week")
    assert got["prev_date"].to_list() == ["2026-09-20", "2026-09-27"]
    assert got["date"].to_list() == ["2026-09-27", "2026-10-04"]
    assert got["arriving_value"].to_list() == [60.0, None]


def test_the_reconciliation_counts_agreement_and_stays_inside_what_both_observe(rows):
    """Both sources name the same two KC changes, over the team-games both hold. Give
    play-by-play a week-3 starter equal to week 2's and the file's two events become a
    disagreement it counts (file 2, pbp 0, both 0); take week 4 out of the file and that game
    is outside the comparison rather than a miss. Mutation: counting a game one source lacks
    (dropping the semi-join that restricts play-by-play to the file's games) leaves the last block red."""
    file_tg = se.team_games(rows)
    pbp_tg = se.team_games_from_pbp(pbp_for(rows), schedule_for(rows))
    agree = se.reconcile(pbp_tg, file_tg)
    assert agree == se.Reconciliation(compared=file_tg.height, file=2, pbp=2, both=2)

    same = se.team_games_from_pbp(
        pbp_for(rows.with_columns(pl.col("qb1").str.replace("Gabbert", "Mahomes"))),
        schedule_for(rows))
    assert se.reconcile(same, file_tg) == se.Reconciliation(
        compared=file_tg.height, file=2, pbp=0, both=0)

    early = file_tg.filter(pl.col("week") <= 3)
    got = se.reconcile(pbp_tg, early)
    assert got == se.Reconciliation(compared=early.height, file=1, pbp=1, both=1)


def test_a_schedule_without_the_dates_and_results_is_refused_by_name(rows):
    """The study cannot date or settle a game off a schedule that lacks the columns, and says
    which ones rather than failing in a join. Mutation: removing the check turns this red with
    a polars error instead of the sentence."""
    with pytest.raises(ValueError, match="gameday"):
        se.schedule_team_games(schedule_for(rows).drop("gameday"))


def test_a_refresh_that_fails_is_served_last_good_and_named(rows):
    """The season in progress is refreshed on every run; if the wire is down the cached entry
    answers and the run line says which source it was (graceful degradation, CLAUDE.md). Each
    source here raises on its first read and serves on its second, so the refresh fails and
    the fallback read succeeds. Mutation: `_last_good` letting the refresh's error through
    makes this red; dropping the `stale` bookkeeping makes the tuple come back empty."""
    calls = {"pbp": 0, "schedules": 0}
    frame, sched = pbp_for(rows), schedule_for(rows)

    def flaky(name, table):
        def read(keys):
            calls[name] += 1
            if calls[name] == 1:
                raise ConnectionError("wire down")
            return table.filter(pl.col("season").is_in([int(k) for k in keys]))
        return read

    replay.serve(pbp=flaky("pbp", frame), schedules=flaky("schedules", sched))
    got = se.pbp_team_games([2026])
    assert got.stale == ("play-by-play 2026", "schedules")
    assert got.tg.filter(pl.col("qb").is_not_null()).height > 0
