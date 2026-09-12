"""The Panel: one row per player-week, every feature measured before its outcome.

Split out of `test_weekly_screen.py` alongside the module itself. These tests are about
*assembly* -- what a source contributes, what survives a join, what a window belongs to. The
statistic that gets measured on the result is tested next door.

Two halves, and the second exists because the first is not enough. Above the divider each leaf
is exercised on a hand-built frame: a window, a share, a two-scrape tie-break. Below it,
`build_panel` itself runs end to end against a frozen capture, because the rule that makes a
Panel a Panel is not in any leaf -- it is in how they are called. That half used to be a single
`inspect.getsource` assertion that read the function's own text back.
"""
import ast
import datetime as dt
import inspect
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import panelarchive as arc
import polars as pl
import pytest

from hub.contracts import ContractViolation
from hub.models import components
from hub.models import panel as pnl
from hub.names import player_key


def _windows():
    """Two NFL weeks: Thursday to Monday, a week apart."""
    return pl.DataFrame({
        "season": [2024, 2024],
        "week": [5, 6],
        "first_kick": [dt.date(2024, 10, 3), dt.date(2024, 10, 10)],
        "last_kick": [dt.date(2024, 10, 7), dt.date(2024, 10, 14)]})


def _scrape(day):
    return pl.DataFrame({"scrape_date": [dt.date(*day)], "key": ["x"], "ecr": [1.0]})


def test_recent_mean_is_strictly_prior():
    """Week w's control may not contain week w. The whole screen rests on it."""
    p = pl.DataFrame({"player_id": ["a"] * 5, "season": [2024] * 5,
                      "week": [1, 2, 3, 4, 5],
                      "targets": [10.0, 0.0, 0.0, 0.0, 99.0]})
    out = pnl.recent_mean(p, "targets").sort("week")
    assert out["targets_recent"].to_list()[4] == pytest.approx(0.0), \
        "weeks 2-4 are zeros; week 5's own 99 must not reach its own control"
    assert out["targets_recent"].to_list()[0] is None, "week 1 has no history"


def test_the_calendar_grid_counts_weeks_and_not_appearances():
    """The reindex `trend` and `recent_mean` both shift over, asserted once for the two.

    A player who missed week 3 must carry week 3 as a null the window counts, not have week
    2 slide into its place: `shift(1)` at week 4 reaches the week he missed. Asserted here
    rather than through one of the callers because it was covered through `recent_mean`
    alone, which left `trend` -- the function whose docstring stated the rule -- with no test
    of it at all, and neither of them noticed.
    """
    p = pl.DataFrame({"player_id": ["a"] * 3, "season": [2024] * 3,
                      "week": [1, 2, 4], "targets": [10.0, 10.0, 4.0]})
    out = pnl._on_calendar_grid(p, "targets", "player_id", "prev", 18,
                                pl.col("targets").shift(1)).sort("week")
    assert out["prev"].to_list() == [None, 10.0, None], \
        "week 2 follows week 1; week 4 follows the week 3 he missed, not week 2"
    assert out.height == 3, "the grid weeks he has no row on do not survive the join back"


def test_a_midweek_scrape_belongs_to_the_week_it_is_inside():
    """The bug that shifted the whole control by a week. An NFL week runs Thursday to Monday
    and FantasyPros scrapes land mid-week: 2024-10-04 is a Friday, *inside* week 5 (Oct 3-7),
    and it ranks week 5's Sunday games. Joining on the next *first* kickoff sent it to week 6.

    The tell was that Saquon Barkley, CeeDee Lamb and Patrick Mahomes were each missing from
    exactly one week -- and it was the week after their team's bye, because a page that
    correctly omits a bye-week player was being attached to the following week.
    """
    got = pnl.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["week"].to_list() == [5], "a Friday inside week 5 is week 5's ranking"


def test_a_scrape_before_a_week_opens_still_belongs_to_it():
    assert pnl.assign_weeks(_scrape((2024, 10, 1)), _windows())["week"].to_list() == [5]


def test_a_scrape_after_a_week_closes_belongs_to_the_next():
    assert pnl.assign_weeks(_scrape((2024, 10, 8)), _windows())["week"].to_list() == [6]


def test_a_scrape_too_far_from_any_week_is_dropped():
    """Pre-season and post-season scrapes rank no week that exists here."""
    assert pnl.assign_weeks(_scrape((2024, 9, 1)), _windows()).is_empty()


def test_lead_days_is_measured_to_the_week_being_ranked():
    got = pnl.assign_weeks(_scrape((2024, 10, 4)), _windows())
    assert got["lead_days"].to_list() == [3], "Friday to the following Monday"


def test_route_share_is_plays_on_over_team_pass_plays():
    plays = pl.DataFrame({
        "week": [1, 1, 1, 1],
        "possession_team": ["PHI"] * 4,
        "offense_players": ["a;b;c", "a;b", "a;c", "a"]})
    got = pnl.route_share_from_plays(plays, 2024).sort("player_id")
    share = dict(zip(got["player_id"], got["route_pct"], strict=True))
    assert share["a"] == pytest.approx(1.0), "on every pass play"
    assert share["b"] == pytest.approx(0.5)
    assert share["c"] == pytest.approx(0.5)


def test_the_denominator_is_per_team_per_week():
    """A player's share is of *his own* team's pass plays, not the league's."""
    plays = pl.DataFrame({
        "week": [1, 1, 1],
        "possession_team": ["PHI", "DAL", "DAL"],
        "offense_players": ["a", "b", "b"]})
    got = pnl.route_share_from_plays(plays, 2024)
    assert set(got["route_pct"].to_list()) == {1.0}, "each is on all of his own team's plays"


def test_an_empty_slate_yields_an_empty_frame_with_the_right_shape():
    got = pnl.route_share_from_plays(pl.DataFrame(), 2024)
    assert got.is_empty()
    assert set(got.columns) == {"season", "week", "player_id", "route_pct"}


def test_blank_ids_in_the_player_list_are_dropped():
    plays = pl.DataFrame({"week": [1], "possession_team": ["PHI"],
                          "offense_players": ["a;;b;"]})
    assert sorted(pnl.route_share_from_plays(plays, 2024)["player_id"].to_list()) == ["a", "b"]


def test_only_efficiency_quantities_get_an_expected_variant():
    """A target is not an estimate -- he was thrown at or he was not. Everything below the
    opportunity is an efficiency, and efficiency is what regresses."""
    assert set(pnl.EXPECTED) == {"receptions", "receiving_yards", "rushing_yards",
                                "passing_yards"}
    for opportunity in ("targets", "carries", "attempts"):
        assert opportunity not in pnl.EXPECTED


def test_the_expected_columns_are_the_ff_opportunity_names():
    assert pnl.EXPECTED["receptions"] == "receptions_exp"
    assert pnl.EXPECTED["receiving_yards"] == "rec_yards_gained_exp"
    assert pnl.XFP_WEEK == "total_fantasy_points_exp"


def test_scheme_rates_are_per_team_week():
    charting = pl.DataFrame({
        "nflverse_game_id": ["g1"] * 4,
        "nflverse_play_id": [1, 2, 3, 4],
        "is_play_action": [True, False, True, False],
        "is_motion": [True, True, True, True],
        "is_no_huddle": [False, False, False, False],
        "is_screen_pass": [True, False, False, False]})
    plays = pl.DataFrame({
        "nflverse_game_id": ["g1"] * 4, "play_id": [1, 2, 3, 4],
        "possession_team": ["PHI"] * 4, "is_pass": [True, True, False, False],
        "week": [3] * 4, "season": [2024] * 4})
    got = pnl.scheme_rates_from_plays(charting, plays).to_dicts()[0]
    assert got["posteam"] == "PHI" and got["week"] == 3
    assert got["pa_rate"] == pytest.approx(0.5)
    assert got["motion_rate"] == pytest.approx(1.0)
    assert got["nohuddle_rate"] == pytest.approx(0.0)
    assert got["pass_rate"] == pytest.approx(0.5)


def test_a_play_charted_but_not_participated_is_dropped():
    """An inner join, so a play FTN charted and participation did not is not a null team."""
    charting = pl.DataFrame({"nflverse_game_id": ["g1", "g1"], "nflverse_play_id": [1, 99],
                             "is_play_action": [True, True], "is_motion": [True, True],
                             "is_no_huddle": [False, False], "is_screen_pass": [False, False]})
    plays = pl.DataFrame({"nflverse_game_id": ["g1"], "play_id": [1],
                          "possession_team": ["PHI"], "is_pass": [True],
                          "week": [3], "season": [2024]})
    assert pnl.scheme_rates_from_plays(charting, plays).height == 1


def test_the_play_id_dtypes_are_reconciled():
    """FTN types it as an integer and nflverse as a float; unreconciled the join raises."""
    charting = pl.DataFrame({"nflverse_game_id": ["g1"], "nflverse_play_id": [1],
                             "is_play_action": [True], "is_motion": [True],
                             "is_no_huddle": [False], "is_screen_pass": [False]})
    plays = pl.DataFrame({"nflverse_game_id": ["g1"], "play_id": [1.0],
                          "possession_team": ["PHI"], "is_pass": [True],
                          "week": [3], "season": [2024]})
    assert pnl.scheme_rates_from_plays(charting, plays).height == 1


def test_an_empty_side_yields_the_right_empty_shape():
    got = pnl.scheme_rates_from_plays(pl.DataFrame(), pl.DataFrame())
    assert got.is_empty() and "pass_rate" in got.columns and "pa_rate" in got.columns


def test_trend_refuses_a_key_that_is_not_unique_per_week():
    """The guard that came out of the scheme build. `trend` fans out on a left join, so a team
    key against a player-week panel multiplies by the roster size -- and five chained calls
    took the process out on memory before this existed."""
    dup = pl.DataFrame({"team": ["PHI", "PHI"], "season": [2024, 2024],
                        "week": [3, 3], "pa_rate": [0.3, 0.4]})
    with pytest.raises(ValueError, match="one row per"):
        pnl.trend(dup, "pa_rate", "team", "pa_rate_trend")


def test_trend_accepts_the_unique_frame_it_asks_for():
    ok = pl.DataFrame({"team": ["PHI"] * 3, "season": [2024] * 3,
                       "week": [1, 2, 3], "pa_rate": [0.3, 0.4, 0.5]})
    assert "pa_rate_trend" in pnl.trend(ok, "pa_rate", "team", "pa_rate_trend").columns


def test_prior_means_returns_a_deterministically_ordered_frame():
    """A group_by emits rows in a hash-dependent order, and every caller of this means them
    again. Floating-point addition is not associative, so an unsorted hand-off moves the answer
    at ~1e-15 and every downstream sort can land differently -- improvements.md #18, which was
    found in `playoff_sos` and then found here by auditing for the pattern.
    """
    df = pl.DataFrame({"player_id": ["b", "a", "b", "a", "c"],
                       "season": [2024] * 5, "week": [1, 1, 2, 2, 1],
                       "pts": [1.0, 2.0, 3.0, 4.0, 5.0]})
    got = pnl.prior_means(df, ["player_id"], ["pts"])
    assert got.equals(got.sort(["player_id", "season", "week"])), \
        "sorted on the way out, not left to the group_by"


def test_the_designation_columns_survive_a_player_with_no_injury_row():
    """A healthy player must come through as Healthy, not as a null that a group_by drops."""
    inj = pl.DataFrame({"season": [2024], "week": [5], "key": ["hurt"],
                        "inj_sev": [1.0], "status": ["Questionable"], "practice": ["Limited"]})
    stats = pl.DataFrame({"season": [2024, 2024], "week": [5, 5],
                          "key": ["hurt", "fine"], "x": [1.0, 2.0]})
    out = (stats.join(inj, on=["season", "week", "key"], how="left")
                .with_columns(pl.col("status").fill_null("Healthy"),
                              pl.col("practice").fill_null("Healthy")))
    assert out.filter(pl.col("key") == "fine")["status"][0] == "Healthy"
    assert out.height == 2


def test_a_two_stage_aggregation_is_stable_when_the_hand_off_is_sorted():
    """The property itself, on a frame big enough for the group order to actually vary."""
    rng = np.random.default_rng(0)
    n = 4000
    df = pl.DataFrame({
        "team": [f"T{i % 32}" for i in range(n)],
        "pos": [["QB", "RB", "WR", "TE"][i % 4] for i in range(n)],
        "week": [(i % 14) + 1 for i in range(n)],
        "pts": rng.normal(10, 6, n)})

    def two_stage(sort_between):
        a = df.group_by(["team", "pos", "week"]).agg(pl.col("pts").sum().alias("s"))
        if sort_between:
            a = a.sort(["team", "pos", "week"])
        return (a.group_by(["team", "pos"]).agg(pl.col("s").mean().alias("m"))
                 .sort(["team", "pos"])["m"])

    unsorted = [two_stage(False) for _ in range(4)]
    sorted_ = [two_stage(True) for _ in range(4)]
    assert all(r.equals(sorted_[0]) for r in sorted_), "sorted: bit-identical every run"
    # the unsorted version is *usually* unstable; assert only that sorting cannot hurt
    assert sorted_[0].len() == unsorted[0].len()


def test_the_reported_lead_comes_from_the_scrape_whose_ecr_was_used():
    """`ecr.min()` was paired with `lead_days.first()` over an unsorted group_by, so the two
    could come from different scrapes. 3.3% of player-weeks carry two scrapes and 1,211 of
    those 1,216 have two different leads, and LEAD_DAYS is what docs/weekly-screen.md cites
    for the confound the whole screen is read against."""
    joined = pl.DataFrame({
        "season": [2024, 2024], "week": [5, 5], "key": ["chase", "chase"],
        # the better rank is the *later* scrape, so an unsorted first() could take the other
        "ecr": [12.0, 3.0], "lead_days": [7, 2],
    })
    r = pnl.best_per_week(joined).row(0, named=True)
    assert (r["ecr"], r["lead_days"]) == (3.0, 2), "the lead must match the ECR that won"


def test_one_scrape_a_week_is_unaffected():
    joined = pl.DataFrame({"season": [2024, 2024], "week": [5, 6], "key": ["x", "x"],
                           "ecr": [10.0, 11.0], "lead_days": [6, 5]})
    got = pnl.best_per_week(joined).sort("week")
    assert got["ecr"].to_list() == [10.0, 11.0]
    assert got["lead_days"].to_list() == [6, 5]


# --- the consensus archive, read through the fetch layer -------------------
#
# The screen's largest input and the only append-only one, so it is the only one an as-of can
# *filter* rather than label. Routed through `hub.fetch.nflverse.load_rankings` for that: the
# rows are validated against `FF_RANKINGS`, cached under a key carrying the as-of, and pinned,
# so two screens at one as-of read the same rankings however much FantasyPros published in
# between. `docs/weekly-screen.md` records what the alternative costs -- a screen whose numbers
# moved with the archive under it.

def _rankings(*specs, page: str = "weekly-op"):
    """(player, ecr, scrape_date) rows on a ranking page, in the archive's own shape."""
    return pl.DataFrame(
        [{"page_type": page, "player": p, "pos": "WR", "team": "CIN", "ecr": e,
          "sd": 1.0, "best": 1.0, "worst": 9.0, "scrape_date": d} for p, e, d in specs])


def _routed(monkeypatch, tmp_path, frame):
    """The archive faked at the fetch boundary, with the loader's cache in a tmp tree.

    RAW is redirected because `load_rankings` writes a dated parquet and a pin beside it: left
    alone, a unit test writes into the developer's own `data/raw` and the next test naming the
    same as-of is served this frame instead of its own.
    """
    import hub.fetch.nflverse as nv
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: frame)
    monkeypatch.setattr(pnl, "week_windows", lambda seasons: _windows())


def test_the_weekly_consensus_comes_through_the_loader_not_from_nflreadpy(monkeypatch,
                                                                         tmp_path):
    """The routing itself. A page type reaches the loader; nothing here reaches the network."""
    import hub.fetch.nflverse as nv
    pages = []
    _routed(monkeypatch, tmp_path, _rankings(("Ja'Marr Chase", 3.0, "2024-10-04")))
    monkeypatch.setattr(nv, "_raw_ff_rankings",
                        lambda p: pages.append(p) or _rankings(("Ja'Marr Chase", 3.0,
                                                               "2024-10-04")))
    got = pnl.weekly_consensus([2024])
    assert pages == [["all"]], "the archive is keyed by page type, not by a season list"
    assert got["ecr"].to_list() == [3.0]


def test_the_panels_as_of_is_inclusive_of_the_day_itself(monkeypatch, tmp_path):
    """One convention with `hub.draft.board.consensus` and with the loader that bounds both.

    A scrape *on* the as-of counts. If this goes red while the board's own boundary test stays
    green, the two have drifted a day apart again -- which moves every number the screen prints
    without any line of code having changed.
    """
    _routed(monkeypatch, tmp_path, _rankings(("Ja'Marr Chase", 3.0, "2024-10-04"),
                                             ("Ja'Marr Chase", 1.0, "2024-10-11")))
    on_the_day = pnl.weekly_consensus([2024], as_of="2024-10-04")
    assert on_the_day["ecr"].to_list() == [3.0], "the boundary day is inside the as-of"
    assert on_the_day["week"].to_list() == [5], "and it still belongs to the week it ranks"


def test_a_scrape_after_the_as_of_does_not_reach_the_panel(monkeypatch, tmp_path):
    """The other half: an as-of that bounds nothing would pin nothing."""
    _routed(monkeypatch, tmp_path, _rankings(("Ja'Marr Chase", 3.0, "2024-10-04"),
                                             ("Ja'Marr Chase", 1.0, "2024-10-11")))
    got = pnl.weekly_consensus([2024], as_of="2024-10-10")
    assert got["week"].to_list() == [5], "week 6's scrape was published after the as-of"


def test_two_as_ofs_are_two_pins_and_one_as_of_reproduces(monkeypatch, tmp_path):
    """The rankings entry's pin, at the seam that produces it.

    `consensus_pin` names the same cache entry `weekly_consensus` read -- source, page,
    columns and as-of -- so a lookup by key describes the load it claims to describe. Since
    #192 the screen prints the run's own read set rather than this one entry; what this holds
    is that an as-of pins the entry and reads back the same pin.
    """
    _routed(monkeypatch, tmp_path, _rankings(("Ja'Marr Chase", 3.0, "2024-10-04"),
                                             ("Ja'Marr Chase", 1.0, "2024-10-11")))
    for as_of in ("2024-10-04", "2024-10-11"):
        pnl.weekly_consensus([2024], as_of=as_of)
    early, late = (pnl.consensus_pin(d) for d in ("2024-10-04", "2024-10-11"))
    assert early is not None and late is not None, "a pinned load leaves a pin behind"
    assert early.digest != late.digest, "two as-ofs over one archive are two claims"
    assert early.pinned_at is None, \
        "an append-only source filtered at its as-of reproduces from the as-of alone"
    pnl.weekly_consensus([2024], as_of="2024-10-04")
    assert pnl.consensus_pin("2024-10-04") == early, "the same as-of reads back the same pin"


def test_an_unpinned_panel_is_still_the_common_case(monkeypatch, tmp_path):
    """No as-of, no filter: the whole archive, and a pin that names bytes but not a date.

    Worth separating, because the digest still moves when the archive does. What an unbounded
    run cannot say is *which* archive it read -- the pin carries `as_of: null`, so it records
    what was fetched and not a date anyone can fetch it at again. That is why the screen labels
    the line rather than leaving a reader to assume the run reproduces.
    """
    _routed(monkeypatch, tmp_path, _rankings(("Ja'Marr Chase", 3.0, "2024-10-04"),
                                             ("Ja'Marr Chase", 1.0, "2024-10-11")))
    assert pnl.weekly_consensus([2024]).height == 2, "both weeks, unbounded"
    from hub.config import UNPINNED, data_digest
    assert data_digest([p for p in (pnl.consensus_pin(None),) if p is not None]) != UNPINNED, \
        "an unbounded load still writes an undated pin; it is the as-of that is absent"


# --- the assembly itself, run end to end against a frozen archive ------------
#
# Everything above this line tests a leaf: a window, a share, a two-scrape tie-break. The
# rule that makes a **Panel** a Panel is not in any leaf -- it is in how `build_panel` calls
# them, on which key, over which grid, joined back to which rows. That was covered by one
# `inspect.getsource` assertion that grepped the function's own text, which passes for a
# function rewritten to do something else so long as the string it looks for survives.
#
# So the assembly runs. `tests/panelarchive.py` serves it a frozen capture of all six sources
# it reads at the last call before the wire, so every narrowing in between -- the REG filter,
# the position filter, the name key, the as-of window, eleven joins -- is the production path.
#
# **Why this rule and not another.** Leakage is the one defect this repo cannot detect after
# the fact. A feature accidentally measured after its outcome makes a screen look *better*,
# clears its gate and publishes cleanly, and stays indistinguishable from a finding until a
# season contradicts it. Every other guard here protects something a reader could eventually
# notice was wrong.

# The play-derived facts a perturbation moves. Not the injury report, the line or the
# consensus page: those are published *before* kickoff and are week-w information by the rule
# itself, so moving them would be testing the opposite claim.
_PLAY = ("fantasy_points_ppr", "target_share", "receiving_yards", "rushing_yards",
         "passing_yards", "receiving_tds", "rushing_tds", "passing_tds", "targets",
         "receptions", "carries", "attempts", "completions", "passing_interceptions",
         "fumbles_lost_total")


def _rewrite_play(season, week, name):
    """One player-week's realised play, made unmistakably different."""
    def edit(df):
        hit = ((pl.col("season") == season) & (pl.col("week") == week)
               & (pl.col("player_display_name") == name))
        return df.with_columns(
            [pl.when(hit).then(pl.col(c).cast(pl.Float64) + 37.0)
               .otherwise(pl.col(c).cast(pl.Float64)).alias(c) for c in _PLAY])
    return edit


def _rewrite_snaps(season, week, name):
    def edit(df):
        hit = ((pl.col("season") == season) & (pl.col("week") == week)
               & (pl.col("player") == name))
        return df.with_columns(pl.when(hit).then(pl.lit(0.03))
                                 .otherwise(pl.col("offense_pct")).alias("offense_pct"))
    return edit


def _both_panels(monkeypatch, tmp_path, season, week, name):
    """The Panel from the archive, and the Panel from the archive with one week rewritten."""
    arc.install(monkeypatch, tmp_path / "as-captured")
    base = pnl.build_panel(arc.SEASONS)
    monkeypatch.undo()
    arc.install(monkeypatch, tmp_path / "rewritten", edits={
        "player_stats": _rewrite_play(season, week, name),
        "snap_counts": _rewrite_snaps(season, week, name)})
    return base, pnl.build_panel(arc.SEASONS)


# Two, at different positions and in different seasons, so a green result is not a property
# of one lucky row. Both sit far enough into their season for every window to be full.
_REWRITTEN = [("CeeDee Lamb", 2024, 8), ("Bijan Robinson", 2023, 9)]


@pytest.mark.parametrize(("name", "season", "week"), _REWRITTEN)
def test_no_feature_moves_when_its_own_weeks_play_is_rewritten(monkeypatch, tmp_path,
                                                               name, season, week):
    """**The rule, asserted by running the assembly.** A feature measured after its outcome
    would move when that outcome moves; every feature here is measured before it, so none may.

    One player-week's realised play is rewritten in the frozen archive -- points, counts,
    yardage, turnovers, target share, snap share, all of it -- and the Panel is rebuilt. Every
    feature on every row at or before that (season, week) must be bit-identical. Not only that
    player's row: a defence-vs-position term reads the whole league's week, and a leak there
    would arrive on somebody else's row.

    What may move on the rewritten row is what the archive itself supplies -- the realised
    columns and the two totals built from them.

    **Run over `panel.feature_columns`, which is what makes that classification a claim under
    test.** The exempt set used to be derived here in the harness, so the module could say
    nothing about which of its own columns kept its own rule (#203). Now the module says it
    and this asserts it: a column `column_role` calls derived -- including any future one that
    merely *ends* in `_prior`, `_recent` or `_trend` without going through `expanding_weeks`
    -- is perturbed here like every other feature, and a mislabelled one moves and fails.
    """
    base, after = _both_panels(monkeypatch, tmp_path, season, week, name)
    feats = pnl.feature_columns(base)
    before = arc.rows_up_to(base, season, week).select(feats)
    got = arc.rows_up_to(after, season, week).select(feats)
    assert before.height == got.height, (
        f"the rewrite changed which rows exist at or before {season} week {week} "
        f"({before.height} -> {got.height}); it is meant to change values, not membership")
    moved = [c for c in feats if not before[c].equals(got[c])]
    assert not moved, (
        f"{moved} moved on rows at or before {season} week {week} when only that week's own "
        f"play was rewritten. A feature that can see its own outcome week is the one defect "
        f"this repo cannot detect after the fact: it makes a screen look better, clears its "
        f"gate and publishes, and reads as a finding until a season contradicts it.")


@pytest.mark.parametrize(("name", "season", "week"), _REWRITTEN)
def test_the_rewritten_week_does_reach_the_features_that_are_allowed_to_see_it(
        monkeypatch, tmp_path, name, season, week):
    """The premise of the test above, which is otherwise satisfiable by changing nothing.

    An edit the assembly never reads makes "no feature moved" vacuously true -- the shape
    every guard in `tests/contracts/test_guards_are_load_bearing.py` was written after. Later
    weeks are exactly where that week's play is *supposed* to arrive: it is their past.
    """
    base, after = _both_panels(monkeypatch, tmp_path, season, week, name)
    feats = pnl.feature_columns(base)
    later = ((pl.col("season") > season)
             | ((pl.col("season") == season) & (pl.col("week") > week)))
    a = base.filter(later).sort(["player_id", "season", "week"]).select(feats)
    b = after.filter(later).sort(["player_id", "season", "week"]).select(feats)
    moved = [c for c in feats if not a[c].equals(b[c])]
    assert len(moved) >= 10, (
        f"only {moved} moved after {season} week {week}; the rewrite is not reaching the "
        f"assembly, so the leakage test beside this one is asserting nothing")
    assert {"ppg_before", "dvp", "snap_trend", "tgt_trend", "targets_recent"} <= set(moved), (
        f"the season-to-date control, the opponent term, both trends and the recent mean all "
        f"read earlier weeks, so all five must move; {sorted(moved)} did")


def test_the_panel_is_one_row_per_player_season_week(monkeypatch, tmp_path):
    """The other half of the definition. A fanned-out join is how a Panel stops being one.

    `trend` raises on a duplicated key for this reason and is tested on a hand-built frame
    above; this asserts the assembly does not hand it one in the first place.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    assert p.height > 0, "an empty Panel would satisfy every assertion in this section"
    assert not p.select("player_id", "season", "week").is_duplicated().any()


@pytest.mark.parametrize(("source", "name_col"), [
    ("player_stats", "player_display_name"),
    ("ff_opportunity", "full_name"),
    ("ff_rankings", "player"),
    # Was a strict xfail while the first `--write` had matched the cohort by display name
    # and left Michael Pittman Jr.'s 25 snap rows out; re-taken 2026-09-11 with the corrected
    # rule and the marker came off (#234).
    ("snap_counts", "player"),
])
def test_every_cohort_player_reaches_every_per_player_capture(source, name_col):
    """The archive's cohort is sixteen players in every source that names players, on the key
    the Panel joins on. Not the injury report, where a player with no row is a real fact.

    The first re-take (#234) matched the cohort by display name, and the sources do not agree
    on one: `ff_rankings` carries `Michael Pittman Jr.` and `Patrick Mahomes II`, `snap_counts`
    the first of those. Both fell out of those captures and nothing said so -- the Panel
    joined, found no row, and carried the null, which is exactly the shape a trimmed fixture
    fails in: it still reads as a capture and proves less. `player_key` is what every join in
    `build_panel` collapses a name to, so the cohort is asserted on it.
    """
    keys = {player_key(n) for n in arc.PLAYERS}
    got = {player_key(n) for n in arc.frame(source)[name_col].drop_nulls().to_list()}
    missing = sorted(keys - got)
    assert not missing, (
        f"{missing} are in the cohort and not in the frozen `{source}` capture; every join "
        f"onto them resolves to null and no test downstream can tell that from a quiet week")


def test_the_trend_features_have_something_to_compute_on(monkeypatch, tmp_path):
    """The archive's own premise. `trend` reaches back six calendar weeks, so a capture of
    three weeks would return an all-null column and every leakage assertion above would hold
    over nothing. Recorded as a number rather than left to whoever next trims the fixture."""
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    for col in ("snap_trend", "tgt_trend", "dvp", "targets_recent", "ppg_before"):
        assert p[col].drop_nulls().len() >= 200, (
            f"{col} is null on all but {p[col].drop_nulls().len()} of {p.height} rows")


def test_the_panel_carries_the_designation_and_the_out_weeks_do_not_reach_it(monkeypatch,
                                                                            tmp_path):
    """The structural fact that stops anyone fitting the injury term on this Panel, measured.

    `player_stats` has no row for a player who did not play, so an "Out" designation and a
    Panel row are nearly disjoint by construction -- 6 of 5,473 across 2021-25 in production,
    and 0 of 3 in this capture. `hub.models.injury` prices an injury row with no stat row as
    *zero*, and the player who did not play is its whole subject; retention fitted here would
    measure "what a Questionable player who played anyway retains" and report it under the
    stronger result's name.

    This replaces an `inspect.getsource` assertion that grepped `build_panel` for the sentence
    "CANNOT be fitted here". That passed for any function still carrying the string and failed
    for a correct one that worded it differently, which is the wrong way round. The sentence
    stays in the docstring, where a reader meets it; what holds it true is here.

    `status` and `practice` are carried regardless, because Gate B builds a complete grid
    where a missing row is a zero, and that is where the term belongs.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    inj = arc.frame("injuries")
    assert {"status", "practice"} <= set(p.columns), "carried for Gate B"
    assert inj.filter(pl.col("report_status") == "Out").height == 3, \
        "the capture holds three Out designations; without one there is nothing to be absent"
    assert p.filter(pl.col("status") == "Out").height == 0, \
        "none of them reaches a Panel row, because he did not play"
    assert p.filter(pl.col("status") == "Questionable").height == 9, \
        "Questionable does reach it -- 9 of the capture's 14 -- which is the weaker term"
    assert p.filter(pl.col("status") == "Healthy").height > 0, \
        "and a player with no injury row comes through as Healthy, not as a dropped null"


def test_the_gate_spec_keeps_the_players_the_consensus_page_leaves_out(monkeypatch, tmp_path):
    """`consensus=False` is not a smaller Panel, it is a wider one, and only running says so.

    The gate needs a projection for every rostered player. Being unranked is the incumbent's
    *answer*, not a reason to have no projection -- only the screen, which measures beyond
    consensus, requires it to exist. The inner join at the end of `build_panel` is where that
    happens, and 66 player-weeks of this capture fall through it.
    """
    arc.install(monkeypatch, tmp_path)
    screen = pnl.build_panel(arc.SEASONS)
    everyone = pnl.build_panel(arc.SEASONS, pnl.PanelSpec(consensus=False))
    assert everyone.height == 410 and screen.height == 344
    assert "ecr" in screen.columns and "ecr" not in everyone.columns
    kept = set(zip(screen["season"].to_list(), screen["week"].to_list(), strict=True))
    all_weeks = set(zip(everyone["season"].to_list(), everyone["week"].to_list(), strict=True))
    assert (2024, 1) in all_weeks and (2024, 1) not in kept, \
        "the archive's consensus page starts at 2024 week 4; weeks 1-3 exist only unranked"


def test_the_expected_spec_moves_the_priors_and_not_the_realised_columns(monkeypatch,
                                                                        tmp_path):
    """What `PanelSpec(expected=True)` actually changes, run rather than read.

    `ff_opportunity` prices each opportunity by its situation, so a six-target week three
    yards downfield stops being the same number as one at fifteen. It is coalesced into the
    frame the priors are taken over -- so `receiving_yards_prior` moves -- while the Panel's
    own realised `receiving_yards` for the week is untouched.
    """
    arc.install(monkeypatch, tmp_path)
    plain = pnl.build_panel(arc.SEASONS)
    xp = pnl.build_panel(arc.SEASONS, pnl.PanelSpec(expected=True))
    assert plain.height == xp.height
    on = ["player_id", "season", "week"]
    j = plain.sort(on).join(xp.sort(on).select(*on, pl.col("receiving_yards").alias("x_real"),
                                               pl.col("receiving_yards_prior").alias("x_prior")),
                            on=on)
    assert j.filter(pl.col("receiving_yards") != pl.col("x_real")).height == 0, \
        "the realised column is what happened and the spec does not rewrite it"
    moved = j.filter((pl.col("receiving_yards_prior") - pl.col("x_prior")).abs() > 1e-9)
    assert moved.height >= 100, \
        f"only {moved.height} priors moved; the expected frame is not reaching them"


# Every spec `build_panel` is called with anywhere in the tree. `routes` and `scheme` are the
# two this archive deliberately cannot drive -- see the fixtures README -- and neither touches
# a yardage column, so the pair below is the whole population for this question.
_TOTALLED_SPECS = ((pnl.SCREEN_SPEC, "default"), (pnl.PanelSpec(expected=True), "expected"))


@pytest.mark.parametrize(("spec", "which"), _TOTALLED_SPECS,
                         ids=[w for _, w in _TOTALLED_SPECS])
def test_a_total_on_the_panel_is_the_sum_of_the_columns_beside_it(monkeypatch, tmp_path,
                                                                  spec, which):
    """`yds` and `tds` are sums of columns the Panel also carries. Under every spec.

    They were not. `p` was bound from the realised frame and the two totals were rebuilt from
    the *coalesced* one afterwards, so under `expected=True` the total was an expected
    quantity while the three columns it is the sum of stayed realised: they disagreed on
    **342 of 344** rows of this archive, against 0 under the default spec.

    Not leakage. Both quantities belong to week w, nothing sees its own outcome, and the
    assertion above this one passes either way -- which is exactly why this needs its own
    test. It is the Panel giving two answers to one question, and the touchdown-rate prior
    dividing one of them by the other.

    An identity rather than a tolerated rate of disagreement, because there is no number of
    rows on which a total may stop being the sum of its parts.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS, spec)
    assert p.height > 0, "an empty Panel would satisfy this vacuously"
    for total, parts in (("yds", ("receiving_yards", "rushing_yards", "passing_yards")),
                         ("tds", ("receiving_tds", "rushing_tds", "passing_tds"))):
        summed = pl.col(parts[0]) + pl.col(parts[1]) + pl.col(parts[2])
        off = p.filter((pl.col(total) - summed).abs() > 1e-9)
        assert off.height == 0, (
            f"under the {which} spec, {off.height} of {p.height} rows carry a `{total}` that "
            f"is not {' + '.join(parts)} on the same row. The Panel is then measuring two "
            f"different things under one name, and whichever consumer reads the total is "
            f"reading a quantity the columns beside it contradict.")


@pytest.mark.parametrize(("spec", "which"), _TOTALLED_SPECS,
                         ids=[w for _, w in _TOTALLED_SPECS])
def test_the_scoring_prior_splits_into_two_halves_that_sum_back_to_it(monkeypatch, tmp_path,
                                                                      spec, which):
    """`td_ppg_before + nontd_ppg_before == ppg_before`, exactly, on every row. #179.

    An identity for the same reason the test above it is one: the pair exists so the screen
    can hold prior touchdown scoring and prior everything-else apart *without* controlling for
    anything the pooled `ppg_before` was not already controlling for. That property is the
    identity. If the two stop summing to the total, the decomposed control set is no longer
    nested inside the pooled one, and a coefficient that moves between the two bases has two
    possible causes instead of one -- which is the whole thing #179 was run to distinguish.

    It holds by construction, because the second half is the first subtracted from the total
    rather than a second sum over the scored columns that are not touchdowns. Rebuilding it
    from `SCORING` instead would leave the return and special-teams scores that `SCORING` does
    not price in neither half, and the identity would fail on exactly the rows where it was
    doing work. Asserted anyway: "holds by construction" is what the `yds`/`tds` totals above
    were also said to do, on 342 of 344 rows.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS, spec).drop_nulls("ppg_before")
    assert p.height > 0, "an empty Panel would satisfy this vacuously"
    off = p.filter(
        (pl.col("td_ppg_before") + pl.col("nontd_ppg_before") - pl.col("ppg_before")).abs()
        > 1e-9)
    assert off.height == 0, (
        f"under the {which} spec, {off.height} of {p.height} rows carry halves that do not "
        f"sum to `ppg_before`. The decomposed basis then controls for something the pooled "
        f"one did not, and #179's comparison stops being attributable to one constraint.")


def test_a_passing_touchdown_is_priced_at_four_and_not_at_six(monkeypatch, tmp_path):
    """`td_ppg_before` prices the three touchdown columns from `SCORING`, separately.

    The ticket that asked for this described `ppg_before` as containing prior touchdowns
    "scored at six points each", which is true of two of the three. A passing touchdown is
    **four**. Pricing all three at six would put an extra two points a passing score into the
    touchdown half and take the same two out of the non-touchdown half beside it -- on
    quarterbacks only, and silently, because the identity above would still hold: the
    remainder is a subtraction and absorbs any mispricing of the thing subtracted.

    So the identity cannot catch this and needs its own assertion. Rebuilt here from the
    Panel's own per-type prior means rather than from a constant, so a weight that moves in
    `SCORING` moves on both sides and this stays a test of the wiring.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    assert set(pnl.TD_COLUMNS) <= set(components.SCORING), \
        "a touchdown column with no price; `td_ppg_before` would multiply by a KeyError"
    assert components.SCORING["passing_tds"] != components.SCORING["rushing_tds"], (
        "the three touchdown weights are equal, so this test cannot fail and the reason "
        "`TD_COLUMNS` is split by type has gone away")

    # The per-type priors are dropped from the Panel on purpose, so rebuild the halves from
    # the frame the Panel takes them over and check the Panel's column against it.
    counted = pnl.weekly_stats(arc.SEASONS).sort(["player_id", "season", "week"])
    own = pnl.prior_means(counted, ["player_id"], list(pnl.TD_COLUMNS), within_season=True)
    want = own.with_columns(
        reduce(operator.add,
               (pl.col(f"{c}_prior") * components.SCORING[c] for c in pnl.TD_COLUMNS))
        .alias("want"))
    j = p.join(want, on=["player_id", "season", "week"], how="inner").drop_nulls("want")
    assert j.height > 0, "nothing joined, so the comparison below is vacuous"
    off = j.filter((pl.col("td_ppg_before") - pl.col("want")).abs() > 1e-9)
    assert off.height == 0, (
        f"{off.height} of {j.height} rows price the touchdown half differently from "
        f"`SCORING`. A flat six is the specific way this goes wrong, and it goes wrong only "
        f"on players who throw.")

    # And the premise: the archive has to contain a passing touchdown, or a flat six would
    # pass everything above.
    passers = j.filter(pl.col("passing_tds_prior") > 0)
    assert passers.height > 0, (
        "no row in this archive carries a prior passing touchdown, so pricing all three at "
        "six would satisfy every assertion here")


def test_the_touchdown_rate_prior_divides_one_measurement_by_itself(monkeypatch, tmp_path):
    """`tds_prior / yds_prior`, and both sides realised whichever spec asked for the Panel.

    `EXPECTED` deliberately leaves touchdowns out -- everything below the opportunity is an
    efficiency and efficiency is what regresses, which is the whole content of this feature at
    -0.040 across five of five seasons. So the numerator has no expected version by choice,
    and a denominator that took one turned the rate into realised touchdowns per expected
    yard, which is not a rate of anything.

    Pinned as **bit-identical between the two specs** rather than as a property of one of
    them: the touchdown rate is the same measurement whoever asked for the Panel, and the only
    way that stays true is if nothing upstream of it moves. The last assertion is the premise
    -- the spec must still be reaching the priors it is for, or this would hold by the flag
    doing nothing at all.
    """
    arc.install(monkeypatch, tmp_path)
    on = ["player_id", "season", "week"]
    plain = pnl.build_panel(arc.SEASONS).sort(on)
    xp = pnl.build_panel(arc.SEASONS, pnl.PanelSpec(expected=True)).sort(on)
    assert plain.height == xp.height, "the spec chooses a measurement, not a row set"
    for col in ("tds", "yds", "tds_prior", "yds_prior", "td_rate_prior"):
        assert plain[col].equals(xp[col]), (
            f"`{col}` moved under `expected=True`. The touchdown rate is a realised count "
            f"over a realised total on both sides, so neither it nor either of its two "
            f"inputs may depend on which sources a caller asked for.")
    assert not plain["receiving_yards_prior"].equals(xp["receiving_yards_prior"]), \
        "the expected spec is reaching nothing, so the four assertions above are vacuous"


def test_an_injected_board_lands_on_the_season_it_was_built_for(monkeypatch, tmp_path):
    """`spec.ranks` is injected rather than fetched, and this is what the injection has to do.

    A board lives under `draft/` and six `draft/` modules import `models/` while nothing goes
    the other way, so `build_panel` takes the frame instead of building one --
    `weekly_gate_data.preseason_ranks` is the caller that does. The join is on (key, season),
    so a board built for one season must reach that season's rows and no others: an August
    opinion about 2024 is not an August opinion about 2023, and a join that lost the season
    would make it one.
    """
    arc.install(monkeypatch, tmp_path)
    board = arc.frame("draft_board")
    ranks = board.select(
        pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
        pl.lit(2024).cast(pl.Int64).alias("season"),
        pl.col("ecr").alias("preseason_ecr")).drop_nulls("preseason_ecr")
    p = pnl.build_panel(arc.SEASONS, pnl.PanelSpec(consensus=False, ranks=ranks))
    assert "preseason_ecr" in p.columns
    got = p.filter(pl.col("preseason_ecr").is_not_null())
    assert got.height > 0 and set(got["season"].to_list()) == {2024}, \
        "a board built for 2024 must not price 2023's weeks"
    assert p.filter((pl.col("season") == 2024)
                    & pl.col("preseason_ecr").is_null()).height == 0, \
        "and every 2024 player here is on that board, so none should come back unpriced"


# --- the injury report's three shapes, and what the Panel does with a refusal -----------
#
# ADR-0023. #132 routed `injury_severity` through `INJURIES.conform` and deleted the private
# lookup it had used, and three of the deleted lines were tolerances rather than duplication:
# `game_status` for `report_status`, `player_name` for `full_name`, and a missing
# `practice_status` filled with the string "None". All three are refusals now, none of them
# was covered in either direction, and "whatever nflverse shipped this week" is not a record
# of a decision. These are.
_REFUSED_SHAPES = (
    ("report_status", lambda df: df.rename({"report_status": "game_status"}), "game_status"),
    ("full_name", lambda df: df.rename({"full_name": "player_name"}), "player_name"),
    ("practice_status", lambda df: df.drop("practice_status"), None),
)
_SHAPE_IDS = ("renamed status", "renamed name", "no practice column")


@pytest.mark.parametrize(("declared", "edit", "arrived"), _REFUSED_SHAPES, ids=_SHAPE_IDS)
def test_the_injury_ordinal_refuses_the_three_shapes_it_used_to_accept(monkeypatch, tmp_path,
                                                                       declared, edit,
                                                                       arrived):
    """Each refusal is deliberate, and it speaks in the declaration's vocabulary.

    A fallback that takes `game_status` when `report_status` is gone is how a silent rename
    goes unnoticed -- the Week 7 failure this repo names -- and it computes the ordinal from a
    column `INJURIES` never promised. The old code raised when *neither* spelling was present,
    so what the deletion changed is only which frames it refuses, and this pins which.

    The missing practice column is the case that looks like a plain narrowing and is not.
    `practice_key` fills a null cell with "None", and `INJURIES` leaves `practice_status` out
    of `non_null` on purpose, so "None" is already a designation meaning "on the report,
    nothing said about practice". A whole column of it is indistinguishable from a real week
    in which nobody was designated -- and it silently narrows the (status, practice) pair
    `hub.models.injury` keys its retention table on to `status` alone. ADR-0023.

    The message names the column the contract declares and not the one that arrived: the next
    reader is being told which declaration to go and change, not offered a synonym.
    """
    arc.install(monkeypatch, tmp_path, edits={"injuries": edit})
    with pytest.raises(ContractViolation) as e:
        pnl.injury_severity(arc.SEASONS)
    assert declared in str(e.value), \
        f"the refusal must name `{declared}`, which is what `INJURIES` declares"
    if arrived is not None:
        assert arrived not in str(e.value), (
            f"`{arrived}` is the name that turned up, not a name this repo knows. A "
            f"message that offers it invites the fallback back in as a bugfix.")


@pytest.mark.parametrize(("declared", "edit", "arrived"), _REFUSED_SHAPES, ids=_SHAPE_IDS)
def test_a_refused_injury_report_nulls_three_columns_and_not_the_panel(monkeypatch, tmp_path,
                                                                      capsys, declared, edit,
                                                                      arrived):
    """ADR-0023's live half: `build_panel` degrades around the refusal above, to null.

    `build_panel` calls `injury_severity` unconditionally and the Sunday panel is the live
    path CLAUDE.md's degradation rule is written for, so a refusal must not cost the other
    forty columns. What it costs is the three columns the report contributes.

    **Null, and not `Healthy`.** The fill under the join means "this player has no row on the
    injury report"; a refused report means nobody has one and nothing is known. Spelling that
    `Healthy` would publish a league in perfect health on the week the source broke, with
    `inj_sev` reaching the screen as a measured zero on every row -- the same silent narrowing
    as filling "None", one level up.
    """
    arc.install(monkeypatch, tmp_path / "healthy")
    healthy = pnl.build_panel(arc.SEASONS).sort(["player_id", "season", "week"])
    capsys.readouterr()

    # Its own cache, as `_both_panels` gives each of its builds. `injuries` now comes through
    # `nflverse.load` (#35), and a second build under the same `RAW` is served the healthy
    # entry the first one wrote -- which is the loader keeping its promise, and the reason a
    # refusal has to be reached on a cold cache to be reached at all.
    arc.install(monkeypatch, tmp_path / "refused", edits={"injuries": edit})
    p = pnl.build_panel(arc.SEASONS).sort(["player_id", "season", "week"])
    err = capsys.readouterr().err

    assert p.height == healthy.height and p.height > 0, \
        "the refusal cost rows, so the Panel was not served around it"
    assert p["snap_trend"].equals(healthy["snap_trend"]), \
        "nothing but the injury columns may move when the injury report is refused"
    for col in ("inj_sev", "status", "practice"):
        assert col in p.columns, f"`{col}` must still be here for a consumer reading it"
        assert p[col].null_count() == p.height, (
            f"`{col}` carries {p.height - p[col].null_count()} non-null values on "
            f"a Panel built with no injury report. Null is the only honest value here.")
    assert "Healthy" not in set(p["status"].to_list()) | set(p["practice"].to_list()), \
        "a refused report is not a league in perfect health"
    assert "None" not in set(p["practice"].to_list()), \
        "and it is not a week in which nobody was designated either"
    assert declared in err, \
        f"the operator is told nothing about `{declared}`, so the loss is silent"
    if arrived is not None:
        assert arrived not in err, "the message names the declaration, not the arrival"


def test_the_panel_only_degrades_around_a_contract_violation(monkeypatch, tmp_path):
    """A source that changed shape is servable. This module being wrong is not.

    `injury_columns` catches `ContractViolation` and nothing wider. A bare `except Exception`
    there would hide a defect in the ordinal itself behind three columns of nulls that look
    exactly like a quiet week -- indistinguishable, on the frame, from the degradation the
    test above asserts, and served every Sunday until someone read the ordinal closely.
    """
    arc.install(monkeypatch, tmp_path)

    def broken(_seasons):
        raise TypeError("the ordinal is wrong, not the source")

    monkeypatch.setattr(pnl, "injury_severity", broken)
    with pytest.raises(TypeError, match="the ordinal is wrong"):
        pnl.build_panel(arc.SEASONS)


# --- which side of its own rule each column is on (#203) --------------------------------
#
# The rule is kept mechanically for the derived features and was kept nowhere for the frame
# they are built on: `build_panel` starts from the realised week-level frame and never strips
# it, so week w's own `targets`, `receptions`, `offense_pct`, `tds` and `yds` reach the
# return alongside the features. They are meant to -- `hub.models.weekly` fits against them
# and the Usage screen measures against them -- and what was missing is that the return said
# nothing about which was which. #176 is the cost when this species is not caught: a coverage
# measurement centred on each player's own realised mean, published, and 81.1% -> 77.4% once
# corrected.
#
# Every spec `build_panel` is called with anywhere in the tree, so a column that reaches only
# the gate's panel is classified too. `routes` and `scheme` are the two this archive cannot
# drive (see the fixtures README); their raw columns are asserted by name in the last test
# here rather than through a Panel that cannot be built.
_SPECS = (
    (pnl.SCREEN_SPEC, "screen"),
    (pnl.PanelSpec(consensus=False), "gate"),
    (pnl.PanelSpec(expected=True), "expected"),
)


@pytest.mark.parametrize(("spec", "which"), _SPECS, ids=[w for _, w in _SPECS])
def test_every_column_the_panel_hands_back_is_on_exactly_one_side_of_its_rule(monkeypatch,
                                                                             tmp_path,
                                                                             spec, which):
    """The partition is total and disjoint, over the frame that is actually served.

    Total is the half that matters: a column on no side is a column a caller can only judge
    by its suffix, which is the whole defect. Disjoint is what makes `feature_columns` and
    `outcome_columns` a decision rather than two overlapping opinions.

    Derived from the served frame rather than compared against a list of today's columns --
    a list would pass forever while the next column slipped past it.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS, spec)
    feats, outs = set(pnl.feature_columns(p)), set(pnl.outcome_columns(p))
    ident = {c for c in p.columns if pnl.column_role(c) == "identity"}
    # The fifth side (#170): observed during week w, and neither a feature of it nor its
    # outcome. Read off `column_role` rather than off `RECORDED`, so a column that acquires
    # the role by any other route is still counted here.
    rec = {c for c in p.columns if pnl.column_role(c) == "recorded"}
    assert p.height > 0 and len(p.columns) > 40, "an empty frame would satisfy this vacuously"
    assert feats | outs | ident | rec == set(p.columns), (
        f"{sorted(set(p.columns) - (feats | outs | ident | rec))} are on no side of the rule; "
        f"`_served` should have refused the frame before it got here")
    assert not (feats & outs) and not (feats & ident) and not (outs & ident), \
        "a column on two sides is not a classification"
    assert feats and outs, "both halves must be non-empty or the split is not doing anything"


def test_the_five_columns_the_issue_names_are_outcomes_and_not_features(monkeypatch,
                                                                       tmp_path):
    """The specific leak #203 opens on, pinned by name. A caller naming one gets refused.

    Not the general guarantee -- that is the test above and the refusal below -- but the five
    that were reachable as features on the day the issue was written. `offense_pct` is the
    sharpest of them: `snap_trend` is built from it and is this repo's one durable in-season
    signal, so the wrong one of that pair is exactly the reach a tired Sunday makes.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    named = ("targets", "receptions", "offense_pct", "tds", "yds")
    assert set(named) <= set(pnl.outcome_columns(p)), \
        "these are week w's own outcome and the Panel must say so"
    for c in named:
        with pytest.raises(pnl.PanelRuleViolation) as e:
            pnl.require_features(p, ["snap_trend", c])
        assert f"`{c}`" in str(e.value), "the refusal must name the column that caused it"
        assert "snap_trend" not in str(e.value), \
            "and not the one that was fine, or the message sends a reader nowhere"


def test_the_refusal_points_at_the_column_measured_before_the_week(monkeypatch, tmp_path):
    """A refusal that only says no costs the next reader the lookup the message could do.

    `targets` has two columns beside it that are the same quantity measured before the week --
    the season-to-date prior and the last three weeks. Naming them is the difference between
    a guard and a wall.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    with pytest.raises(pnl.PanelRuleViolation) as e:
        pnl.require_features(p, ["targets"])
    assert "targets_prior" in str(e.value) and "targets_recent" in str(e.value), \
        "both of `targets`' before-the-week measurements are on this frame and go unnamed"
    with pytest.raises(pnl.PanelRuleViolation) as e:
        pnl.require_features(p, ["not_a_column_at_all"])
    assert "not on this frame" in str(e.value), \
        "a name that is on no side because it is on no frame says the simpler thing"


def test_the_three_named_pre_kickoff_exceptions_stay_usable_as_week_w_information(monkeypatch,
                                                                                  tmp_path):
    """The line, the injury report and the opponent are week-*w* information by the rule.

    `docs/method.md` rule #2 and `CONTEXT.md`'s **Panel** entry both name them: they are
    published *for* week w and are legitimately available before kickoff. A tightening that
    put them out of a caller's reach would have broken the Panel rather than tightened it --
    a screen with no line, no injury report and no opponent measures nothing this repo asks.

    Consensus rides here too. Not one of the three, and the same species: `assign_weeks` and
    `CONSENSUS_MAX_LEAD_DAYS` exist to keep a scrape on the near side of its kickoff, and
    `ecr` is half of `weekly_screen.CONTROLS`, so a Panel that refused it would refuse the
    screen's own control set.

    **`wind` used to be in `line` here and is deliberately not**, since #170: it comes off the
    same schedule frame and is the one column on it that is *observed at kickoff* rather than
    published for it. The tests below this one are what say so.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    line = ("own_spread", "total_line", "implied_total", "rest", "roof", "is_home")
    report = ("inj_sev", "status", "practice")
    opponent = ("opp", "opponent_team")
    consensus = ("ecr", "lead_days")
    for group in (line, report, opponent, consensus):
        assert set(group) <= set(pnl.feature_columns(p)), \
            f"{sorted(set(group) - set(pnl.feature_columns(p)))} became unreachable"
        pnl.require_features(p, list(group))       # raises if any of them is refused
    assert set(pnl.INJURY_COLUMNS) == set(report), \
        "the report's three are named once in `INJURY_COLUMNS`; this must not drift from it"


# --- wind is observed weather, and a missing reading is not calm (#170) --------------------
#
# It was read straight off the schedule into `PRE_KICKOFF` and screened as a week-w feature
# with a pre-registered negative sign, which is `docs/method.md` rule 2 broken by the Panel
# that exists to keep it: the reading is taken *at* kickoff, inside the outcome window. The
# null-filling compounded it -- a dome or an absent reading became `0.0`, a substantive value
# on a feature whose sign was pre-stated, so "no measurement" and "no wind" were one number.
#
# Three tests, one per acceptance criterion the ticket names. The fourth criterion -- the
# family size -- lives next door in `test_weekly_screen.py`, because that is where the family
# is.


def test_wind_is_a_recorded_condition_and_cannot_be_screened_as_a_feature(monkeypatch,
                                                                          tmp_path):
    """Criterion one: the column's status says it is not available before kickoff.

    Reclassified rather than removed. Wind is a real thing about a real football game and a
    caller describing a week by the conditions it was played in should still be able to reach
    it -- what it may not be is a *predictor* of the week it was measured on. `RECORDED` is the
    side that says exactly that, and it is neither `feature_columns` nor `outcome_columns`:
    filing it as an outcome would have satisfied the refusal while claiming wind is this
    player's own realised play, which it is not.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    assert "wind" in p.columns, "still served: describable after the fact, just not screenable"
    assert pnl.column_role("wind") == "recorded"
    assert "wind" not in pnl.feature_columns(p), "a feature of the week it was measured on"
    assert "wind" not in pnl.outcome_columns(p), (
        "not this player's realised play either -- it is a property of the fixture, and "
        "calling it an outcome makes `yds` and `wind` the same kind of thing")
    with pytest.raises(pnl.PanelRuleViolation, match="observed during week w"):
        pnl.require_features(p, ["wind"])


def test_a_missing_wind_reading_reaches_the_panel_as_null_and_not_as_calm(monkeypatch,
                                                                          tmp_path):
    """Criterion two, first half: the fill is gone, and it was fabricating every zero it wrote.

    `game_context` used to end `pl.col("wind").fill_null(0.0)`. The archive is what says how
    much that cost: **175 of 416 captured game rows carry no reading** -- 132 indoors and 33
    at open-air stadiums -- and **not one game has a measured wind of zero**. So before this
    change every zero in the column was a non-measurement, on a feature whose sign was
    pre-registered as negative, and the screen could not have told the two apart because there
    was nothing to tell apart: the measured population contained no zeros at all.

    The counts are asserted rather than quoted, so the sentence above cannot outlive the
    capture it describes.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    indoors = p.filter(pl.col("roof").is_in(["dome", "closed", "open"]))
    assert indoors.height > 0, "no indoor game reached the Panel; this asserts nothing"
    assert indoors["wind"].null_count() == indoors.height, (
        "an indoor game has no wind reading, and `0` is a measurement rather than the "
        "absence of one")
    assert p.filter(pl.col("wind") == 0).height == 0, (
        "a player-week the Panel calls a measured calm. There are none in this capture, "
        "which is why the fill was indistinguishable from data")


def test_a_dome_a_missing_reading_and_a_measured_calm_are_three_different_things(monkeypatch,
                                                                                 tmp_path):
    """Criterion two, second half. The capture holds no measured calm, so one is introduced.

    The archive cannot demonstrate the distinction on its own -- there is no game in it with a
    measured zero -- so the schedule capture is edited to give one week's open-air games a
    reading of exactly `0`. That is the case the old code was indistinguishable from, and the
    only honest way to test it is to create one and watch it stay distinct.

    Three states, and all three are reachable off the served Panel: a measured calm is `0.0`,
    an unread outdoor game is null beside `roof == "outdoors"`, and a dome is null beside a
    roof that is not. `roof` is what separates the second from the third, which is why this
    change needed no new column.
    """
    season, week = 2023, 11

    def measured_calm(df):
        hit = ((pl.col("roof") == "outdoors") & (pl.col("season") == season)
               & (pl.col("week") == week))
        return df.with_columns(
            pl.when(hit).then(pl.lit(0)).otherwise(pl.col("wind")).cast(pl.Int32).alias("wind"))

    arc.install(monkeypatch, tmp_path, edits={"schedules": measured_calm})
    p = pnl.build_panel(arc.SEASONS)

    that_week = (pl.col("season") == season) & (pl.col("week") == week)
    calm = p.filter(that_week & (pl.col("roof") == "outdoors"))
    assert calm.height > 0, "the edit reached no Panel row, so this proves nothing"
    assert calm["wind"].null_count() == 0 and calm["wind"].max() == 0.0, (
        "a game that was measured and found calm must survive as a zero -- the point of "
        "dropping the fill is that this is now the *only* thing a zero can mean")

    unread = p.filter(~that_week & (pl.col("roof") == "outdoors") & pl.col("wind").is_null())
    domed = p.filter(pl.col("roof").is_in(["dome", "closed"]))
    assert unread.height > 0 and domed.height > 0, (
        "the capture must hold an unread open-air game and an indoor one, or the three-way "
        "distinction below is being asserted over two states")
    assert domed["wind"].null_count() == domed.height, "a dome reads null, not calm"
    assert set(unread["roof"].unique()) == {"outdoors"} and not set(
        domed["roof"].unique()) & {"outdoors"}, (
        "`roof` is what separates an unread outdoor game from an indoor one; without it the "
        "two nulls would be the same row and criterion two would need a new column")


def test_the_injected_preseason_board_is_pre_kickoff_and_not_unclassified(monkeypatch,
                                                                          tmp_path):
    """`spec.ranks` is caller-supplied, so it is the one join whose columns come from outside.

    An August **Consensus** opinion about a September season is as pre-kickoff as a fact gets,
    and `hub.season.weekly_gate_data.preseason_ranks` is the caller that injects it. Left
    undeclared, `_served` would refuse the gate's own panel -- so this is the case where the
    boundary check has to be right about a column this module does not itself build.
    """
    arc.install(monkeypatch, tmp_path)
    board = arc.frame("draft_board")
    ranks = board.select(
        pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("key"),
        pl.lit(2024).cast(pl.Int64).alias("season"),
        pl.col("ecr").alias("preseason_ecr")).drop_nulls("preseason_ecr")
    p = pnl.build_panel(arc.SEASONS, pnl.PanelSpec(consensus=False, ranks=ranks))
    assert pnl.column_role("preseason_ecr") == "pre-kickoff"
    assert "preseason_ecr" in pnl.feature_columns(p)
    pnl.require_features(p, ["preseason_ecr"])


# The extra column a source grows in the two tests below. A realised quantity on purpose:
# yards after the catch is week w's own play, so a Panel that served it unremarked is exactly
# the leak #203 is about.
_GROWN = "yards_after_catch"


def _snap_share_that_grew_a_column(monkeypatch):
    """`snap_share`, returning one more realised column than it used to.

    Monkeypatched at this repo's own function rather than at the capture, because every source
    narrows with an explicit `select` -- so the way a raw outcome column really starts
    surviving to the return is an edit *there*: a new join, a new total, or a source joined
    whole after it grew a column upstream. This stands in for that edit. Same technique as
    `test_the_panel_only_degrades_around_a_contract_violation` above.
    """
    real = pnl.snap_share

    def grown(seasons):
        return real(seasons).with_columns(pl.lit(1.0).alias(_GROWN))

    monkeypatch.setattr(pnl, "snap_share", grown)


def test_the_panel_refuses_to_serve_a_column_that_is_on_neither_side(monkeypatch, tmp_path):
    """**The load-bearing one.** A new raw outcome column reaching the return is refused.

    A test that checked today's known columns against a list would pass forever while the
    next one slipped through, so this adds one the module has never heard of and asserts the
    Panel will not hand it back. The default for an unclassified column is unsafe, which is
    the only direction a leakage guard may fail in: a column nobody has classified is one a
    caller can judge by its suffix and nothing else, which is the defect itself.

    The refusal has to be actionable, so it names the column and where to declare it. A
    reader who meets this message is being told to make a decision, not to delete a line.
    """
    arc.install(monkeypatch, tmp_path)
    _snap_share_that_grew_a_column(monkeypatch)
    with pytest.raises(pnl.PanelRuleViolation) as e:
        pnl.build_panel(arc.SEASONS)
    msg = str(e.value)
    assert _GROWN in msg, "the refusal must name the column that caused it"
    assert "OUTCOMES" in msg and "PRE_KICKOFF" in msg, \
        "and where to put it, or the next reader's cheapest fix is to delete the guard"


def test_declaring_the_grown_column_serves_it_and_refuses_it_as_a_feature(monkeypatch,
                                                                          tmp_path):
    """The other half: the refusal above is a fork in the road, not a dead end.

    Without this, "the Panel refuses" is satisfiable by a Panel that refuses everything, and
    the test above would pass against a guard that had simply broken the assembly. Declaring
    the grown column an outcome is the resolution its own message asks for, and what that buys
    is the distinction the issue is about -- it is served, a fit could use it as an outcome,
    and `require_features` refuses it where a feature was meant.
    """
    arc.install(monkeypatch, tmp_path)
    _snap_share_that_grew_a_column(monkeypatch)
    monkeypatch.setattr(pnl, "OUTCOMES", (*pnl.OUTCOMES, _GROWN))
    p = pnl.build_panel(arc.SEASONS)
    assert _GROWN in p.columns, "declared, it is served -- the raw columns are kept on purpose"
    assert _GROWN in pnl.outcome_columns(p) and _GROWN not in pnl.feature_columns(p)
    with pytest.raises(pnl.PanelRuleViolation, match=_GROWN):
        pnl.require_features(p, [_GROWN])


def test_the_modules_outcome_set_agrees_with_what_the_captures_supply(monkeypatch, tmp_path):
    """Two derivations of the same set, meeting. Neither is a restatement of the other.

    `panelarchive.play_derived_columns` reads the captured `player_stats` and `snap_counts`
    frames; `panel.OUTCOMES` reads this module's own declarations. A column the module calls
    an outcome that the captures do not supply from week-w play is a misclassification in one
    direction, and a play-derived column the module does not account for is the other.

    The two sets are not equal and should not be: the captures also carry the row's own
    address (`player_id`, `season`, `week`, ...) and the opponent, which are facts about the
    fixture rather than about the play.
    """
    arc.install(monkeypatch, tmp_path)
    p = pnl.build_panel(arc.SEASONS)
    captured = arc.play_derived_columns()
    assert set(pnl.outcome_columns(p)) <= captured, (
        f"{sorted(set(pnl.outcome_columns(p)) - captured)} are called week-w outcomes by the "
        f"module and are not week-w play in the captures")
    unaccounted = ((set(p.columns) & captured) - set(pnl.outcome_columns(p))
                   - set(pnl.IDENTITY) - {"opponent_team"})
    assert not unaccounted, (
        f"{sorted(unaccounted)} come off a week-w capture and reach the Panel as something "
        f"other than an outcome, the row's address, or the opponent")


def test_the_opt_in_specs_raw_columns_are_declared_though_the_archive_cannot_drive_them():
    """`routes` and `scheme` join raw week-w measurements too, and this capture cannot run them.

    `route_pct` is the share of week w's charted pass plays he was on the field for, and the
    scheme rates are how the offence actually played week w -- each joined onto the Panel
    beside the `_trend` built from it, which is the same shape as `offense_pct`/`snap_trend`
    and the same reach. The fixtures README records why the capture cannot drive either spec,
    so the classification is asserted here rather than through a Panel that cannot be built.
    """
    for c in ("route_pct", "pass_rate", *pnl.SCHEME):
        assert pnl.column_role(c) == "outcome", \
            f"`{c}` is measured on week w's own plays and nothing says so"
        assert pnl.column_role(f"{c}_trend") == "derived", \
            f"`{c}_trend` is the feature built from it and must stay reachable"




# --- every source goes through the validated, cached path (#35) ----------------------------
#
# The path is `hub.fetch.nflverse.load`. What it buys over reaching `nflreadpy` from this
# module is the source's `Contract`, a cache entry keyed by `(source, seasons, columns,
# as-of)`, and a `Pin` beside that entry -- which together are what makes "a panel built twice
# at one as-of is frame-identical" a claim anything can check.
#
# Three tests, each holding one of the ticket's three criteria. The scan says no source is
# fetched directly; the watched build says each source the archive can drive is loaded *and
# pinned* by an assembly, not merely importable; the double build says the second Panel at
# one as-of is the first one's bytes, read back from the entries the first one pinned.
# Until #234 the second of these was a pair: a test naming which three sources still reached
# `nflreadpy`, and one asserting *why* -- that their captures were missing contract columns --
# which went red the day the archive was re-taken, as designed, and was replaced by this.

_NFLREADPY_TO_SOURCE = {
    "load_pbp": "pbp", "load_player_stats": "player_stats",
    "load_ff_opportunity": "ff_opportunity", "load_schedules": "schedules",
    "load_participation": "participation", "load_ftn_charting": "ftn_charting",
    "load_injuries": "injuries", "load_snap_counts": "snap_counts",
    "load_ff_rankings": "ff_rankings",
}

# The six nflverse sources the frozen archive can drive a Panel from. `participation` and
# `ftn_charting` are the two it deliberately refuses (play-level; see the fixtures README), so
# they are routed but not watched here.
_ARCHIVE_SOURCES = ("player_stats", "ff_opportunity", "ff_rankings",
                    "schedules", "snap_counts", "injuries")


def _sources_reaching_nflreadpy_directly() -> set[str]:
    """Every nflverse source `hub.models.panel` still fetches without going through `load`.

    An AST scan rather than a grep: `nfl.load_schedules(...)` inside a comment or a docstring
    is not a call, and this module is full of prose naming these functions. Same technique as
    `test_experiment.py`'s expanding-window scan, and for the same reason -- what is asserted
    is a property of the code, and a string search asserts a property of the text.
    """
    tree = ast.parse(Path(inspect.getfile(pnl)).read_text())
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _NFLREADPY_TO_SOURCE
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "nfl"):
            found.add(_NFLREADPY_TO_SOURCE[node.func.attr])
    return found


def test_no_source_reaches_nflreadpy_directly_from_the_panel():
    """The first criterion: no source in the Panel is fetched directly.

    A *new* unrouted source -- a join added, a source grown, someone reaching for `nfl.` out
    of habit -- lands in this set and fails here, rather than reaching the live Sunday path as
    a frame no contract has seen.
    """
    direct = _sources_reaching_nflreadpy_directly()
    assert direct == set(), (
        f"{sorted(direct)} are fetched from `nflreadpy` inside hub.models.panel. Every source "
        f"goes through `hub.fetch.nflverse.load` -- validated, cached and pinned -- and the "
        f"archive in tests/golden/fixtures/panel_archive/ carries every contract's columns, "
        f"so nothing blocks that any more.")


def _watched_build(monkeypatch, tmp_path, **spec) -> tuple[list[str], tuple]:
    """One Panel, with the sources the loader was asked for and the pins it left behind."""
    import hub.fetch.nflverse as nv
    real, seen = nv.load, []

    def watched(source, seasons, *a, **kw):
        seen.append(source)
        return real(source, seasons, *a, **kw)

    arc.install(monkeypatch, tmp_path)
    monkeypatch.setattr(nv, "load", watched)
    with nv.reads_of_one_run():
        pnl.build_panel(arc.SEASONS, pnl.PanelSpec(**spec))
        pins = nv.pins_this_run()
    return seen, pins


@pytest.mark.parametrize("source", _ARCHIVE_SOURCES)
def test_each_source_the_archive_drives_is_loaded_and_pinned_by_a_build(monkeypatch, tmp_path,
                                                                        source):
    """The behavioural half of the first criterion. The scan reads the source; this watches
    a build, and what it pins is that the loader is on the path a Panel is actually assembled
    by -- not merely imported by the module -- and that the build can name what it read.

    `ff_rankings` arrives through `load_rankings`, which is `load` under another name. The
    pin is asserted as well as the call because the pin is the whole point of the route: a
    read the run cannot name is one whose bytes a digest cannot deny.
    """
    seen, pins = _watched_build(monkeypatch, tmp_path, expected=True)
    assert source in seen, (
        f"a Panel was assembled and the loader saw only {sorted(set(seen))}; `{source}` "
        f"reached nflreadpy round the side, unvalidated, uncached and unpinned")
    pinned = {p.source for p in pins if hasattr(p, "source")}
    assert source in pinned, (
        f"the loader was asked for `{source}` and left no pin: the run read bytes it cannot "
        f"name, and the digest over its reads is short by one source while looking complete")


def test_a_panel_built_twice_at_one_as_of_is_the_first_ones_bytes_read_back(monkeypatch,
                                                                            tmp_path):
    """The second criterion: a Panel built twice at one as-of is frame-identical.

    Asserted as the mechanism and not only the outcome. Two builds under one `RAW` at one
    as-of: the second must fetch nothing -- every source is served from the entry the first
    one wrote -- and read back exactly the pins the first one left, so the two Panels are the
    same bytes because they are the same reads. Row order is sorted before the compare only
    because a `group_by` upstream of a join emits in hash order; every value is compared.

    The fetch count is what a mutation of `load`'s cache check fails on: with the cache
    ignored, the second build refetches six sources and this says so.
    """
    import hub.fetch.nflverse as nv
    real_fetch, fetched = nv._fetch, []

    def counted(source, keys):
        fetched.append(source)
        return real_fetch(source, keys)

    arc.install(monkeypatch, tmp_path)
    monkeypatch.setattr(nv, "_fetch", counted)
    keys, spec = ["player_id", "season", "week"], pnl.PanelSpec(expected=True)
    with nv.reads_of_one_run():
        first = pnl.build_panel(arc.SEASONS, spec, as_of="2024-12-31").sort(keys)
        first_pins = nv.pins_this_run()
    assert len(fetched) >= len(_ARCHIVE_SOURCES), "the first build fetches every source"
    del fetched[:]
    with nv.reads_of_one_run():
        second = pnl.build_panel(arc.SEASONS, spec, as_of="2024-12-31").sort(keys)
        second_pins = nv.pins_this_run()
    assert fetched == [], (
        f"the second build at the same as-of fetched {fetched} again; a re-run that refetches "
        f"is one whose result depends on what upstream holds that minute, not on the as-of")
    assert second_pins == first_pins, "the second build reads back the pins the first left"
    assert first.equals(second), "and so it is the first Panel's bytes, every cell"
