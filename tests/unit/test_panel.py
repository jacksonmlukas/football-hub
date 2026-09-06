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
import datetime as dt

import numpy as np
import panelarchive as arc
import polars as pl
import pytest

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
    """What the screen prints as its data digest, at the seam that produces it.

    `consensus_pin` names the same cache entry `weekly_consensus` read -- source, page,
    columns and as-of -- so a run's provenance describes the load it claims to describe.
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
    columns and the two totals built from them -- and `panelarchive.play_derived_columns`
    derives that exempt set from the captures rather than listing it, so a feature added to
    the Panel is tested rather than quietly exempted.
    """
    base, after = _both_panels(monkeypatch, tmp_path, season, week, name)
    feats = arc.features(base)
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
    feats = arc.features(base)
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
