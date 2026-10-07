"""Nightly: the dated snapshots and the schedule's moving field are the same quantity.

This is the evidence that replaces a gate. Issue #6 proposed pricing predictions from the
stored snapshot rather than nflverse's `spread_line`, and the obvious instinct is to gate
the change on accuracy. It should not be gated. Both inputs are the *same* number -- the
betting market's spread on the home team -- so a two-half accuracy gate would compare two
nearly identical series, return "no detectable difference" by construction, and spend a
pre-registration and four seasons of harness saying so. That is the vacuous-gate trap
`docs/component-projection.md` already records on issue #1.

What the change needs instead is evidence the two agree where both exist, so that switching
does not silently reprice anything. Asserted here rather than measured once because the
claim decays silently: a bookmaker sign convention flipping, or `hub.fetch.odds` matching an
event to the wrong game, would widen the disagreement without breaking anything else.

**The fallback is deliberately not asserted against real data.** The ticket assumed far weeks
would have no snapshot when the fit first runs; measured on 2026-09-04 that is false in the
useful direction -- The Odds API is already posting week 18, so all 272 games price from a
snapshot and no real game takes the fallback. A golden test asserting the fallback fires
would therefore fail for a good reason. The fallback's own behaviour is pinned
deterministically in `tests/unit/test_ratings.py`, and what is checked here is the standing
*reason* it exists: the moving field alone leaves far weeks unpriced.

Marked `golden` and deselected by default for the reason the rest of this directory is --
a missing local store must not block a docs commit.

    uv run pytest -m golden -k line_agreement
"""
import polars as pl
import pytest

from hub import schedule, store
from hub.config import SEASON_AHEAD

# Both sources quote the market's spread on the home team, so they are comparable point for
# point. They are not the *same* quote: nflverse publishes one lookahead number per game and
# this repo's snapshot is a median across the books the pull returned. Observed 2026-09-04
# over 112 games carrying both: mean absolute difference 0.159, worst 3.0. Set with headroom
# so ordinary movement does not trip it and an inverted sign does.
#
# Restated 2026-10-07 (#440, rule 13; the paragraph above is kept as written). The figure
# was measured when every snapshot was days old and it was then applied to *every* game
# carrying both, however old its last poll. From week 2 that pairs a month-old lookahead
# quote with a line that has since moved: 93 games, mean 1.344 here and 2.336 on the CI run,
# split 0.138 on the 29 games with a poll inside `STALE_AFTER_DAYS` and 1.891 on the 64
# without, growing by week (wk1 0.63 ... wk4 2.63), signed mean ~0 -- movement, not a sign
# convention. The tolerance is NOT loosened: the comparison is scoped to `live` rows
# (`schedule.comparable_quotes`), the only ones where both sources are the same quote at the
# same time, and 0.138 sits under 0.5 as 0.159 did.
MEAN_TOLERANCE = 0.5
MAX_TOLERANCE = 5.0

# Below this the two sources can put the favourite on opposite sides without disagreeing
# about anything: a game the books price at -1.5 and nflverse at +1.5 is a pick-em to both.
# 2026_06_HOU_JAX is exactly that case, stable at -1.5 across all seven snapshots, and it is
# why the sign check is scoped rather than absolute.
PICK_EM = 2.0


@pytest.fixture(scope="module")
def priced():
    if "lines" not in store.tables():
        pytest.skip("no local store of dated lines; run `hub.fetch.odds --snapshot` first")
    # No `at`: the default is now in UTC, which is how `captured_at` is stamped. Passing a
    # local `datetime.now()` silently asks an as-of question four hours in the past and hides
    # the morning's snapshots -- which it did, until the coverage it reported disagreed with
    # the store.
    games = schedule.priced_games(SEASON_AHEAD)
    both = schedule.comparable_quotes(games)
    if both.height < 10:
        pytest.skip(f"only {both.height} games carry both sources from a live snapshot; "
                    f"too few to compare")
    return games, both


@pytest.mark.golden
def test_the_two_sources_agree_where_both_exist(priced):
    _, both = priced
    got = schedule.line_agreement(both, PICK_EM)
    assert got.mean_abs < MEAN_TOLERANCE, (
        f"the dated snapshot and the moving field disagree by "
        f"{got.mean_abs:.3f} points a game over {got.n} live games. A flipped sign "
        f"convention in `hub.fetch.odds` is the first thing to check.")
    assert got.max_abs < MAX_TOLERANCE, f"worst game differs by {got.max_abs:.1f} points"


@pytest.mark.golden
def test_they_agree_on_which_side_is_favoured(priced):
    """The disagreement a tolerance on magnitude would not catch. An inverted home/away
    mapping moves a near-pick-em game barely at all and reverses the prediction."""
    _, both = priced
    clear = both.filter((pl.col("snapshot_spread").abs() >= PICK_EM)
                        & (pl.col("schedule_spread").abs() >= PICK_EM))
    assert clear.height >= 10, f"only {clear.height} games are priced away from pick-em"
    bad = schedule.line_agreement(both, PICK_EM).flipped
    assert not bad, f"{len(bad)} games have the favourite on opposite sides: {bad[:5]}"


@pytest.mark.golden
def test_the_moving_field_alone_would_leave_far_weeks_unpriced(priced):
    """Why the coalesce runs in this direction. `spread_line` is a lookahead number that
    upstream has not filled in for the end of the season, so a fit reading only that field
    prices the near weeks and drops the far ones out of the slate entirely."""
    games, _ = priced
    late = games.filter(pl.col("week") >= 15)
    assert late["schedule_spread"].null_count(), (
        "spread_line now covers the late season; the fallback's justification has changed "
        "and this test should be re-read rather than deleted")
    assert late["close_spread"].null_count() < late["schedule_spread"].null_count()


@pytest.mark.golden
def test_the_coverage_report_accounts_for_every_game(priced):
    """The number reported each fit is the one a dead poller shows up in, so it has to be
    exhaustive rather than indicative."""
    games, _ = priced
    cov = schedule.by_source(games)
    assert sum(cov.values()) == games.height
    assert cov["live"] or cov["stale"], (
        "no game priced from a snapshot; the store or the as-of moment")


@pytest.mark.golden
def test_every_game_the_snapshot_priced_names_the_snapshot_that_did_it(priced):
    """"Which source" is only half of provenance. There are seven snapshots in this store
    and the row has to say which one, or re-deriving it means guessing."""
    games, _ = priced
    from_snap = games.filter(pl.col("price_source").is_in(["live", "stale"]))
    assert from_snap["priced_at"].null_count() == 0
    # and a row the moving field priced over a stale snapshot cites no snapshot (#281)
    assert games.filter(pl.col("price_source") == "schedule")["priced_at"].null_count() == (
        games.filter(pl.col("price_source") == "schedule").height)
