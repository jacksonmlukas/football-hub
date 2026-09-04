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
import numpy as np
import polars as pl
import pytest

from hub import store
from hub.config import SEASON_AHEAD
from hub.models import ratings

# Both sources quote the market's spread on the home team, so they are comparable point for
# point. They are not the *same* quote: nflverse publishes one lookahead number per game and
# this repo's snapshot is a median across the books the pull returned. Observed 2026-09-04
# over 112 games carrying both: mean absolute difference 0.159, worst 3.0. Set with headroom
# so ordinary movement does not trip it and an inverted sign does.
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
    games = ratings.games_for(SEASON_AHEAD)
    both = games.filter(pl.col("snapshot_spread").is_not_null()
                        & pl.col("schedule_spread").is_not_null())
    if both.height < 10:
        pytest.skip(f"only {both.height} games carry both sources; too few to compare")
    return games, both


@pytest.mark.golden
def test_the_two_sources_agree_where_both_exist(priced):
    _, both = priced
    diff = (both["snapshot_spread"] - both["schedule_spread"]).to_numpy()
    assert float(np.abs(diff).mean()) < MEAN_TOLERANCE, (
        f"the dated snapshot and the moving field disagree by "
        f"{np.abs(diff).mean():.3f} points a game over {both.height} games. A flipped sign "
        f"convention in `hub.fetch.odds` is the first thing to check.")
    assert float(np.abs(diff).max()) < MAX_TOLERANCE, (
        f"worst game differs by {np.abs(diff).max():.1f} points")


@pytest.mark.golden
def test_they_agree_on_which_side_is_favoured(priced):
    """The disagreement a tolerance on magnitude would not catch. An inverted home/away
    mapping moves a near-pick-em game barely at all and reverses the prediction."""
    _, both = priced
    a = both["snapshot_spread"].to_numpy()
    b = both["schedule_spread"].to_numpy()
    clear = (np.abs(a) >= PICK_EM) & (np.abs(b) >= PICK_EM)
    assert clear.sum() >= 10, f"only {clear.sum()} games are priced away from pick-em"
    flipped = np.sign(a) * np.sign(b) < 0
    bad = both.filter(pl.Series(flipped & clear))
    assert bad.is_empty(), (
        f"{bad.height} games have the favourite on opposite sides: "
        f"{bad['game_id'].to_list()[:5]}")


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
    cov = ratings.coverage(games)
    assert sum(cov.values()) == games.height
    assert cov["snapshot"], "no game priced from a snapshot; the store or the as-of moment"


@pytest.mark.golden
def test_every_game_the_snapshot_priced_names_the_snapshot_that_did_it(priced):
    """"Which source" is only half of provenance. There are seven snapshots in this store
    and the row has to say which one, or re-deriving it means guessing."""
    games, _ = priced
    from_snap = games.filter(pl.col("price_source") == "snapshot")
    assert from_snap["priced_at"].null_count() == 0
