"""ESPN's projection for one scoring period, published beside `mu` (#492).

The roster parse used to read the season split only (`espn_avg`, `espn_total`). The weekly
number sits in the same payload under its own scoring period, and `roster_rows` now carries it.
"""
import hub.fetch.espn as E


class _Player:
    def __init__(self, name, position, stats=None):
        self.name, self.position, self.lineupSlot = name, position, "BE"
        self.injuryStatus, self.playerId, self.proTeam = "ACTIVE", 1, "GB"
        self.projected_avg_points, self.projected_total_points = 10.0, 170.0
        if stats is not None:
            self.stats = stats


class _Team:
    def __init__(self, players):
        self.owners, self.roster, self.team_id = [], players, 1


def test_the_weekly_projection_is_read_from_the_asked_for_scoring_period():
    """The season block (period 0) sits beside the weekly ones and is ten times larger; the
    weekly column must be the week asked for, not the first block found."""
    p = _Player("A Receiver", "WR", {
        0: {"projected_points": 290.0, "projected_avg_points": 17.1},
        7: {"projected_points": 18.4}, 8: {"projected_points": 9.9}})
    got = E.roster_rows(_Team([p]), week=7).row(0, named=True)
    assert got["espn_week"] == 7
    assert got["espn_week_proj"] == 18.4


def test_a_missing_or_zero_weekly_projection_is_null_not_zero():
    other_week = _Player("A B", "WR", {0: {"projected_points": 290.0}})
    zero = _Player("C D", "RB", {7: {"projected_points": 0.0}})
    no_stats = _Player("E F", "TE")
    got = E.roster_rows(_Team([other_week, zero, no_stats]), week=7)
    assert got["espn_week_proj"].to_list() == [None, None, None]


def test_no_week_means_no_weekly_projection():
    p = _Player("A B", "WR", {7: {"projected_points": 12.0}})
    got = E.roster_rows(_Team([p])).row(0, named=True)
    assert got["espn_week"] is None and got["espn_week_proj"] is None
