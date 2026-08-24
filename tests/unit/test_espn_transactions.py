"""Does this league actually trade?

`docs/championship-leverage.md` gates its L4 trade evaluator on this and is explicit about
not guessing: *"Pull two or three prior seasons and count actual trades. A twenty-minute
spike that decides whether L4 is worth building."*

The obvious route does not work. `espn_api`'s `recent_activity` returns HTTP 400 for every
past season of this league, because it reads the `communication/` endpoint and ESPN 404s
that for anything but the current season. The per-team `transactionCounter` in the `mTeam`
view survives in `leagueHistory` and carries the same counts, so that is what this reads.

The one arithmetic trap is that a trade increments the counter on *both* teams, so the
league total is half the sum. Getting that wrong doubles the answer and could flip the
build/skip decision, so it is pinned here.
"""
import polars as pl
import pytest

from hub.fetch import espn


def _payload(rows):
    """rows: (team_name, trades, acquisitions, drops)"""
    return {"teams": [{"id": i, "name": r[0],
                       "transactionCounter": {"trades": r[1], "acquisitions": r[2],
                                              "drops": r[3]}}
                      for i, r in enumerate(rows)]}


def _fake(by_season):
    def fetch(season, view="mTeam"):
        return by_season[season]
    return fetch


def test_counts_come_back_per_team_per_season():
    got = espn.transaction_counts(
        [2024], fetch=_fake({2024: _payload([("A", 2, 20, 19), ("B", 2, 15, 14)])}))
    assert got.height == 2
    assert set(got.columns) >= {"season", "team", "trades", "acquisitions", "drops"}


def test_a_trade_is_counted_once_for_the_league_not_once_per_side():
    """The trap. Both teams' counters increment, so a league that made exactly one trade
    shows a sum of two. Reporting two would double every number in the spike."""
    counts = espn.transaction_counts(
        [2024], fetch=_fake({2024: _payload([("A", 1, 0, 0), ("B", 1, 0, 0),
                                             ("C", 0, 0, 0)])}))
    assert espn.trade_summary(counts)["trades"][2024] == 1


def test_waiver_activity_is_not_halved():
    """An acquisition has one side. Halving it too would be the same fix misapplied."""
    counts = espn.transaction_counts(
        [2024], fetch=_fake({2024: _payload([("A", 0, 30, 28), ("B", 0, 20, 19)])}))
    assert espn.trade_summary(counts)["acquisitions"][2024] == 50


def test_an_odd_trade_sum_is_flagged_rather_than_rounded_away():
    """Halving assumes every trade has exactly two sides. An odd total means that
    assumption broke -- a three-way trade, or a team removed from league history -- and
    quietly rounding would hide it."""
    counts = espn.transaction_counts(
        [2024], fetch=_fake({2024: _payload([("A", 1, 0, 0), ("B", 1, 0, 0),
                                             ("C", 1, 0, 0)])}))
    assert 2024 in espn.trade_summary(counts)["uneven_seasons"]


def test_seasons_are_kept_separate():
    counts = espn.transaction_counts(
        [2023, 2024], fetch=_fake({2023: _payload([("A", 2, 5, 5), ("B", 2, 5, 5)]),
                                   2024: _payload([("A", 0, 9, 9), ("B", 0, 9, 9)])}))
    s = espn.trade_summary(counts)
    assert s["trades"] == {2023: 2, 2024: 0}


def test_a_missing_counter_is_zero_not_a_crash():
    """Older seasons drop fields. A KeyError here would kill the whole spike over one
    team."""
    got = espn.transaction_counts(
        [2024], fetch=lambda season, view="mTeam": {
            "teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "B",
                                               "transactionCounter": {"trades": 2}}]})
    assert got["trades"].to_list() == [0, 2]
    assert got["acquisitions"].to_list() == [0, 0]


def test_a_season_that_returns_nothing_is_reported_not_silently_dropped():
    """A season counted as zero trades because the fetch came back empty would argue
    against building a trade evaluator on the strength of a network failure."""
    with pytest.raises(espn.NoLeagueHistory):
        espn.transaction_counts([2019], fetch=lambda season, view="mTeam": {"teams": []})
