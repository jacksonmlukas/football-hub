"""Synthetic objects shaped like `espn_api`'s League, for the league-capture tests (#545).

Nothing here loads a League or touches the network: `SimpleNamespace`s carrying the attributes
`hub.fetch.league` reads.
"""
from types import SimpleNamespace as NS


def player(pid, name, pos, stats=None):
    return NS(name=name, position=pos, playerId=pid, proTeam="DAL", lineupSlot="BE",
              injuryStatus="ACTIVE", projected_avg_points=8.0, projected_total_points=80.0,
              stats=stats or {})


def make_league(deadline_ms=1733100000000, week=10, n=4, reg=14, playoff=4):
    teams = [NS(team_id=i + 1, team_name=f"Team {i + 1}", wins=i, losses=9 - i, ties=0,
                points_for=1000.0 + i, owners=[{"id": f"{{SWID-{i + 1}}}"}],
                roster=[player(100 * (i + 1), f"Player {i + 1}", "RB",
                               {10: {"projected_points": 11.5}, 11: {"projected_points": 9.0},
                                12: {"projected_points": 0.0}})],
                schedule=[]) for i in range(n)]
    for t in teams:        # opponent for each week 1..reg: never itself, cycling through the rest
        t.schedule = [teams[(t.team_id - 1 + 1 + w % (n - 1)) % n] for w in range(reg)]
    settings = NS(reg_season_count=reg, playoff_team_count=playoff, trade_deadline=deadline_ms)
    return NS(teams=teams, settings=settings, current_week=week, year=2026)
