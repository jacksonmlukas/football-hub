

# --- the year was hardcoded twice on one call path -------------------------

def test_the_season_reaches_the_league_it_filters_against(monkeypatch):
    """`player_market(season=...)` filtered stat blocks by season while `league_settings`
    built `League(year=2026)` as a literal. Asking for 2027 fetched the 2026 league, matched
    no stat block, and returned a full frame with proj_ppg all null and no error -- and
    `proj_blend` coalesces to `xfp_per_game`, so the board built, THE PICK worked, and
    ESPN's projection had silently stopped contributing."""
    from hub.fetch import espn
    seen = {}

    class _Req:
        def league_get(self, params=None, headers=None):
            return {"players": [{"player": {
                "fullName": "A", "injuryStatus": "ACTIVE",
                "ownership": {"averageDraftPosition": 3.0},
                "stats": [{"statSourceId": 1, "statSplitTypeId": 0,
                           "seasonId": seen["year"], "appliedAverage": 15.0}]}}]}

    class _L:
        espn_request = _Req()

    def _settings(year=espn.SEASON_AHEAD, league_id=None):
        seen["year"] = year
        return espn.LeagueView(_L(), {})

    monkeypatch.setattr(espn, "league_settings", _settings)
    got = espn.player_market(season=2027)
    assert seen["year"] == 2027, "the League must be built for the season being asked for"
    assert got["proj_ppg"][0] == 15.0, "and the projection must survive the filter"


def test_league_settings_returns_one_object_not_a_tuple():
    """Every call site discarded half the tuple -- three spelled it `lg, _` and one
    `_, slots` -- so it was never carrying two things anybody wanted at once."""
    from hub.fetch import espn
    v = espn.LeagueView("league-handle", {"RB": 2})
    assert v.league == "league-handle" and v.roster_slots == {"RB": 2}


def test_the_season_has_one_owner():
    """It was five literals: board.SEASON_AHEAD, league_settings' League(year=2026),
    state.sync_from_espn's `year or 2026`, playoff_sos' default and publish --season."""
    import inspect

    from hub.config import SEASON_AHEAD
    from hub.draft import board, playoff_sos
    from hub.fetch import espn
    assert board.SEASON_AHEAD is SEASON_AHEAD
    assert inspect.signature(espn.league_settings).parameters["year"].default == SEASON_AHEAD
    assert (inspect.signature(playoff_sos.playoff_sos)
            .parameters["season_ahead"].default == SEASON_AHEAD)
    for mod in (espn, board):
        assert "= 2026" not in inspect.getsource(mod), f"{mod.__name__} kept a literal"
