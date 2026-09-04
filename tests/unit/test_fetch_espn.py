"""What `hub.fetch.espn` reads off ESPN, and the vocabulary it owns.

The identity and roster-payload tests below moved here from `test_roster.py` on 2026-09-04,
with the code. `hub.season.roster` used to duck-type the vendor object itself, which left
ESPN's season projection with two readers -- `_parse_projection` here and
`projected_avg_points` over in `season/` -- in packages that shared no code.
"""
import pytest

from hub.fetch import espn as E

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


# --- graceful degradation, which is a hard rule and was untested ------------
#
# CLAUDE.md: "If a fetch fails, serve last-good state from data/processed/ rather than
# erroring. Systems that need an operator die in October." `_get` implements exactly that --
# four host/user-agent combinations, then the cache -- and none of it was covered.


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"{self.status}")

    def json(self):
        return self._payload


def _cache_at(monkeypatch, tmp_path):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "CACHE", tmp_path)
    return espn


def test_a_403_on_the_first_host_falls_through_to_the_next(monkeypatch, tmp_path):
    """Aug 2026: site.api.espn.com started 403ing scripted traffic. The fix was to try
    `.web` in the host and a browser UA -- four combinations, in order."""
    espn = _cache_at(monkeypatch, tmp_path)
    tried = []

    def _get(url, params=None, headers=None, timeout=None):
        tried.append((url.split("/")[2], (headers or {})["User-Agent"]))
        return _Resp({"ok": True}, 200 if len(tried) > 1 else 403)
    monkeypatch.setattr(espn.requests, "get", _get)
    assert espn._get("football/nfl/scoreboard") == {"ok": True}
    assert tried[0][0] == "site.web.api.espn.com", "the working host is tried first"
    assert len(tried) == 2


def test_a_successful_pull_is_cached_for_the_next_outage(monkeypatch, tmp_path):
    espn = _cache_at(monkeypatch, tmp_path)
    monkeypatch.setattr(espn.requests, "get",
                        lambda *a, **k: _Resp({"events": [1]}))
    espn._get("football/nfl/scoreboard", cache_key="sb_nfl_now")
    assert (tmp_path / "sb_nfl_now.json").exists()


def test_all_four_combinations_failing_serves_the_cache(monkeypatch, tmp_path):
    """The rule itself. A dead endpoint must not raise into the dashboard."""
    espn = _cache_at(monkeypatch, tmp_path)
    (tmp_path / "sb_nfl_now.json").write_text('{"events": ["stale"]}')

    calls = {"n": 0}

    def _dead(*a, **k):
        calls["n"] += 1
        raise OSError("connection refused")
    monkeypatch.setattr(espn.requests, "get", _dead)
    assert espn._get("football/nfl/scoreboard", cache_key="sb_nfl_now") == {"events": ["stale"]}
    assert calls["n"] == 4, "two hosts times two user agents, then the cache"


def test_no_cache_and_no_network_raises_rather_than_returning_a_lie(monkeypatch, tmp_path):
    """Degradation serves last-good; it does not invent an empty payload. An empty
    scoreboard would read as 'no games today'."""
    import pytest
    espn = _cache_at(monkeypatch, tmp_path)
    monkeypatch.setattr(espn.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="ESPN unreachable and no cache"):
        espn._get("football/nfl/scoreboard", cache_key="missing")


def test_cfb_asks_for_fbs_only_and_a_full_saturday(monkeypatch, tmp_path):
    """The default page size truncates a 60-game Saturday, and group 80 is FBS."""
    espn = _cache_at(monkeypatch, tmp_path)
    seen = {}

    def _get(url, params=None, headers=None, timeout=None):
        seen.update(params or {})
        return _Resp({"events": []})
    monkeypatch.setattr(espn.requests, "get", _get)
    espn.scoreboard("cfb", date="20261003")
    assert seen == {"dates": "20261003", "groups": "80", "limit": "200"}


def test_summary_is_keyed_by_event_so_two_games_do_not_share_a_cache_entry(monkeypatch, tmp_path):
    espn = _cache_at(monkeypatch, tmp_path)
    monkeypatch.setattr(espn.requests, "get", lambda *a, **k: _Resp({"winprobability": []}))
    espn.summary("401671", "nfl")
    espn.summary("401672", "nfl")
    assert {p.name for p in tmp_path.glob("*.json")} == {"sum_401671.json", "sum_401672.json"}


def _event(competitors, event_id="1"):
    """One scoreboard event. `competitors` is handed in whatever order the test is about."""
    return {"id": event_id, "competitions": [{
        "status": {"type": {"state": "in", "shortDetail": "Q3 4:12"}},
        "competitors": competitors,
        "situation": {"possession": "1", "downDistanceText": "2nd & 7"}}]}


# PHI at home, DAL away, said explicitly -- which is how ESPN says it.
_PHI = {"homeAway": "home", "team": {"abbreviation": "PHI"}, "score": "21"}
_DAL = {"homeAway": "away", "team": {"abbreviation": "DAL"}, "score": "17"}


def test_live_state_flattens_what_the_overlay_needs(monkeypatch, tmp_path):
    espn = _cache_at(monkeypatch, tmp_path)
    monkeypatch.setattr(espn, "scoreboard",
                        lambda league="nfl": {"events": [_event([_PHI, _DAL])]})
    got = espn.live_state()[0]
    assert got["home"] == "PHI" and got["away"] == "DAL"
    assert got["home_score"] == "21" and got["away_score"] == "17"
    assert got["state"] == "in" and got["down_distance"] == "2nd & 7"


def test_the_side_comes_from_the_field_that_names_it_not_from_the_order(monkeypatch,
                                                                        tmp_path):
    """The array order is undocumented and not guaranteed. Reading position instead of the
    field swaps both teams and both scores, and the overlay is the one artifact that moves
    during a Sunday -- there is nothing on the page to compare it against, so it renders as
    a plausible scoreline that is exactly backwards."""
    espn = _cache_at(monkeypatch, tmp_path)

    def state(order):
        monkeypatch.setattr(espn, "scoreboard",
                            lambda league="nfl": {"events": [_event(order)]})
        return espn.live_state()[0]

    assert state([_PHI, _DAL]) == state([_DAL, _PHI])
    assert state([_DAL, _PHI])["home"] == "PHI"
    assert state([_DAL, _PHI])["home_score"] == "21"


def test_an_event_that_does_not_name_its_sides_is_left_out_and_said_so(monkeypatch,
                                                                       tmp_path, capsys):
    """Degrade by saying so, not by guessing. Emitting the game with a side picked off the
    array would put a possibly-inverted scoreline on the page, which is worse than a game
    the panel does not show."""
    espn = _cache_at(monkeypatch, tmp_path)
    bare = [{"team": {"abbreviation": "KC"}, "score": "3"},
            {"team": {"abbreviation": "LV"}, "score": "0"}]
    monkeypatch.setattr(espn, "scoreboard", lambda league="nfl": {
        "events": [_event([_PHI, _DAL]), _event(bare, event_id="2")]})
    got = espn.live_state()
    assert [g["id"] for g in got] == ["1"], "the good game still renders"
    assert "2" in capsys.readouterr().out


def test_two_competitors_on_the_same_side_is_not_a_game_either(monkeypatch, tmp_path):
    """A payload naming two home teams names no away team. Taking the first of each would
    invent an opponent."""
    espn = _cache_at(monkeypatch, tmp_path)
    both = [_PHI, {**_DAL, "homeAway": "home"}]
    monkeypatch.setattr(espn, "scoreboard",
                        lambda league="nfl": {"events": [_event(both)]})
    assert espn.live_state() == []


# --- the tiered poller ------------------------------------------------------
#
# One scoreboard request per tick; the per-game summary fan-out only every Nth tick and
# capped at twelve. This is the loop `hub.draft.live.poll` was rewritten to match.


def _run_poll(monkeypatch, espn, ticks, **kw):
    """Run `poll` for `ticks` ticks by making the sleep raise on the last one."""
    n = {"i": 0}

    def _sleep(_):
        n["i"] += 1
        if n["i"] >= ticks:
            raise KeyboardInterrupt
    monkeypatch.setattr(espn.time, "sleep", _sleep)
    try:
        espn.poll(0, **kw)
    except KeyboardInterrupt:
        pass


def test_the_poller_writes_one_file_per_tick(monkeypatch, tmp_path):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [{"state": "in"}])
    out = tmp_path / "live.json"
    _run_poll(monkeypatch, espn, 2, out=out)
    import json
    # `rows`, not `games`: the poller writes the same envelope the site writer does, so the
    # page can read whichever ran last. See tests/contracts/test_live_artifact_shape.py.
    assert json.loads(out.read_text())["rows"] == [{"state": "in"}]


def test_the_summary_fanout_is_capped_at_twelve(monkeypatch, tmp_path):
    """Undocumented endpoints, and ESPN asks that request volume stay low."""
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [])
    asked = []
    monkeypatch.setattr(espn, "summary", lambda gid, lg="nfl": asked.append(gid) or
                        {"winprobability": [{"homeWinPercentage": 0.6}]})
    _run_poll(monkeypatch, espn, 1, out=tmp_path / "live.json",
              games_of_interest=[str(i) for i in range(30)])
    assert len(asked) == 12


def test_the_fanout_only_fires_every_nth_tick(monkeypatch, tmp_path):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [])
    asked = []
    monkeypatch.setattr(espn, "summary", lambda gid, lg="nfl": asked.append(gid) or
                        {"winprobability": [{"homeWinPercentage": 0.6}]})
    _run_poll(monkeypatch, espn, 4, out=tmp_path / "live.json",
              games_of_interest=["g1"], summary_every=4)
    assert len(asked) == 1, "ticks 0 of 0..3 only"


def test_one_bad_game_does_not_lose_the_others(monkeypatch, tmp_path):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [])

    def _summary(gid, lg="nfl"):
        if gid == "bad":
            raise RuntimeError("404")
        return {"winprobability": [{"homeWinPercentage": 0.7}]}
    monkeypatch.setattr(espn, "summary", _summary)
    out = tmp_path / "live.json"
    _run_poll(monkeypatch, espn, 1, out=out, games_of_interest=["bad", "good"])
    import json
    assert list(json.loads(out.read_text())["detail"]) == ["good"]


def test_the_poller_serves_stale_rather_than_dying(monkeypatch, tmp_path, capsys):
    """The precedent `hub.draft.live.poll` was rewritten to match (improvements.md #14)."""
    from hub.fetch import espn

    def _boom(league="nfl"):
        raise RuntimeError("scoreboard 500")
    monkeypatch.setattr(espn, "live_state", _boom)
    _run_poll(monkeypatch, espn, 3, out=tmp_path / "live.json")
    out = capsys.readouterr().out
    assert "poll error (serving stale)" in out


# --- the private-league reads, all of which need cookies -------------------


def test_the_league_is_built_for_the_year_asked_for(monkeypatch):
    """`League(year=...)` was the literal 2026. This is the constructor call itself, which
    the season-owner test above can only check through a stub."""
    import espn_api.football as espn_football

    from hub.fetch import espn
    seen = {}

    class _League:
        def __init__(self, **kw):
            seen.update(kw)
            self.settings = type("S", (), {"roster_slots": {"RB": 2}})()
    monkeypatch.setattr(espn_football, "League", _League)
    monkeypatch.setenv("ESPN_LEAGUE_ID", "123")
    monkeypatch.setenv("ESPN_S2", "s2")
    monkeypatch.setenv("ESPN_SWID", "swid")
    view = espn.league_settings(2027)
    assert seen["year"] == 2027 and seen["league_id"] == 123
    assert view.roster_slots == {"RB": 2}


def test_an_explicit_league_id_beats_the_environment(monkeypatch):
    """So a test, or a second league, never reads the one in .env by accident."""
    import espn_api.football as espn_football

    from hub.fetch import espn
    seen = {}

    class _League:
        def __init__(self, **kw):
            seen.update(kw)
            self.settings = type("S", (), {})()
    monkeypatch.setattr(espn_football, "League", _League)
    monkeypatch.setenv("ESPN_LEAGUE_ID", "999")
    espn.league_settings(2026, league_id=123)
    assert seen["league_id"] == 123


def test_a_league_with_no_roster_slots_yields_an_empty_map_not_a_crash(monkeypatch):
    import espn_api.football as espn_football

    from hub.fetch import espn

    class _League:
        def __init__(self, **kw):
            self.settings = type("S", (), {"roster_slots": None})()
    monkeypatch.setattr(espn_football, "League", _League)
    monkeypatch.setenv("ESPN_LEAGUE_ID", "1")
    assert espn.league_settings().roster_slots == {}


def test_scoring_settings_reads_the_league_rather_than_assuming(monkeypatch):
    """What turns `components.SCORING` from an assumption into something checkable."""
    from hub.fetch import espn
    stat_id = next(iter(espn.SCORING_STAT_IDS))
    name = espn.SCORING_STAT_IDS[stat_id]

    class _Req:
        def league_get(self, params=None, headers=None):
            return {"settings": {"scoringSettings": {"scoringItems": [
                {"statId": stat_id, "points": 0.5},
                {"statId": stat_id, "points": 0},          # zero weight: not a rule
                {"statId": -1, "points": 4.0},             # unknown stat: skipped
            ]}}}

    class _L:
        espn_request = _Req()
    monkeypatch.setattr(espn, "league_settings", lambda: espn.LeagueView(_L(), {}))
    assert espn.scoring_settings() == {name: 0.5}


def test_missing_scoring_settings_is_an_empty_map(monkeypatch):
    from hub.fetch import espn

    class _L:
        espn_request = type("R", (), {"league_get": lambda self, params=None: {}})()
    monkeypatch.setattr(espn, "league_settings", lambda: espn.LeagueView(_L(), {}))
    assert espn.scoring_settings() == {}


def test_league_history_unwraps_the_list_espn_returns(monkeypatch):
    """`leagueHistory` returns a one-element list; every other view returns an object."""
    from hub.fetch import espn
    monkeypatch.setenv("ESPN_LEAGUE_ID", "1")
    monkeypatch.setattr(espn.requests, "get", lambda *a, **k: _Resp([{"teams": []}]))
    assert espn.league_history(2024) == {"teams": []}


def test_the_player_filter_rides_only_on_the_player_view(monkeypatch):
    """`x-fantasy-filter` is what keeps kona_player_info from returning the whole database;
    sending it on mTeam is meaningless and was worth being sure about."""
    from hub.fetch import espn
    monkeypatch.setenv("ESPN_LEAGUE_ID", "1")
    seen = {}

    def _get(url, params=None, headers=None, cookies=None, timeout=None):
        seen[(params or {})["view"]] = headers
        return _Resp({"ok": True})
    monkeypatch.setattr(espn.requests, "get", _get)
    espn.league_history(2024, view="mTeam")
    espn.league_history(2024, view="kona_player_info")
    assert seen["mTeam"] is None
    assert "x-fantasy-filter" in (seen["kona_player_info"] or {})


class _Player:
    def __init__(self, name, position, slot="BE", injury="ACTIVE", pid=1, pro="GB",
                 avg=10.0, total=170.0):
        self.name, self.position, self.lineupSlot = name, position, slot
        self.injuryStatus, self.playerId, self.proTeam = injury, pid, pro
        # ESPN's two projections; their ratio is the availability signal. The defaults are a
        # full 17-game slate, so a test that says nothing about availability gets a fit player.
        self.projected_avg_points, self.projected_total_points = avg, total


class _Team:
    def __init__(self, owners, players, team_id=1):
        self.owners, self.roster, self.team_id = owners, players, team_id


class _League:
    def __init__(self, teams):
        self.teams = teams


# --- your team, and the seven fields anything downstream depends on ---
#
# Moved here from test_roster.py on 2026-09-04 with the code. The roster
# module used to duck-type the vendor object itself, which left ESPN's season
# projection with two readers in two packages that shared no code.

def test_the_team_is_the_one_whose_owner_matches_the_swid():
    a = _Team([{"id": "{AAA}"}], [], team_id=3)
    b = _Team([{"id": "{BBB}"}], [], team_id=14)
    assert E.my_team(_League([a, b]), swid="BBB").team_id == 14


def test_swid_matching_ignores_braces_and_case():
    t = _Team([{"id": "{abc-123}"}], [], team_id=7)
    assert E.my_team(_League([t]), swid="ABC-123").team_id == 7


def test_a_swid_owning_no_team_raises_rather_than_picking_one():
    """A wrong team here computes cleanly against sixteen players who are not yours."""
    with pytest.raises(LookupError, match="no team in this league"):
        E.my_team(_League([_Team([{"id": "{AAA}"}], [])]), swid="ZZZ")


def test_an_absent_swid_is_its_own_error():
    with pytest.raises(LookupError, match="ESPN_SWID is not set"):
        E.my_team(_League([]), swid="")


def test_a_team_owned_by_several_people_still_matches():
    t = _Team([{"id": "{AAA}"}, {"id": "{BBB}"}], [], team_id=9)
    assert E.my_team(_League([t]), swid="BBB").team_id == 9


# --- reading the roster off ESPN ---

def test_every_attribute_the_producer_depends_on_is_read():
    got = E.roster_rows(_Team([], [_Player("Ja'Marr Chase", "WR", "WR", "QUESTIONABLE", 42, "CIN")]))
    r = got.row(0, named=True)
    assert r["player"] == "Ja'Marr Chase"
    assert (r["pos"], r["slot"], r["injury_status"]) == ("WR", "WR", "QUESTIONABLE")
    assert (r["espn_id"], r["nfl_team"]) == (42, "CIN")


def test_an_empty_roster_yields_the_right_empty_shape():
    got = E.roster_rows(_Team([], []))
    assert got.height == 0
    assert list(got.columns) == list(E.ROSTER_SCHEMA)
