"""ESPN fetch layer. Two surfaces, both unofficial:

  1. site.api.espn.com  -> public scoreboard / summary / win probability (no auth)
  2. fantasy ESPN v3    -> your league, via espn_s2 + swid cookies

Both break without notice. Every call here retries across hostname and User-Agent
variants before giving up, and falls back to last-good cached state.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import requests

from hub import jsonio
from hub.config import SEASON_AHEAD
from hub.contracts import ESPN_SCOREBOARD

CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "espn"
CACHE.mkdir(parents=True, exist_ok=True)

# As of Aug 2026 site.api.espn.com started 403ing scripted traffic; adding `.web`
# to the host, or using a non-browser UA, resolves it. Try all four combinations.
HOSTS = ["site.web.api.espn.com", "site.api.espn.com"]
AGENTS = ["football-hub/0.1", "Mozilla/5.0"]

LEAGUE_PATHS = {"nfl": "football/nfl", "cfb": "football/college-football"}


def _get(path: str, params: dict | None = None, cache_key: str | None = None) -> dict:
    last = None
    for host in HOSTS:
        for ua in AGENTS:
            try:
                r = requests.get(
                    f"https://{host}/apis/site/v2/sports/{path}",
                    params=params or {},
                    headers={"User-Agent": ua},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                if cache_key:
                    (CACHE / f"{cache_key}.json").write_text(json.dumps(data))
                return data
            except Exception as e:
                last = e
    # Graceful degradation: serve last-good rather than raising into the dashboard.
    if cache_key and (CACHE / f"{cache_key}.json").exists():
        return json.loads((CACHE / f"{cache_key}.json").read_text())
    raise RuntimeError(f"ESPN unreachable and no cache for {path}: {last!r}")


def scoreboard(league: str = "nfl", date: str | None = None) -> dict:
    """date is YYYYMMDD, not YYYY-MM-DD. Future games return fewer fields."""
    params = {"dates": date} if date else {}
    if league == "cfb":
        params["groups"] = "80"  # FBS only
        params["limit"] = "200"  # default page size truncates a full Saturday
    return _get(f"{LEAGUE_PATHS[league]}/scoreboard", params, f"sb_{league}_{date or 'now'}")


def summary(event_id: str, league: str = "nfl") -> dict:
    """Box score, plays, and win probability. WP only exists once the feed is live."""
    return _get(f"{LEAGUE_PATHS[league]}/summary", {"event": event_id}, f"sum_{event_id}")


# The three columns the contract types. Given explicitly because a scoreboard with no games
# in progress carries all-null `possession` and `down_distance`, and polars infers those as
# `Null` -- which is its own dtype family and matches nothing, exactly the case `_family` was
# written for.
SCOREBOARD_TYPES: dict[str, Any] = {"id": pl.Utf8, "state": pl.Utf8, "home": pl.Utf8,
                                    "away": pl.Utf8}


def _sides(competitors: list[dict]) -> tuple[dict, dict] | None:
    """The home and away competitors, by the field that names them.

    ESPN puts `homeAway` on every competitor and documents nothing about the order of the
    array. Reading position instead -- which this did -- swaps both teams and both scores the
    first time a payload arrives away-first. That is the same class of error as an inverted
    spread sign in `store.verify`: the result stays plausible, and the overlay is the one
    artifact that moves during a Sunday, so there is nothing beside it on the page to
    contradict a scoreline that is exactly backwards.

    None when the payload does not name exactly one of each, which the caller reports rather
    than resolving. Picking the first of each side would invent an opponent.
    """
    by: dict[str, dict] = {}
    for c in competitors:
        side = str(c.get("homeAway", "")).lower()
        if side in ("home", "away"):
            by.setdefault(side, c)
    return (by["home"], by["away"]) if len(by) == 2 else None


def live_state(league: str = "nfl") -> list[dict]:
    """Flattened in-progress game state for the dashboard overlay."""
    out: list[dict] = []
    unnamed: list[str] = []
    for ev in scoreboard(league).get("events", []):
        c = ev["competitions"][0]
        st = c["status"]["type"]
        sides = _sides(c.get("competitors", []))
        if sides is None:
            # Left out rather than guessed at, and named so the omission is visible. A
            # missing game is a gap the page can show; a game with the wrong team winning
            # is not, because nothing on the page disagrees with it.
            unnamed.append(str(ev.get("id", "?")))
            continue
        home, away = sides
        out.append({
            "id": ev["id"],
            "state": st["state"],           # pre | in | post
            "detail": st.get("shortDetail"),
            "home": home["team"]["abbreviation"], "home_score": home.get("score"),
            "away": away["team"]["abbreviation"], "away_score": away.get("score"),
            "possession": c.get("situation", {}).get("possession"),
            "down_distance": c.get("situation", {}).get("downDistanceText"),
        })
    if unnamed:
        print(f"  live: {len(unnamed)} event(s) did not name home and away and are absent "
              f"from the overlay: {', '.join(unnamed)}")
    # Asserted at the boundary. This endpoint is undocumented and is now read unattended every
    # ten minutes through a game window, so drift -- a renamed field, a null where there was
    # never one, duplicated ids -- has to fail here rather than reach the page. Both callers
    # already degrade on an exception: `publish.live` keeps last-good, `poll` serves stale.
    # Typed even when empty: `pl.DataFrame([])` has no columns at all, and a contract cannot
    # tell "no games today" from "every field is gone" without them.
    ESPN_SCOREBOARD.validate(pl.DataFrame(out, schema_overrides=SCOREBOARD_TYPES) if out
                             else pl.DataFrame(schema=SCOREBOARD_TYPES))
    return out


LIVE_JSON = Path(__file__).resolve().parents[3] / "site" / "data" / "live.json"


def poll_once(out: Path, league: str = "nfl", watch: Sequence[str] = ()) -> dict:
    """One tick: the scoreboard, optionally some win probabilities, written as an artifact.

    Separated from the loop so it can be tested -- and because the loop is the part with
    nothing to test. It writes the same document `hub.publish.live` writes, through the same
    envelope in `hub.jsonio`, which is the whole point: this used to emit `{ts, games, detail}`
    while the page read `rows` and `generated_at`, so a running poller would have blanked every
    score column. The dashboard worked because nothing was polling.

    `watch` empty is the ordinary tick. The tier-2 fan-out costs one request per game and the
    caller decides how often to spend that, which is ADR-0005.
    """
    state = live_state(league)                          # tier 1: one request
    detail: dict[str, dict] = {}
    for gid in watch:                                   # tier 2: bounded fan-out
        try:
            wp = (summary(gid, league).get("winprobability") or [{}])[-1]
            detail[gid] = {"home_win_prob": wp.get("homeWinPercentage")}
        except Exception:
            continue
    payload = jsonio.artifact("live", "espn_scoreboard", state, league=league, detail=detail)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jsonio.dumps(payload))
    return payload


def poll(interval: int = 45, league: str = "nfl", out: Path | None = None,
         games_of_interest: list[str] | None = None, summary_every: int = 4):
    """Tiered poller. Run locally, not in Actions -- cron fires late under load.

    The concurrency problem is smaller than it looks, because the fan-out is optional.

      Tier 1, every tick: ONE scoreboard request per league. It already carries score,
      status, possession, and down/distance for every game -- 13 NFL games or 60+ CFB
      games in a single response. This is all the dashboard needs.

      Tier 2, every Nth tick: the summary endpoint, which costs one request PER GAME but
      adds win probability and box score. Only fetched for games you actually care about
      (your fantasy players' teams, your survivor pick), capped hard.

    Worst case is roughly 1 + 12 requests per 3 minutes, so a plain sync loop is fine and
    no async is warranted. If that ever changes, wrap the tier-2 fan-out in a
    ThreadPoolExecutor(max_workers=8) -- but do not, because these endpoints are
    undocumented and ESPN asks that request volume stay low.
    """
    out = out or LIVE_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    watch = list(games_of_interest or [])[:12]
    tick = 0
    while True:
        try:
            got = poll_once(out=out, league=league,
                            watch=watch if tick % summary_every == 0 else [])
            live = sum(g["state"] == "in" for g in got["rows"])
            detail = got["detail"]
            print(f"[{time.strftime('%H:%M:%S')}] {got['n']} games, {live} live"
                  f"{f', {len(detail)} detailed' if detail else ''}", flush=True)
        except Exception as e:
            print(f"poll error (serving stale): {e!r}", flush=True)
        tick += 1
        time.sleep(interval)


def _parse_projection(player: dict, season: int) -> float | None:
    """ESPN's own projected points per game for `season`.

    statSourceId 1 is the projection (0 is actuals) and statSplitTypeId 0 is the season
    split. Both matter: the same payload also carries a projection for the *current week*
    and actuals for last season, and picking the wrong one silently substitutes a number
    that is off by an order of magnitude.
    """
    for st in player.get("stats") or []:
        if (st.get("statSourceId") == 1 and st.get("statSplitTypeId") == 0
                and st.get("seasonId") == season):
            avg = st.get("appliedAverage")
            if isinstance(avg, (int, float)) and avg > 0:
                return float(avg)
    return None


def _parse_market(payload: dict, season: int) -> pl.DataFrame:
    """Pull (player, adp, proj_ppg, injury_status) out of a kona_player_info response.

    Split out from the fetch so the parsing is testable without a network call, and
    because this is where every silent-failure mode lives.

    **ESPN writes 0.0 for a player with no draft history.** That is a sentinel for
    undrafted, not the first overall pick, and letting it through would invert the top of
    the board. It becomes a null `adp` rather than dropping the row, because the row still
    carries a projection and an injury status that the board wants -- an undrafted player
    is unpriced, not unknown.

    The empty frame is explicitly typed. The board's substance guard checks dtype, and
    an untyped empty frame comes back as Null -- which is exactly the failure this
    whole path is meant to stop.

    There used to be a second parser here, `_parse_adp`. It was the one with the seven
    tests, and it had no production callers: `player_market` reimplemented the same parse
    inline, so the tested code was dead and the live code was untested.
    """
    rows = []
    for entry in payload.get("players") or []:
        p = entry.get("player") or {}
        name = p.get("fullName")
        if not name:
            continue
        adp = (p.get("ownership") or {}).get("averageDraftPosition")
        rows.append({
            "player": name,
            "adp": float(adp) if isinstance(adp, (int, float))
                   and not isinstance(adp, bool) and adp > 0 else None,
            "proj_ppg": _parse_projection(p, season),
            # Today's designation, which is a different quantity from last season's
            # durability and is carried for judgment rather than priced. See
            # hub.draft.durability.
            "injury_status": p.get("injuryStatus"),
        })
    return pl.DataFrame(rows, schema={"player": pl.Utf8, "adp": pl.Float64,
                                      "proj_ppg": pl.Float64,
                                      "injury_status": pl.Utf8})


def player_market(limit: int = 500, season: int = SEASON_AHEAD) -> pl.DataFrame:
    """ADP and ESPN's projection together, from one request.

    These are the two things the rest of the room can see, so they belong in one place:
    ADP is where the room takes a player, the projection is why.

    espn_api cannot give you the ADP. It reads `ownership.percentOwned` and discards the
    rest of the block, where `averageDraftPosition` lives, so we go to the raw view.
    One bulk request covers the whole draftable pool; do not loop over players.
    """
    lg = league_settings(year=season).league
    filters = {"players": {"limit": limit,
                           "sortDraftRanks": {"sortPriority": 100, "sortAsc": True,
                                              "value": "PPR"}}}
    payload = lg.espn_request.league_get(params={"view": "kona_player_info"},
                                         headers={"x-fantasy-filter": json.dumps(filters)})
    return _parse_market(payload, season)


@dataclass(frozen=True)
class LeagueView:
    """A league handle and the settings read off it, together.

    One object rather than a tuple, because every call site discarded half of it -- three
    spelled it `lg, _` and one `_, slots` -- so the tuple was never carrying two things
    anybody wanted at once.
    """
    league: Any
    roster_slots: dict[str, int]


def league_settings(year: int = SEASON_AHEAD, league_id: int | None = None) -> LeagueView:
    """Read roster slots straight from your league instead of hardcoding them.

    `year` is a parameter because it used to be the literal 2026 while `player_market` took a
    `season` argument and filtered stat blocks by it. The two were independent: asking for
    2027 fetched the 2026 league, matched no stat block, and returned a full frame with
    `proj_ppg` all null -- and `proj_blend` coalesces to `xfp_per_game`, so the board built,
    THE PICK worked, and ESPN's projection had silently stopped contributing.
    """
    from dotenv import load_dotenv
    from espn_api.football import League
    load_dotenv()
    lg = League(
        league_id=int(league_id if league_id is not None else os.environ["ESPN_LEAGUE_ID"]),
        year=year,
        espn_s2=os.environ.get("ESPN_S2") or None,
        swid=os.environ.get("ESPN_SWID") or None,
    )
    return LeagueView(lg, getattr(lg.settings, "roster_slots", {}) or {})


# --- your team, as rows -----------------------------------------------------
#
# This module's job is knowing ESPN. It used to stop at handing back a raw `espn_api` League
# and let `hub.season.roster` read the payload itself -- duck-typing seven player attributes,
# matching the SWID, and deriving availability from two projections. The cost was that ESPN's
# season projection had two readers: `_parse_projection` above, picking through raw stat
# blocks for the Board, and `projected_avg_points` over in `season/`. Same number, two routes,
# in packages that shared no code -- which is why the Board could hold 5.10 for a player ESPN
# had already repriced to 8.01 and nothing was positioned to notice they were the same field.


def my_team(league: Any, swid: str | None = None) -> Any:
    """The team owned by the SWID already in the environment for the ESPN cookie.

    Identity is inferred rather than configured, so there is no `ESPN_TEAM_ID` to keep in sync
    with a league you might leave. Raises rather than guessing: a wrong team here would be
    invisible downstream, because every number would compute cleanly against a roster that is
    not yours.
    """
    me = (swid if swid is not None else os.environ.get("ESPN_SWID") or "").strip("{}").upper()
    if not me:
        raise LookupError("ESPN_SWID is not set, so no team can be identified as yours")
    for team in league.teams:
        owners = team.owners or []
        ids = {str(o.get("id", o) if isinstance(o, dict) else o).strip("{}").upper()
               for o in owners}
        if me in ids:
            return team
    raise LookupError(
        f"no team in this league is owned by the configured SWID "
        f"(checked {len(league.teams)} teams)")


# The seven attributes anything downstream depends on. Pinned here, in the module that owns
# ESPN's vocabulary, so a rename in `espn_api` breaks one place with a test on it rather than
# silently filling a column with empty strings somewhere else.
ROSTER_SCHEMA: dict[str, Any] = {
    "player": pl.Utf8, "key": pl.Utf8, "pos": pl.Utf8, "espn_id": pl.Int64,
    "nfl_team": pl.Utf8, "slot": pl.Utf8, "injury_status": pl.Utf8,
    "espn_avg": pl.Float64, "espn_total": pl.Float64,
}


def roster_rows(team: Any) -> pl.DataFrame:
    """One row per rostered player, as data rather than as a vendor object.

    `espn_avg` and `espn_total` are ESPN's own two projections. Their *ratio* is the only
    season-level availability signal in the payload, and it is the one that matters: ESPN
    reported Josh Jacobs `DAY_TO_DAY` through a six-game ban, and only
    `projected_total_points / projected_avg_points` -- 11.0 where the rest of the roster was
    17.0 -- gave it away. `hub.season.roster.availability` reads it off these two columns.
    """
    from hub.names import player_key
    rows = []
    for p in team.roster:
        name = str(getattr(p, "name", "") or "")
        rows.append({
            "player": name,
            "key": player_key(name),
            "pos": str(getattr(p, "position", "") or ""),
            "espn_id": int(getattr(p, "playerId", 0) or 0),
            "nfl_team": str(getattr(p, "proTeam", "") or ""),
            "slot": str(getattr(p, "lineupSlot", "") or ""),
            "injury_status": str(getattr(p, "injuryStatus", "") or ""),
            "espn_avg": float(getattr(p, "projected_avg_points", 0.0) or 0.0),
            "espn_total": float(getattr(p, "projected_total_points", 0.0) or 0.0),
        })
    return pl.DataFrame(rows, schema=ROSTER_SCHEMA)


def my_roster(year: int = SEASON_AHEAD) -> pl.DataFrame:  # pragma: no cover - network
    """Your roster, from the league you are logged into."""
    return roster_rows(my_team(league_settings(year).league))


# --- league transaction history -------------------------------------------
#
# `docs/championship-leverage.md` gates its trade evaluator on measuring whether this
# league trades at all, and says not to guess. The obvious route does not work:
# `espn_api`'s `recent_activity` returns HTTP 400 for every past season here, because it
# reads `communication/` and ESPN 404s that endpoint for anything but the current season.
# The per-team `transactionCounter` on the `mTeam` view survives in `leagueHistory`, so
# that is what this reads.

LM_API = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"


class NoLeagueHistory(Exception):
    """ESPN returned no teams for a season."""


# kona_player_info returns nothing without a filter naming how many players to send and in
# what order. `filterIds` looks like the precise way to ask for the drafted players only and
# returns HTTP 400 here, so take the top of the draft-rank ordering in one request and match
# locally -- which is also the rule against looping over players.
PLAYER_FILTER = {"players": {"limit": 800,
                            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True,
                                               "value": "PPR"}}}


def league_history(season: int, view: str = "mTeam") -> dict:
    """Raw `leagueHistory` payload for one past season. Needs the private-league cookies."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    r = requests.get(
        f"{LM_API}/leagueHistory/{os.environ['ESPN_LEAGUE_ID']}",
        params={"view": view, "seasonId": season},
        headers=({"x-fantasy-filter": json.dumps(PLAYER_FILTER)}
                 if view == "kona_player_info" else None),
        cookies={"espn_s2": os.environ.get("ESPN_S2", ""),
                 "SWID": os.environ.get("ESPN_SWID", "")},
        timeout=60)
    r.raise_for_status()
    body = r.json()
    return body[0] if isinstance(body, list) and body else body


_league_history = league_history


def transaction_counts(seasons, fetch=None) -> pl.DataFrame:
    """Per-team trade, acquisition and drop counts for each season."""
    fetch = fetch or _league_history
    rows = []
    for season in seasons:
        teams = (fetch(season) or {}).get("teams") or []
        if not teams:
            raise NoLeagueHistory(
                f"no teams for season {season}. Counting that as zero activity would "
                "argue from a network failure.")
        for t in teams:
            c = t.get("transactionCounter") or {}
            rows.append({"season": int(season),
                         "team": t.get("name") or str(t.get("id")),
                         "trades": int(c.get("trades") or 0),
                         "acquisitions": int(c.get("acquisitions") or 0),
                         "drops": int(c.get("drops") or 0)})
    return pl.DataFrame(rows)


def trade_summary(counts: pl.DataFrame) -> dict:
    """League totals per season.

    A trade increments the counter on both sides, so the league total is half the sum --
    but acquisitions and drops have one side each and must not be halved. An odd trade sum
    means the two-sided assumption broke (a three-way deal, or a team missing from league
    history), and that is surfaced rather than rounded away.
    """
    per = counts.group_by("season").agg(
        pl.col("trades").sum().alias("t"), pl.col("acquisitions").sum().alias("a"),
        pl.col("drops").sum().alias("d"),
        (pl.col("trades") > 0).sum().alias("n")).sort("season")
    return {
        "trades": {int(r["season"]): int(r["t"]) // 2 for r in per.iter_rows(named=True)},
        "acquisitions": {int(r["season"]): int(r["a"]) for r in per.iter_rows(named=True)},
        "drops": {int(r["season"]): int(r["d"]) for r in per.iter_rows(named=True)},
        "teams_trading": {int(r["season"]): int(r["n"]) for r in per.iter_rows(named=True)},
        "uneven_seasons": [int(r["season"]) for r in per.iter_rows(named=True)
                           if int(r["t"]) % 2],
    }


# ESPN stat ids for the items `hub.models.components.SCORING` models. ESPN omits any item
# worth zero, so an absent id means "not scored" rather than "disagrees".
SCORING_STAT_IDS = {
    3: "passing_yards", 4: "passing_tds", 20: "interceptions",
    24: "rushing_yards", 25: "rushing_tds",
    42: "receiving_yards", 43: "receiving_tds", 53: "receptions",
    72: "fumbles_lost",
}


def scoring_settings() -> dict[str, float]:
    """The league's own scoring weights, read from mSettings.

    The aggregation weights are a league setting. Reading them is what turns
    `components.SCORING` from an assumption into something checkable.
    """
    lg = league_settings().league
    raw = lg.espn_request.league_get(params={"view": "mSettings"})
    items = (((raw.get("settings") or {}).get("scoringSettings") or {})
             .get("scoringItems") or [])
    out: dict[str, float] = {}
    for it in items:
        name = SCORING_STAT_IDS.get(it.get("statId"))
        pts = it.get("points")
        if name and isinstance(pts, (int, float)) and pts:
            out[name] = float(pts)
    return out
