"""ESPN fetch layer. Two surfaces, both unofficial:

  1. site.api.espn.com  -> public scoreboard / summary / win probability (no auth)
  2. fantasy ESPN v3    -> your league, via espn_s2 + swid cookies

Both break without notice. Every call here retries across hostname and User-Agent
variants before giving up, and falls back to last-good cached state.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import requests
import polars as pl

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
            except Exception as e:  # noqa: BLE001
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


def live_state(league: str = "nfl") -> list[dict]:
    """Flattened in-progress game state for the dashboard overlay."""
    out = []
    for ev in scoreboard(league).get("events", []):
        c = ev["competitions"][0]
        st = c["status"]["type"]
        home, away = c["competitors"][0], c["competitors"][1]
        out.append({
            "id": ev["id"],
            "state": st["state"],           # pre | in | post
            "detail": st.get("shortDetail"),
            "home": home["team"]["abbreviation"], "home_score": home.get("score"),
            "away": away["team"]["abbreviation"], "away_score": away.get("score"),
            "possession": c.get("situation", {}).get("possession"),
            "down_distance": c.get("situation", {}).get("downDistanceText"),
        })
    return out


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
    out = out or Path(__file__).resolve().parents[3] / "site" / "data" / "live.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    watch = list(games_of_interest or [])[:12]
    tick = 0
    while True:
        try:
            state = live_state(league)                      # tier 1: one request
            detail = {}
            if watch and tick % summary_every == 0:         # tier 2: bounded fan-out
                for gid in watch:
                    try:
                        s = summary(gid, league)
                        wp = (s.get("winprobability") or [{}])[-1]
                        detail[gid] = {"home_win_prob": wp.get("homeWinPercentage")}
                    except Exception:  # noqa: BLE001, S112
                        continue
            out.write_text(json.dumps({"ts": time.time(), "games": state, "detail": detail}))
            live = sum(g["state"] == "in" for g in state)
            print(f"[{time.strftime('%H:%M:%S')}] {len(state)} games, {live} live"
                  f"{f', {len(detail)} detailed' if detail else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"poll error (serving stale): {e!r}", flush=True)
        tick += 1
        time.sleep(interval)


def _parse_adp(payload: dict) -> pl.DataFrame:
    """Pull (player, adp) out of a kona_player_info response.

    Split out from the fetch so the parsing is testable without a network call, and
    because this is where every silent-failure mode lives. ESPN writes 0.0 for a
    player with no draft history; that is a sentinel for undrafted, not the first
    overall pick, so it is dropped rather than allowed to invert the top of the board.

    The empty frame is explicitly typed. The board's substance guard checks dtype, and
    an untyped empty frame comes back as Null -- which is exactly the failure this
    whole path is meant to stop.
    """
    rows = []
    for entry in payload.get("players") or []:
        p = entry.get("player") or {}
        name = p.get("fullName")
        adp = (p.get("ownership") or {}).get("averageDraftPosition")
        if name and isinstance(adp, (int, float)) and not isinstance(adp, bool) and adp > 0:
            rows.append({"player": name, "adp": float(adp)})
    return pl.DataFrame(rows, schema={"player": pl.Utf8, "adp": pl.Float64})


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


def player_market(limit: int = 500, season: int = 2026) -> pl.DataFrame:
    """ADP and ESPN's projection together, from one request.

    These are the two things the rest of the room can see, so they belong in one place:
    ADP is where the room takes a player, the projection is why.
    """
    lg, _ = league_settings()
    filters = {"players": {"limit": limit,
                           "sortDraftRanks": {"sortPriority": 100, "sortAsc": True,
                                              "value": "PPR"}}}
    payload = lg.espn_request.league_get(params={"view": "kona_player_info"},
                                         headers={"x-fantasy-filter": json.dumps(filters)})
    rows = []
    for entry in payload.get("players") or []:
        pl_ = entry.get("player") or {}
        name = pl_.get("fullName")
        if not name:
            continue
        adp = (pl_.get("ownership") or {}).get("averageDraftPosition")
        rows.append({
            "player": name,
            "adp": float(adp) if isinstance(adp, (int, float))
                   and not isinstance(adp, bool) and adp > 0 else None,
            "proj_ppg": _parse_projection(pl_, season),
        })
    return pl.DataFrame(rows, schema={"player": pl.Utf8, "adp": pl.Float64,
                                      "proj_ppg": pl.Float64})


def player_adp(limit: int = 500) -> pl.DataFrame:
    """ESPN's own average draft position -- what your room is actually drafting off.

    espn_api cannot give you this. It reads `ownership.percentOwned` and discards the
    rest of the block, where `averageDraftPosition` lives, so we go to the raw view.
    One bulk request covers the whole draftable pool; do not loop over players.
    """
    lg, _ = league_settings()
    filters = {"players": {"limit": limit,
                           "sortDraftRanks": {"sortPriority": 100, "sortAsc": True,
                                              "value": "PPR"}}}
    payload = lg.espn_request.league_get(params={"view": "kona_player_info"},
                                         headers={"x-fantasy-filter": json.dumps(filters)})
    return _parse_adp(payload)


def league_settings():
    """Read roster slots straight from your league instead of hardcoding them."""
    from dotenv import load_dotenv
    from espn_api.football import League
    load_dotenv()
    lg = League(
        league_id=int(os.environ["ESPN_LEAGUE_ID"]),
        year=2026,
        espn_s2=os.environ.get("ESPN_S2") or None,
        swid=os.environ.get("ESPN_SWID") or None,
    )
    return lg, lg.settings.roster_slots if hasattr(lg.settings, "roster_slots") else {}


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


def _league_history(season: int, view: str = "mTeam") -> dict:
    """Raw `leagueHistory` payload for one past season. Needs the private-league cookies."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    r = requests.get(
        f"{LM_API}/leagueHistory/{os.environ['ESPN_LEAGUE_ID']}",
        params={"view": view, "seasonId": season},
        cookies={"espn_s2": os.environ.get("ESPN_S2", ""),
                 "SWID": os.environ.get("ESPN_SWID", "")},
        timeout=30)
    r.raise_for_status()
    body = r.json()
    return body[0] if isinstance(body, list) and body else body


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
