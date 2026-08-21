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
