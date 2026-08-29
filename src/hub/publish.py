"""Write the site's JSON from the processed store.

Two rules from the docs shape this more than anything technical.

`docs/track-record.md` rule 1: a prediction counts only if it was committed before kickoff.
The value of a public record is entirely in the timestamp, so the track-record artifact
records how many predictions were *pre-registered* and refuses to present a backfilled
score as though it were one. A page that cannot tell those apart proves nothing.

`CLAUDE.md`: if a fetch fails, serve last-good state rather than erroring. So a source that
is missing never blanks an artifact. The previous file stays, the manifest marks it stale
and says why, and the page renders yesterday's numbers with a warning -- which on a bad
Sunday is worth far more than an empty screen.

    uv run python -m hub.publish --all --week 1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import jsonio, store
from hub.config import SEASON_AHEAD
from hub.models.scoring_rules import brier, log_loss, reliability

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"
NFL_WEEKS = 18


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write(out: Path, name: str, payload: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(jsonio.dumps(payload, indent=2))
    return p


def _artifact(name: str, source: str, rows: list[dict[str, Any]],
              **extra: Any) -> dict[str, Any]:
    return {"name": name, "source": source, "generated_at": _now(),
            "n": len(rows), "rows": rows, **extra}


# --- weekly predictions ---------------------------------------------------

def predictions(season: int, week: int, base: Path | None = None,
                out: Path | None = None) -> dict[str, Any] | None:
    """One artifact per week, so the git history pins each week to a commit.

    Returns None and touches nothing when the store has no predictions for that week --
    the caller records it as stale rather than publishing an empty page.
    """
    out = out or SITE
    name = f"preds_wk{store.week_key(week)}"
    # Asked, not caught. `store.tables` exists precisely so a caller can tell "this clone has
    # no predictions yet" from "the query is broken", and a bare `except Exception` here made
    # a schema break, a DuckDB lock and a typo in the SQL all arrive as
    # `"stale": true, "reason": "no predictions"`.
    if "preds" not in store.tables(base):
        return None
    df = store.sql("SELECT * FROM preds WHERE season = ? AND week = ?",
                   params=[season, store.week_key(week)], base=base)
    if df.is_empty():
        return None

    payload = _artifact(name, "preds", df.to_dicts(), season=season, week=week)
    _write(out, name, payload)
    return payload


# --- calibration ----------------------------------------------------------

# The scoring rules moved to `hub.models.scoring_rules`; `hub.models.eval` was importing
# them from here, which meant model evaluation could not be read without the site writer.
# Re-exported because `publish` still uses all three and the site's own tests name them here.
def _scored(base: Path | None) -> pl.DataFrame:
    """Predictions joined to results. Empty frame when either side is missing."""
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "home_win_prob": pl.Float64,
                                 "home_won": pl.Int64, "predicted_at": pl.Utf8})
    if "preds" not in store.tables(base):
        return empty
    preds = store.sql("SELECT * FROM preds", base=base)
    if preds.is_empty():
        return empty

    try:
        import nflreadpy as nfl
        sched = (nfl.load_schedules()
                 .filter(pl.col("result").is_not_null())
                 .select(pl.col("game_id"),
                         (pl.col("result") > 0).cast(pl.Int64).alias("home_won")))
    except Exception as e:
        # This one stays broad -- it is a network call on a schedule that runs unattended --
        # but it says which failure it was rather than presenting every cause as "no results".
        print(f"  schedules unavailable ({type(e).__name__}); track record served without "
              f"results", flush=True)
        return empty
    return preds.join(sched, on="game_id", how="inner")


def track_record(base: Path | None = None, out: Path | None = None,
                 n_bins: int = 10) -> dict[str, Any]:
    """Calibration, and an honest count of what was actually pre-registered.

    `n_preregistered` is deliberately separate from `n_scored`. Scoring a prediction after
    the game is a backtest; the record only means something for predictions whose commit
    predates kickoff. Until the season starts both are zero, and the page says so rather
    than filling the space with a backfill dressed up as history.
    """
    out = out or SITE
    df = _scored(base)

    payload: dict[str, Any] = {
        "name": "track_record", "source": "preds+results", "generated_at": _now(),
        "n_scored": df.height,
        # Nothing is pre-registered until a prediction is committed before kickoff, which
        # the Sunday Actions job does. Counting it here would be marking my own homework.
        "n_preregistered": 0,
        "is_backtest": df.height > 0,
        "note": ("No pre-registered predictions yet. A prediction counts only once its "
                 "commit predates kickoff -- see docs/track-record.md."),
        "bins": [], "log_loss": None, "brier": None,
    }
    if df.height:
        probs = df["home_win_prob"].to_list()
        won = df["home_won"].to_list()
        payload["bins"] = reliability(df, n_bins)
        payload["log_loss"] = log_loss(probs, won)
        payload["brier"] = brier(probs, won)
    _write(out, "track_record", payload)
    return payload


# --- the live overlay -----------------------------------------------------

def live(out: Path | None = None, league: str = "nfl") -> dict[str, Any] | None:
    """Current scores, written separately from the model's numbers on purpose.

    This is the only artifact that changes during a Sunday, and keeping it in its own file
    is what lets the page hold the two apart. Predictions are frozen at lock and must never
    be regenerated mid-slate: a win probability that drifts while games are in progress is
    indistinguishable from one that was always going to look right, which destroys the
    pre-registration the whole record depends on.

    So: scores move, model numbers do not, and the page says which is which.
    """
    out = out or SITE
    try:
        from hub.fetch.espn import live_state
        rows = live_state(league)
    except Exception as e:
        print(f"  live: ESPN unavailable ({type(e).__name__}); leaving last-good in place")
        return None
    payload = _artifact("live", "espn_scoreboard", rows, league=league)
    _write(out, "live", payload)
    return payload


# --- everything, plus a manifest -----------------------------------------

def publish_all(season: int, week: int, base: Path | None = None,
                out: Path | None = None) -> dict[str, Any]:
    """Write every artifact and a manifest describing what the page can trust."""
    out = out or SITE
    arts: list[dict[str, Any]] = []

    def record(name: str, payload: dict[str, Any] | None, reason: str) -> None:
        path = out / f"{name}.json"
        present = path.exists()
        arts.append({
            "name": name, "present": present,
            "stale": payload is None,
            "reason": None if payload else reason,
            "generated_at": payload["generated_at"] if payload else (
                json.loads(path.read_text()).get("generated_at") if present else None),
        })

    record(f"preds_wk{store.week_key(week)}", predictions(season, week, base=base, out=out),
           f"no predictions in the store for {season} week {week}")
    record("track_record", track_record(base=base, out=out), "no scored predictions")
    record("live", live(out=out), "ESPN scoreboard unavailable")

    board = (out / "draft_board.json")
    arts.append({"name": "draft_board", "present": board.exists(),
                 "stale": not board.exists(),
                 "reason": None if board.exists() else "run `make draft`",
                 "generated_at": None})
    arts.append(survivor(season, out=out))

    man = {"generated_at": _now(), "season": season, "week": week, "artifacts": arts}
    _write(out, "manifest", man)
    return man


def survivor(season: int, out: Path | None = None) -> dict[str, Any]:
    """The survivor plan, as its own artifact.

    Wrapped rather than inlined because it reaches the network for a schedule. A failing
    source marks the panel stale and leaves the last good plan in place -- taking the whole
    page down over one panel is the operator-dependence CLAUDE.md warns about.
    """
    from hub.season import survivor as sv
    out = out or SITE
    try:
        grid = sv.grid_from_schedule(season)
        # Against the full season, not against the weeks the grid happens to have -- asking
        # coverage about its own weeks makes `missing` empty by construction, and the panel
        # would never say a week still needs a pick.
        cov = sv.coverage(grid, list(range(1, NFL_WEEKS + 1)))
        plan = sv.solve(grid, weeks=cov["covered"])
    except Exception as e:
        return {"name": "survivor", "present": False, "stale": True,
                "reason": f"{type(e).__name__}: {e}"[:120], "generated_at": None}
    rows = plan.to_dicts()
    art = _artifact("survivor", "hub.season.survivor", rows, season=season,
                    survival=sv.survival(plan), unpriced_weeks=cov["missing"])
    _write(out, "survivor", art)
    return {"name": "survivor", "present": True, "stale": False,
            "reason": None, "generated_at": art["generated_at"]}


def default_week(season: int, base: Path | None = None) -> int:
    """The latest week already predicted, or 1 before the season starts.

    `--week` used to default to 1, so `make slate` with no week set would republish week 1
    every Sunday and look like it had worked. Read from the store rather than the network:
    a weekly refresh that needs a live API to decide what week it is has one more way to
    fail on a Sunday.
    """
    if "preds" not in store.tables(base):
        return 1
    got = store.sql("SELECT max(week) AS w FROM preds WHERE season = ?",
                    params=[season], base=base)
    if got.height and got["w"][0] is not None:
        return int(got["w"][0])
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.publish", description="Write site/data/*.json from the processed store.")
    ap.add_argument("--all", action="store_true", help="every artifact plus the manifest")
    ap.add_argument("--predictions", action="store_true", help="one week's predictions")
    ap.add_argument("--track-record", action="store_true", help="calibration page data")
    ap.add_argument("--live", action="store_true", help="live scores overlay only")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, default=None,
                    help="defaults to the latest week already predicted")
    a = ap.parse_args(argv)
    if a.week is None:
        a.week = default_week(a.season)

    if a.predictions:
        got = predictions(a.season, a.week)
        print(f"  preds_wk{a.week:02d}: "
              + (f"{got['n']} games" if got else "nothing in the store; left as-is"))
        return 0
    if a.live:
        got = live()
        print("  live: " + (f"{got['n']} games" if got else "unavailable; last-good kept"))
        return 0
    if a.track_record:
        got = track_record()
        print(f"  track_record: {got['n_scored']} scored, "
              f"{got['n_preregistered']} pre-registered")
        return 0
    if not a.all:
        ap.print_help()
        return 0

    man = publish_all(a.season, a.week)
    print(f"  published {len(man['artifacts'])} artifacts for {a.season} week {a.week}")
    for art in man["artifacts"]:
        mark = "stale" if art["stale"] else "ok"
        print(f"    {art['name']:<16} {mark:<6} {art['reason'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
