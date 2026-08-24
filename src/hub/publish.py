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
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import polars as pl

from hub import store

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write(out: Path, name: str, payload: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
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
    name = f"preds_wk{week:02d}"
    try:
        df = store.sql(
            "SELECT * FROM preds WHERE season = ? AND week = ?",
            params=[season, f"{week:02d}"], base=base)
    except Exception:  # noqa: BLE001
        return None
    if df.is_empty():
        return None

    payload = _artifact(name, "preds", df.to_dicts(), season=season, week=week)
    _write(out, name, payload)
    return payload


# --- calibration ----------------------------------------------------------

def log_loss(probs: Sequence[float] | np.ndarray,
             outcomes: Sequence[int] | np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log likelihood.

    Clipped at eps because a model that said 1.0 and was wrong would otherwise put an
    infinity on the page. Clipping bounds the penalty at ~34 per game, which is still
    ruinous and still renders.
    """
    # len() rather than truthiness: `if not probs` raises on a numpy array, which went
    # unnoticed while every caller passed lists. Vectorised because hub.models.eval
    # bootstraps this thousands of times per comparison.
    q = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    if q.size == 0:
        return float("nan")
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def brier(probs: Sequence[float] | np.ndarray,
          outcomes: Sequence[int] | np.ndarray) -> float:
    q = np.asarray(probs, dtype=float)
    if q.size == 0:
        return float("nan")
    return float(np.mean((q - np.asarray(outcomes, dtype=float)) ** 2))


def reliability(df: pl.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram: predicted versus actual, with the count in each bin.

    Counts are not decoration. `docs/track-record.md` asks for them because a bin holding
    four games says nothing, and a diagram that hides its bin sizes invites exactly the
    over-reading the page exists to prevent.
    """
    if df.is_empty():
        return []
    out = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = df.filter((pl.col("home_win_prob") >= lo)
                        & (pl.col("home_win_prob") < (hi if i < n_bins - 1 else 1.01)))
        n = sel.height
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": n,
            "predicted": float(cast(float, sel["home_win_prob"].mean())) if n else None,
            "actual": float(cast(float, sel["home_won"].mean())) if n else None,
        })
    return out


def _scored(base: Path | None) -> pl.DataFrame:
    """Predictions joined to results. Empty frame when either side is missing."""
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "home_win_prob": pl.Float64,
                                 "home_won": pl.Int64, "predicted_at": pl.Utf8})
    try:
        preds = store.sql("SELECT * FROM preds", base=base)
    except Exception:  # noqa: BLE001
        return empty
    if preds.is_empty():
        return empty

    try:
        import nflreadpy as nfl
        sched = (nfl.load_schedules()
                 .filter(pl.col("result").is_not_null())
                 .select(pl.col("game_id"),
                         (pl.col("result") > 0).cast(pl.Int64).alias("home_won")))
    except Exception:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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

    record(f"preds_wk{week:02d}", predictions(season, week, base=base, out=out),
           f"no predictions in the store for {season} week {week}")
    record("track_record", track_record(base=base, out=out), "no scored predictions")
    record("live", live(out=out), "ESPN scoreboard unavailable")

    board = (out / "draft_board.json")
    arts.append({"name": "draft_board", "present": board.exists(),
                 "stale": not board.exists(),
                 "reason": None if board.exists() else "run `make draft`",
                 "generated_at": None})
    # Survivor is Phase 5.1 and deliberately absent. The page renders the panel as
    # unavailable rather than pretending it is coming.
    arts.append({"name": "survivor", "present": False, "stale": True,
                 "reason": "not built yet (foundation-plan 5.1)", "generated_at": None})

    man = {"generated_at": _now(), "season": season, "week": week, "artifacts": arts}
    _write(out, "manifest", man)
    return man


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.publish", description="Write site/data/*.json from the processed store.")
    ap.add_argument("--all", action="store_true", help="every artifact plus the manifest")
    ap.add_argument("--predictions", action="store_true", help="one week's predictions")
    ap.add_argument("--track-record", action="store_true", help="calibration page data")
    ap.add_argument("--live", action="store_true", help="live scores overlay only")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=1)
    a = ap.parse_args(argv)

    if a.predictions:
        got = predictions(a.season, a.week)
        print(f"  preds_wk{a.week:02d}: "
              + (f"{got['n']} games" if got else "nothing in the store; left as-is"))
        return 0
    if a.live:
        got = live()
        print(f"  live: " + (f"{got['n']} games" if got else "unavailable; last-good kept"))
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
