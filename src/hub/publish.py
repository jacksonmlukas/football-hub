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
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub import jsonio, store
from hub.config import SEASON_AHEAD
from hub.models.scoring_rules import brier, log_loss, reliability
from hub.paths import ROSTER_PARQUET
from hub.season.roster import lock

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


# --- the roster, and the lineup it implies --------------------------------

def roster(out: Path | None = None, path: Path | None = None) -> dict[str, Any] | None:
    """Serialise the roster and the lock decision. Does not *make* the decision.

    Reads `data/processed/roster.parquet` rather than ESPN, so the panel is publishable
    without a network round trip and shows last-good when a sync fails -- the same contract
    every other artifact here keeps.

    `season.roster.lock` computes set-versus-best, including which players are withheld as
    unavailable. That arithmetic lived here for one evening, which meant the only way to ask
    the Sunday question was to publish a website; it is a decision, not a rendering.
    """
    src = path or ROSTER_PARQUET
    if not src.exists():
        return None
    df = pl.read_parquet(src)
    lk = lock(df)
    # The best lineup, reconstructed from the moves: the set starters, less those to sit,
    # plus those to start. Parenthesised because `-` binds tighter than `|` and the reader
    # should not have to know that.
    was = set(df.filter(pl.col("projected") & pl.col("starting"))["player"])
    start = (was - set(lk.bench)) | set(lk.start)

    rows = []
    for r in df.iter_rows(named=True):
        rows.append({
            "player": r["player"], "pos": r["pos"], "nfl_team": r["nfl_team"],
            "mu": r["mu"], "sd": r["sd"], "projected": r["projected"],
            "starting": r["starting"], "best_start": r["player"] in start,
            "injury_status": r["injury_status"],
            "available": r.get("available", True),
            # Two neighbouring facts, and the panel needs both: `available` is whether we
            # expect him to play, `can_start` whether the league will let him.
            "can_start": r.get("can_start", True),
            "missing_games": r.get("missing_games", 0),
        })
    payload = _artifact("roster", "roster.parquet", rows,
                        set_total=lk.set_total, optimal_total=lk.best_total, gain=lk.gain,
                        withheld=lk.withheld, start=lk.start, sit=lk.bench)
    _write(out or SITE, "roster", payload)
    return payload


# --- everything, plus a manifest -----------------------------------------

def _generated_at(path: Path) -> str | None:
    """Last-good's timestamp, or None if the file has no dict to read it from.

    `draft_board.json` is a list of rows, so a bare `.get` on it raises. That branch was
    unreachable while draft_board bypassed the contract below; it is reachable now.
    """
    try:
        got = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return got.get("generated_at") if isinstance(got, dict) else None


class Artifact(NamedTuple):
    """One published file, and the freshness contract every one of them keeps.

    `CLAUDE.md`'s degradation rule -- *a panel whose data is missing says so and keeps
    rendering* -- used to be written three ways inside `publish_all`: a `record` closure for
    four artifacts, a hand-rolled dict for `draft_board`, and `survivor` returning its own
    manifest entry. The two that bypassed `record` were also the two that always reported
    `generated_at: null`, so the page could not age them.

    A producer returns its payload, or **None** meaning "nothing new, keep last-good". It
    never builds a manifest entry itself; that is this module's single job.
    """
    name: str
    produce: Callable[[], dict[str, Any] | None]
    reason: str

    def record(self, out: Path) -> dict[str, Any]:
        payload = self.produce()
        path = out / f"{self.name}.json"
        present = path.exists()
        return {
            "name": self.name, "present": present, "stale": payload is None,
            "reason": None if payload else self.reason,
            "generated_at": (payload.get("generated_at") if payload
                             else (_generated_at(path) if present else None)),
        }


def _board(out: Path) -> dict[str, Any] | None:
    """`draft_board.json` has no producer here -- `hub.draft.board` writes it.

    So its "producer" only reports whether it is there. It carries no `generated_at` because
    the file is a bare list of rows; the board's own age comes from the parquet's mtime, via
    `board.board_age_hours`.
    """
    return {"generated_at": None} if (out / "draft_board.json").exists() else None


def artifacts(season: int, week: int, base: Path | None = None,
              out: Path | None = None) -> list[Artifact]:
    """Everything the page reads, declared once. Adding a panel is one entry."""
    out = out or SITE
    return [
        Artifact(f"preds_wk{store.week_key(week)}",
                 lambda: predictions(season, week, base=base, out=out),
                 f"no predictions in the store for {season} week {week}"),
        Artifact("track_record", lambda: track_record(base=base, out=out),
                 "no scored predictions"),
        Artifact("live", lambda: live(out=out), "ESPN scoreboard unavailable"),
        Artifact("roster", lambda: roster(out=out),
                 "no roster yet -- run `python -m hub.season.roster --write`"),
        Artifact("draft_board", lambda: _board(out), "run `make draft`"),
        Artifact("survivor", lambda: survivor(season, out=out), "schedule unavailable"),
    ]


def publish_all(season: int, week: int, base: Path | None = None,
                out: Path | None = None) -> dict[str, Any]:
    """Write every artifact and a manifest describing what the page can trust."""
    out = out or SITE
    arts = [a.record(out) for a in artifacts(season, week, base=base, out=out)]
    man = {"generated_at": _now(), "season": season, "week": week, "artifacts": arts}
    _write(out, "manifest", man)
    return man


def survivor(season: int, out: Path | None = None) -> dict[str, Any] | None:
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
        # Reported the way `live` reports its own failure: printed for the operator, None for
        # the caller. It used to hand back a manifest entry of its own -- the only producer
        # that did -- which is why it could never be recorded like the rest.
        print(f"  survivor: schedule unavailable ({type(e).__name__}: {e})"[:160])
        return None
    art = _artifact("survivor", "hub.season.survivor", plan.to_dicts(), season=season,
                    survival=sv.survival(plan), unpriced_weeks=cov["missing"],
                    # Which weeks exist in this plan only because the store was read. They
                    # are the ones a reader should not expect to find on nflverse.
                    snapshot_only_weeks=sv.snapshot_only_weeks(grid, cov["covered"]))
    _write(out, "survivor", art)
    return art


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
