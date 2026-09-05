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
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub import jsonio, schedule, store
from hub.config import SEASON_AHEAD
from hub.models.scoring_rules import brier, log_loss, reliability
from hub.paths import ROSTER_PARQUET
from hub.season.roster import lock

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"
NFL_WEEKS = 18


def _write(out: Path, name: str, payload: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(jsonio.dumps(payload, indent=2))
    return p


def _published_n(out: Path, name: str) -> int:
    """How many rows the artifact already on disk carries. Unreadable or absent counts 0."""
    try:
        got = json.loads((out / f"{name}.json").read_text())
    except (OSError, ValueError):
        return 0
    return int(got.get("n", 0)) if isinstance(got, dict) else 0


def _publish(out: Path, name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Write the artifact, unless doing so would replace something with nothing.

    **The one place this rule lives.** `CLAUDE.md`'s degradation rule is implemented as
    *a producer returning None means keep last-good*, and an empty-but-valid payload walks
    straight past it: `Artifact.record` reads any payload as success. That has bitten three
    times -- `track_record` published an empty record over sixteen scored predictions and
    turned a log-loss of 0.556 into null; the carry-forward re-stamped last-good as fresh;
    `roster` guarded `not src.exists()` and nothing else, so an ESPN sync returning an empty
    league would have published a roster of nobody. Each was fixed where it was found, which
    is why the next one was a surprise.

    Not "never write an empty artifact". A first run on a day with no games has an honest
    empty answer and should say it; what must not happen is that answer *replacing* a
    fuller one. So the comparison is against what is already published, and a producer that
    goes quiet keeps last-good and is marked stale.
    """
    if not payload.get("n") and (had := _published_n(out, name)):
        print(f"  {name}: 0 rows against {had} already published; keeping last-good",
              flush=True)
        return None
    _write(out, name, payload)
    return payload


def preds_name(season: int, week: int) -> str:
    """The weekly artifact's filename stem. Season *and* week, because a week is not a slate.

    It was `preds_wk{week}` alone, and `site/data/preds_wk18.json` holds 2025 -- so
    publishing 2026 week 18 would overwrite it. That file is not merely a page: the track
    record is scored by reading every published artifact back (`_published`), so the sixteen
    2025 games it holds are the whole of the site's calibration numbers, and they would have
    gone without anything reporting a loss.

    `store.week_key` still owns the padding, so the store and the site agree on how a week
    is spelled.
    """
    return f"preds_{season}_wk{store.week_key(week)}"


# --- weekly predictions ---------------------------------------------------

def predictions(season: int, week: int, base: Path | None = None,
                out: Path | None = None) -> dict[str, Any] | None:
    """One artifact per week, so the git history pins each week to a commit.

    Returns None and touches nothing when the store has no predictions for that week --
    the caller records it as stale rather than publishing an empty page.
    """
    out = out or SITE
    name = preds_name(season, week)
    # Asked, not caught. `store.predictions` distinguishes "this clone has no predictions
    # yet" from "the query is broken" -- a bare `except Exception` here once made a schema
    # break, a DuckDB lock and a typo in the SQL all arrive as
    # `"stale": true, "reason": "no predictions"`. It also returns one row per game rather
    # than one per fitted version: this page listed every week-1 game five times.
    df = store.predictions(season=season, week=week, base=base)
    # Empty first, and before the merge. Carrying forward is for a *partial* run -- one that
    # could still price some of the week. A run that priced none has nothing to say, and
    # re-stamping last-good as fresh is the defect this repo already fixed once, for the
    # track record; the merge reintroduced it here until this guard came back.
    if df.is_empty():
        return None
    rows = _keeping_published(df.to_dicts(), out / f"{name}.json", season)

    payload = jsonio.artifact(name, "preds", rows, season=season, week=week,
                              provenance=_obtainable(rows))
    return _publish(out, name, payload)


def _keeping_published(fresh: list[dict[str, Any]], path: Path,
                       season: int) -> list[dict[str, Any]]:
    """This run's predictions, plus any already published for a game it no longer returns.

    **The layer that matches what is committed.** `ratings._with_committed` carries a started
    game forward by reading the partition the previous run wrote, which works on a machine
    that keeps a store and not on an Actions runner: `data/processed/` is gitignored, so every
    scheduled run starts empty, finds no partition, and publishes only the games it can still
    predict. The artifact is what gets committed, so the game that had already kicked off was
    deleted from the public record -- trimming exactly the predictions reality had tested.

    A game the fresh query *does* return is replaced, not kept: it has not kicked off, so
    re-pricing it is legitimate and its commit still predates it (`docs/track-record.md`
    rule 1). Only games that have fallen out of the slate are carried.

    **Scoped to one season, and still scoped after the filenames were.** `preds_wk18.json`
    held 2025, so an unscoped merge handed 2026 week 18 all sixteen of the previous season's
    games and published two seasons as one slate -- reproduced at 17 rows. The artifact is
    named by season and week now (`preds_name`), which stops the collision at the file; this
    stays because a merge that reads a file must not depend on the filename being right.

    An unreadable artifact is treated as no artifact. It is about to be overwritten either
    way, and refusing to publish because the previous file is corrupt would turn one bad file
    into a stalled record.
    """
    have = {r.get("game_id") for r in fresh}
    if not path.exists():
        return fresh
    try:
        prior = json.loads(path.read_text()).get("rows", [])
    except (OSError, ValueError):
        print(f"  {path.name} is unreadable; publishing this run's rows alone", flush=True)
        return fresh
    kept = [r for r in prior
            if r.get("game_id") not in have and r.get("season") == season]
    if kept:
        print(f"  carried forward {len(kept)} published prediction(s) this run could not "
              f"make: {', '.join(sorted(str(r.get('game_id')) for r in kept))}")
    return sorted(fresh + kept, key=lambda r: str(r.get("game_id")))


def _obtainable(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each price source in this week, whether a reader can obtain the input behind it.

    A record whose credibility rests on a reader checking it has to say what a reader can
    check. `priced_at` names a dated snapshot and that citation is complete on the machine
    holding the store; the store is unpublished, so following it is something only one person
    can do. Saying so costs nothing and is the difference between a track record and an
    assertion.

    Once per source rather than once per row: it is a property of where the number came from,
    and sixteen copies of one sentence is noise in an artifact a person reads.

    A week written before `price_source` existed classifies nothing rather than raising --
    the store spans schemas by design, and an old partition must not take the page down.
    """
    # Off the rows rather than a frame: a published week mixes this run's rows with rows
    # carried forward from the committed artifact, whose timestamps are JSON strings rather
    # than datetimes, and polars cannot infer one schema across the two.
    used = sorted({r["price_source"] for r in rows
                   if isinstance(r, dict) and r.get("price_source")})
    return {s: schedule.provenance(s)._asdict() for s in used}


# --- calibration ----------------------------------------------------------

# The scoring rules moved to `hub.models.scoring_rules`; `hub.models.eval` was importing
# them from here, which meant model evaluation could not be read without the site writer.
# Re-exported because `publish` still uses all three and the site's own tests name them here.
def _published(out: Path) -> pl.DataFrame:
    """Every prediction this repo has published, read back from the artifacts it published.

    Not from the store, and the difference is the point. `docs/track-record.md` rule 1 counts
    a prediction because its *commit* predates kickoff, so the thing scored should be the
    thing pre-registered. Reading the store left the two free to differ: a prediction written
    to the store and never published would have been scored anyway, and one published from a
    store since rebuilt would have vanished from the record.

    It also makes the record reproducible by anyone. `data/processed/` is gitignored as
    redistributed third-party data, so a store-scored record could only ever be built on one
    machine -- which is why a scheduled run published an empty one over sixteen scored
    predictions before this. These artifacts are committed, so a reader can rebuild the same
    numbers from the same files.
    """
    rows: list[dict[str, Any]] = []
    # Both namings. `preds_wk18.json` was written before the season joined the filename and
    # is still the pre-registered record of those sixteen games; a glob that stopped seeing
    # it would drop them from the calibration -- the same loss this rename exists to prevent.
    for path in sorted(set(out.glob("preds_wk*.json")) | set(out.glob("preds_[0-9]*.json"))):
        try:
            rows += json.loads(path.read_text()).get("rows", [])
        except (OSError, ValueError):
            print(f"  track_record: {path.name} is unreadable and was skipped", flush=True)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _scored(out: Path) -> pl.DataFrame:
    """Published predictions joined to results. Empty frame when either side is missing."""
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "home_win_prob": pl.Float64,
                                 "home_won": pl.Int64, "predicted_at": pl.Utf8})
    preds = _published(out)
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
                 n_bins: int = 10) -> dict[str, Any] | None:
    """Calibration, and an honest count of what was actually pre-registered.

    `n_preregistered` is deliberately separate from `n_scored`. Scoring a prediction after
    the game is a backtest; the record only means something for predictions whose commit
    predates kickoff. Until the season starts both are zero, and the page says so rather
    than filling the space with a backfill dressed up as history.

    **Nothing to score returns None, which means keep last-good.** It used to return a valid
    payload describing nothing, and `Artifact.record` reads a payload as success -- so the
    first scheduled run published an empty record over one holding sixteen scored
    predictions, turning a log-loss of 0.556 into null and ten calibration bins into zero.
    The runner had no history to score because `data/processed/` is gitignored, and it had no
    way to know that "nothing" was a fact about the machine rather than about the season.

    `CLAUDE.md`'s degradation rule is written as *serve last-good rather than erroring*, and
    it is implemented as *None means keep last-good*. An empty-but-valid payload walks past
    it, so an artifact that can be produced emptier than its predecessor has to say None.
    """
    out = out or SITE
    df = _scored(out)
    if df.is_empty():
        return None

    payload: dict[str, Any] = {
        "name": "track_record", "source": "preds+results", "generated_at": jsonio.stamp(),
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
    # `detail` empty rather than absent: the poller fills it with per-game win probability
    # and this writer has none, and a reader written against one document must be written
    # against the other. Same keys, not merely compatible ones.
    payload = jsonio.artifact("live", "espn_scoreboard", rows, league=league, detail={})
    return _publish(out, "live", payload)


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
    # Before `lock`, which cannot price an empty pool and would raise rather than say
    # nothing. A parquet that exists and holds no rows is a sync that came back empty.
    if df.is_empty():
        return None
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
    payload = jsonio.artifact("roster", "roster.parquet", rows,
                        set_total=lk.set_total, optimal_total=lk.best_total, gain=lk.gain,
                        withheld=lk.withheld, start=lk.start, sit=lk.bench)
    return _publish(out or SITE, "roster", payload)


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
        Artifact(preds_name(season, week),
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
    man = {"generated_at": jsonio.stamp(), "season": season, "week": week, "artifacts": arts}
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
        # **From here, not from week 1.** Survivor is one assignment problem because spending
        # a team early costs you that team later -- so a grid that still prices played weeks
        # hands the solver its best teams for games that are over, and every remaining pick
        # comes from a pool degraded by picks that were never available. Both halves are wrong
        # in-season and neither shows in the output: the plan looks like a plan.
        ahead = sv.forthcoming(grid)
        gone = sv.played(grid)
        # What this entry has already used, read back from the plan it published. The only
        # record there is, and named as what it is in `spent_teams`.
        spent = sv.spent_teams(sv.published_plan(out / "survivor.json"), gone, season=season)
        # Against every week still to come, not against the weeks the grid happens to have --
        # asking coverage about its own weeks makes `missing` empty by construction and the
        # panel would never say a week needs a pick. A week already played is in neither list.
        remaining = [w for w in range(1, NFL_WEEKS + 1) if w not in gone]
        cov = sv.coverage(ahead, remaining)
        plan = sv.solve(ahead, weeks=cov["covered"], spent=spent)
    except Exception as e:
        # Reported the way `live` reports its own failure: printed for the operator, None for
        # the caller. It used to hand back a manifest entry of its own -- the only producer
        # that did -- which is why it could never be recorded like the rest.
        print(f"  survivor: schedule unavailable ({type(e).__name__}: {e})"[:160])
        return None
    art = jsonio.artifact("survivor", "hub.season.survivor", plan.to_dicts(), season=season,
                    survival=sv.survival(plan), unpriced_weeks=cov["missing"],
                    # The plan's own scope, said out loud. A survival probability means
                    # nothing without the weeks it is over, and a reader looking at a plan
                    # that starts in week 9 should not have to infer why.
                    weeks_remaining=cov["covered"], weeks_played=gone, spent=spent,
                    # Which weeks exist in this plan only because the store was read. They
                    # are the ones a reader should not expect to find on nflverse.
                    snapshot_only_weeks=sv.snapshot_only_weeks(ahead, cov["covered"]))
    return _publish(out, "survivor", art)


def default_week(season: int, base: Path | None = None) -> int:
    """The latest week already predicted, or 1 before the season starts.

    `--week` used to default to 1, so `make slate` with no week set would republish week 1
    every Sunday and look like it had worked. Read from the store rather than the network:
    a weekly refresh that needs a live API to decide what week it is has one more way to
    fail on a Sunday.
    """
    week = store.latest_week(season, base=base)
    return week if week is not None else 1


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
        print(f"  {preds_name(a.season, a.week)}: "
              + (f"{got['n']} games" if got else "nothing in the store; left as-is"))
        return 0
    if a.live:
        got = live()
        print("  live: " + (f"{got['n']} games" if got else "unavailable; last-good kept"))
        return 0
    if a.track_record:
        got = track_record()
        print("  track_record: " + (
            f"{got['n_scored']} scored, {got['n_preregistered']} pre-registered" if got
            else "nothing to score; last-good kept"))
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
