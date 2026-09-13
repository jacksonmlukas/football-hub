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
from typing import Any, NamedTuple, cast

import polars as pl

from hub import jsonio, schedule, store
from hub.config import SEASON_AHEAD, UNCONFIRMED_POOL_RULES, pool_digest, resolved_config
from hub.models import coverage
from hub.models.margin import home_won  # the repo's one tie convention -- issue #64
from hub.models.scoring_rules import brier, log_loss, reliability
from hub.paths import ROSTER_PARQUET
from hub.season.roster import lock

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"


def _write(out: Path, name: str, payload: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(jsonio.dumps(payload, indent=2))
    return p


class Kept(NamedTuple):
    """Last-good kept, and why. A producer that ran and had nothing to say.

    Whatever is published stands, which on a page with nothing published yet is nothing --
    the panel is stale either way, and what this carries is the sentence the reader gets.

    Three states rather than two. `Artifact.record` read a payload as fresh and `None` as
    stale, so `None` had to carry both "the source was never read" and "the source was read
    and came back empty" -- and the manifest stamped the producer's standing reason on a
    panel whose source had just answered. A reader was told "no roster yet -- run
    `python -m hub.season.roster --write`" about a sync that had run and found an empty
    league: the wrong fix, for a problem they do not have (issue #27).

    A returned sentinel rather than a flag set somewhere on the way past, because the reason
    belongs to the run that declined. Six producers sharing one module-level slot is the
    same defect one layer down.
    """
    why: str


class Reading(NamedTuple):
    """A producer's answer, decoded. The reader's half of the thing `Kept` is half of.

    `Kept` concentrates what a producer *says*. Nothing owned what a caller *hears*, so the
    three states were decoded at four sites and no two the same way: `Artifact.record` built
    the manifest entry inline, two CLI branches ran their own three-way cascade over
    different count keys, and the `live` branch checked two states and relied on whoever
    read it already knowing the third could not arrive. `CONTEXT.md` is explicit that the
    manifest's `reason` is all a reader gets, so four decodings of one answer were four
    chances for the panel to say the wrong sentence about why it is not current.

    `fresh` is this run's payload, or None. `reason` is None exactly when `fresh` is not,
    and otherwise is the producer's own sentence -- a `Kept` answered, so it says why in its
    own words -- or the panel's standing one, which belongs only to a producer that did not
    answer at all. Telling a reader to run a sync that had just run and found an empty
    league is issue #27, and it is the distinction this pair exists to keep.
    """
    fresh: dict[str, Any] | None
    reason: str | None

    @property
    def stale(self) -> bool:
        """What the manifest says about all three states, which is why it names none of
        them. The reason does that, and only the reason."""
        return self.fresh is None

    def sentence(self, said: Callable[[dict[str, Any]], str]) -> str:
        """One line for a reader. `said` renders a payload -- a row count, a pair of counts
        -- and is the only part of the decoding that differs between callers, which is why
        it is the only part they still write.

        The other two states already carry their words. That asymmetry is the point: a
        caller cannot accidentally give a producer that answered and a producer that did
        not the same sentence, because it never writes either one.
        """
        return said(self.fresh) if self.fresh is not None else cast(str, self.reason)


def read(answer: dict[str, Any] | Kept | None, standing: str, *,
         keeps: bool = True) -> Reading:
    """The one decoding of `Kept | payload | None`.

    `standing` is the panel's sentence for a producer that did not run, and is used for
    that state alone.

    `keeps` is the `live` branch's assumption, made checkable rather than remembered. That
    branch was written knowing `live` never answers `Kept` -- ADR-0018 has it relay someone
    else's fact, so there is no last-good of this repo's to hold -- and knowing it was all
    that kept a two-way check correct. An unstated invariant that a reader has to already
    hold is the shape this whole change is about, so it is stated: a producer declared not
    to keep and answering `Kept` is refused here rather than mis-rendered downstream.

    A refusal and not a degradation, for `hub.models.predict.CorrelationVoid`'s reason
    rather than against `CLAUDE.md`'s. Serving last-good needs a last-good to serve; what
    the fallback would put on the page here is a frozen overlay under a fresh timestamp, on
    the one artifact whose whole job is to move, and nothing downstream could tell. It
    cannot fire on a bad Sunday either, only on a change ADR-0018 forbids.
    """
    if isinstance(answer, dict):
        return Reading(answer, None)
    if isinstance(answer, Kept):
        # GUARD a-producer-that-cannot-keep-is-refused [unit/test_publish.py]: deleting it
        # puts `live` back to an assumption a reader has to already hold.
        if not keeps:
            raise ValueError(
                "a producer declared not to keep last-good answered Kept: "
                f"{answer.why!r}. ADR-0018 is why `live` is declared that way -- a live "
                "score is relayed, not asserted, so there is nothing of this repo's to "
                "keep. Either the producer or the declaration is wrong.")
        # /GUARD
        return Reading(None, answer.why)
    return Reading(None, standing)


def _last_good_n(out: Path, name: str, key: str = "n") -> int:
    """How many rows the artifact already on disk carries. Unreadable or absent counts 0.

    **`key` because the track record counts in `n_scored` where the row-shaped panels count
    in `n`.** Asking the wrong key reads every published record as empty, which lets an empty
    one replace it -- the exact regression `_keeping` below exists to prevent. This is the
    one statement of that; `_publish`'s `count_key` is this argument arriving from a caller.

    Named apart from `_published`, which returns every prediction *row* the site has ever
    published. These were `_published_n` and `_published`: near-homographs, in one module,
    answering unrelated questions.
    """
    try:
        got = json.loads((out / f"{name}.json").read_text())
    except (OSError, ValueError):
        return 0
    return int(got.get(key, 0) or 0) if isinstance(got, dict) else 0


def _keeping(out: Path, name: str, why: str, key: str = "n") -> Kept | None:
    """Whether there is already an artifact to keep, and the sentence for it.

    **On the file existing, not on it having rows.** Asking "are there rows worth
    protecting" fell through for an artifact already published empty, and the caller then
    rewrote it -- advancing its `generated_at` on every run, which is the re-stamp the
    carry-forward did once already. An empty artifact is still a published artifact and is
    still what the page is reading.

    None when there is no file at all: nothing to keep, so the caller decides whether to
    write one.

    Split out of `_publish` for `predictions`, which has to answer emptiness *before* the
    carry-forward merge and so cannot hand its payload over -- one implementation of the
    question, two places that ask it, rather than the four that grew here once already.
    """
    if not (out / f"{name}.json").exists():
        return None
    had = _last_good_n(out, name, key)
    # The printed line branches with the returned one. It used to say "0 rows against 0
    # already published; keeping last-good" whenever the published artifact was itself
    # empty -- an operator reading the slate's log was told last-good was being kept and
    # that there was nothing to keep, in one sentence.
    print(f"  {name}: read and empty; keeping the {had} row(s) last published" if had
          else f"  {name}: read and empty, and so is the artifact already published",
          flush=True)
    return Kept(f"{why}; keeping the {had} row(s) last published" if had
                else f"{why}, and so is the artifact already published")


def _publish(out: Path, name: str, payload: dict[str, Any],
             count_key: str = "n") -> dict[str, Any] | Kept:
    """Write the artifact, unless doing so would replace something with nothing.

    **The one place this rule lives**, over the one question `_keeping` above answers.
    `CLAUDE.md`'s degradation rule is implemented as
    *a producer returning None means keep last-good*, and an empty-but-valid payload walks
    straight past it: `Artifact.record` reads any payload as success. That has bitten three
    times -- `track_record` published an empty record over sixteen scored predictions and
    turned a log-loss of 0.556 into null; the carry-forward re-stamped last-good as fresh;
    `roster` guarded `not src.exists()` and nothing else, so an ESPN sync returning an empty
    league would have published a roster of nobody. Each was fixed where it was found, which
    is why the next one was a surprise.

    Not "never write an empty artifact", and not "an empty answer is a fresh one" either.
    Both were tried. A first run on a day with no games has to leave the page a file it can
    read, so the artifact is written -- but issue #22's criterion is about *freshness*, not
    about rows: "a run that produces nothing leaves the previous artifact's timestamp
    untouched -- the panel goes stale with a reason rather than looking current". So an
    empty payload is never reported fresh, and it never rewrites a file that already exists:

    * empty, something already published -- keep it, write nothing, say why.
    * empty, nothing published at all -- write it once so the page has a valid file, and
      still report `Kept`, because "here is an artifact of nothing" and "this is current"
      are different claims and only the first one is true.

    Reporting that first run fresh made a source that ran and came back empty
    indistinguishable from one that succeeded -- verbatim the shape #22 was filed against --
    and, because `_last_good_n` reads the file back as zero rows, every later run fell
    through the guard and rewrote it.

    **Every panel this module writes goes through here, `live` excepted** -- including the
    empty ones: `roster` builds an artifact of no players rather than deciding for itself,
    and `track_record` an artifact of nothing scored. Written as a rule about *writes*
    rather than as a count of producers, because that count has now been recorded wrong
    twice: it said one bypass when there were three, then two when there were three.

    `live` is the exception, and ADR-0018 is why: every deploy fetches the previously
    published overlay back before refreshing it, so something is always published and the
    first-run case above can never arise for it -- keeping last-good would only ever freeze
    the one artifact whose job is to move, and a live score is someone else's fact relayed
    rather than a claim this repo makes.

    Two more things in this module are not covered by that sentence and neither is a bypass.
    `predictions` writes through here when it has rows and asks `_keeping` -- the same
    question, one implementation -- when it has none, because it must answer before the
    carry-forward merge rather than after building a payload. `draft_board` writes nothing
    here at all: `hub.draft.board` writes that file and `_board` only reports whether it is
    there. (`survivor` does hand its payload over; its own `Kept` comes from `Infeasible`,
    which is a failure to produce and a different question from this one.)

    That this rule is *reachable* is the point of writing it once: it was unreachable for a
    day, because each producer had learned to answer emptiness before calling, and deleting
    the lines below left every test passing.

    `count_key` is `_last_good_n`'s `key`, and is documented there.
    """
    why = "the source was read and came back empty"
    # GUARD last-good-not-blanked: an empty payload never replaces a published one
    if not payload.get(count_key):
        if kept := _keeping(out, name, why, count_key):
            return kept
        _write(out, name, payload)
        return Kept(f"{why}; published as an empty artifact so the page has one to read")
    # /GUARD
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
                out: Path | None = None) -> dict[str, Any] | Kept | None:
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
        return _keeping(out, name, "the store held no prediction for this week")
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
    # Every row scanned for the schema, not polars' default first hundred. `season` is on
    # some artifacts and not others -- `preds_wk18.json` predates the field -- and a column
    # polars does not see is a season `_by_season` cannot tell apart, which is the pooling
    # it exists to stop arriving by the back door. Measured: 101 rows without the key ahead
    # of one with it drops the column entirely.
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _scored(out: Path) -> pl.DataFrame | None:
    """Published predictions joined to results.

    An empty frame means the sources answered and nothing is scorable yet -- no predictions
    published, or none of them finished. **None means the results could not be fetched at
    all**, which is not the same fact and must not reach the page as a record of nothing:
    the caller keeps last-good instead of publishing zero scored predictions because
    nflverse was down for ninety seconds.

    **What a tied game means, and this side used to say nothing about it.** The outcome
    comes from `hub.models.margin.home_won`, which drops a tie rather than scoring it. This
    derived it inline as `result > 0`, so a tie arrived as a home loss and log loss took
    full credit for it -- a confident correct call on a game nobody won, awarded to whichever
    model gave the home side the lower probability, on the one path that feeds the public
    record. `hub.models.eval` had always dropped ties, so the repo answered the question two
    ways and only the wrong answer was published (issue #64). Latent when it was fixed:
    when the fix landed on 2026-09-05, sixteen of the 32 published predictions were scored
    and none was a tie, so no published number moved. That was a count taken by hand
    against nflverse, true until the first slate that finishes level; `track_record` now
    publishes it as `n_tied`, re-derived from `_finished` on every run, so the claim is the
    artifact's and not this docstring's (issue #52).
    """
    finished = _finished(out)
    return None if finished is None else _record_of(finished)


def _finished(out: Path) -> pl.DataFrame | None:
    """Published predictions joined to every game that has a result, ties included.

    One fetch, two readers: `_record_of` keeps the rows the record can score, and
    `track_record` counts the ties that leave it. Same None-versus-empty contract as
    `_scored`, for the same reason.
    """
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "home_win_prob": pl.Float64,
                                 "result": pl.Int64, "predicted_at": pl.Utf8})
    preds = _published(out)
    if preds.is_empty():
        return empty

    try:
        import nflreadpy as nfl
        sched = (nfl.load_schedules().select("game_id", "result").drop_nulls("result")
                 .with_columns(pl.col("result").cast(pl.Int64)))
    except Exception as e:
        # This one stays broad -- it is a network call on a schedule that runs unattended --
        # but it says which failure it was rather than presenting every cause as "no results".
        print(f"  schedules unavailable ({type(e).__name__}); track record left as "
              f"published", flush=True)
        return None
    return preds.join(sched, on="game_id", how="inner")


def _record_of(finished: pl.DataFrame) -> pl.DataFrame:
    """The scorable rows of `finished`, with the outcome the scoring rule reads."""
    return home_won(finished).drop("result")


def _n_tied(finished: pl.DataFrame) -> int:
    """How many published predictions were on a game that finished level.

    Counted off the finished games and not off what the record dropped, so it is the same
    number whichever way `margin.DROP_TIES` is set: with ties dropped it is what the record
    left out, with them kept it is what the record scored as a home loss. Either way the
    page says how many, which is what a hand count in a docstring could not keep doing.
    """
    return int((finished["result"] == 0).sum()) if finished.height else 0


def track_record(base: Path | None = None, out: Path | None = None,
                 n_bins: int = 10) -> dict[str, Any] | Kept | None:
    """Calibration, and an honest count of what was actually pre-registered.

    `n_preregistered` is deliberately separate from `n_scored`. Scoring a prediction after
    the game is a backtest; the record only means something for predictions whose commit
    predates kickoff. Until the season starts both are zero, and the page says so rather
    than filling the space with a backfill dressed up as history.

    **Nothing to score does not blank the record, and `_publish` is what enforces that.**
    This used to return a valid payload describing nothing, and `Artifact.record` reads a
    payload as success -- so the first scheduled run published an empty record over one
    holding sixteen scored predictions, turning a log-loss of 0.556 into null and ten
    calibration bins into zero. The runner had no history to score because
    `data/processed/` is gitignored, and it had no way to know that "nothing" was a fact
    about the machine rather than about the season.

    It was then fixed *here*, with an early return, which is how the shared guard came to be
    unreachable. The empty payload is built and handed over like any other: with a record
    already published `_publish` keeps it and says the source was read; with nothing
    published it writes the empty record once, so the page has a file with its note in it,
    and still reports it as not current.

    A *failed* results fetch is the different fact `_scored` reports as None, and that one
    does return None: an empty record on a machine whose network is down is a claim about
    the season made from a fact about the machine.

    **One curve per season, never one across two.** `_published` returns every artifact the
    site holds, and it holds `preds_2025_wk18.json` beside `preds_2026_wk01.json`. Pooled,
    the two are scored as though one model had made them: a calibration failure in either is
    diluted by the other, and the log loss that comes out describes no season anybody ran
    (issue #23). The top-level `log_loss`/`brier`/`bins` stay, as the *newest* season's, so
    a reader written against them keeps working and reads one season instead of a blend.
    """
    out = out or SITE
    finished = _finished(out)
    if finished is None:
        return None
    df = _record_of(finished)

    seasons = [_curve(season, rows, n_bins) for season, rows in _by_season(df)]
    # An empty record still carries the trio, as nulls: `site/index.html` reads them off the
    # top level unconditionally, and a missing key and a null one render differently.
    newest = seasons[0] if seasons else {"bins": [], "log_loss": None, "brier": None}
    # Through `jsonio.summary` rather than three literal envelope keys, so `shape: "summary"`
    # is written by the writer that knows this record has no rows. Until #227 that fact lived
    # in `NOT_ROW_SHAPED` in the contract instead, where nothing connected it to here.
    payload: dict[str, Any] = jsonio.summary(
        "track_record", "preds+results",
        n_scored=df.height,
        # The ties the record could not score, said as a number: the #64 fix rested on there
        # being none among the published predictions, and that stays true only as long as
        # something re-derives it (#52).
        n_tied=_n_tied(finished),
        # Nothing is pre-registered until a prediction is committed before kickoff, which
        # the Sunday Actions job does. Counting it here would be marking my own homework.
        n_preregistered=0,
        is_backtest=df.height > 0,
        note=("No pre-registered predictions yet. A prediction counts only once its "
              "commit predates kickoff -- see docs/track-record.md."),
        seasons=seasons,
        bins=newest["bins"], log_loss=newest["log_loss"], brier=newest["brier"],
        # The other half of the record, and the half that was never on the page. Log loss
        # and Brier score the *game* probability; `interval_coverage` is whether the weekly
        # player interval covers, from `hub.models.coverage`. It reads the last committed
        # measurement rather than running one -- a Sunday refresh does not fit five seasons
        # of player-weeks -- and it is None when none has been made, which is the graceful
        # degradation `CLAUDE.md` requires: a missing research artifact drops a field, it
        # does not take the page down.
        interval_coverage=coverage.published_summary(),
    )
    return _publish(out, "track_record", payload, count_key="n_scored")


def _by_season(df: pl.DataFrame) -> list[tuple[int | None, pl.DataFrame]]:
    """The scored rows split by the season they belong to, newest first.

    A row carrying no season goes in a group of its own, ordered last, and keeps its
    `season: null` in the artifact. `preds_wk18.json` was written before the field existed
    and its sixteen games are the whole of the site's committed 2025 calibration, so
    dropping them would lose the record this scoring exists to keep; folding them into the
    newest season would put rows nothing can date inside a curve that names one. Neither is
    a thing to do silently, so the page is told which they are.
    """
    if not df.height:
        return []
    if "season" not in df.columns:
        return [(None, df)]
    # Cast rather than compared as it arrives. Both committed artifacts carry integers, so a
    # `season` of `"2026"` is unreachable today -- but it comes off JSON that a hand edit or
    # an older writer could have typed, and `pl.col("season") == 2026` against a Utf8 column
    # raises `ComputeError` and takes the whole record down. Unparseable lands in the
    # undated group, which is the group that exists for rows nothing can date.
    df = df.with_columns(pl.col("season").cast(pl.Int64, strict=False))
    known = sorted({int(s) for s in df["season"].drop_nulls().to_list()}, reverse=True)
    groups: list[tuple[int | None, pl.DataFrame]] = [
        (s, df.filter(pl.col("season") == s)) for s in known]
    undated = df.filter(pl.col("season").is_null())
    if undated.height:
        groups.append((None, undated))
    return groups


def _curve(season: int | None, df: pl.DataFrame, n_bins: int) -> dict[str, Any]:
    """One season's calibration: what it scored, and how well."""
    probs = df["home_win_prob"].to_list()
    won = df["home_won"].to_list()
    return {"season": season, "n_scored": df.height,
            "log_loss": log_loss(probs, won), "brier": brier(probs, won),
            "bins": reliability(df, n_bins)}


# --- the live overlay -----------------------------------------------------

# What `--live` exits with when ESPN could not be reached. Distinct from 0 (published) and
# from 1 (this program is broken), because the unattended refresher has to tell "nothing to
# publish" from "publish and deploy" without reading prose off stdout.
NOTHING_FRESH = 3


def live(out: Path | None = None, league: str = "nfl") -> dict[str, Any] | None:
    """Current scores, written separately from the model's numbers on purpose.

    This is the only artifact that changes during a Sunday, and keeping it in its own file
    is what lets the page hold the two apart. Predictions are frozen at lock and must never
    be regenerated mid-slate: a win probability that drifts while games are in progress is
    indistinguishable from one that was always going to look right, which destroys the
    pre-registration the whole record depends on.

    So: scores move, model numbers do not, and the page says which is which.

    **The one producer that does not go through `_publish`, and ADR-0018 is why.** The ADR
    quotes the paragraph above and draws the line it rests on: a live score is *someone
    else's fact*, ESPN's, checkable against ESPN at the time, and this repo asserts nothing
    by relaying it. So a scoreboard with no games is ESPN saying there are no games -- a
    Tuesday, or a slate that has finished -- and writing that is the relay working, not an
    empty answer replacing a full one. `_publish`'s own exception for an honest first empty
    answer cannot rescue it either: the ADR has every deploy fetch the previously published
    overlay back before refreshing it, so something is always already published and the
    guard would freeze the one artifact whose whole job is to move.

    A *failed* fetch is the case the ADR does protect, and it is the `return None` below:
    nothing is written, the carried-forward file stays exactly as it was, and the manifest
    marks the panel stale.

    **A cache-served fetch is a failed fetch**, and issue #91 is what happens when it is not
    treated as one. The fetch layer degrades to last-good, so a scoreboard can arrive here
    that ESPN was never asked for -- and this function stamps what it is handed. The overlay
    would then carry a fresh `generated_at` over frozen scores, the heartbeat the watchdog
    reads would say the page is current, and the freeze would erase its own evidence. So the
    read below refuses the cache, and every unreachable-ESPN outcome comes out of the same
    `return None`.

    **The cost, accepted.** A 200 from ESPN carrying no events mid-slate now blanks the
    overlay, where a last-good guard would have held Sunday's scores on the page until the
    next poll. That is the right trade for a relay and not an oversight: the same response
    is how a finished slate and a Tuesday look, this repo cannot tell those apart from the
    outside, and holding scores that ESPN is no longer reporting would be asserting
    something ESPN is not. The page ages the panel from `generated_at`, so a stale-looking
    empty overlay is visible as one. Do not put the guard back without reopening ADR-0018.
    """
    out = out or SITE
    try:
        from hub.fetch.espn import live_state
        # `allow_cache=False`, because the artifact this writes is a *timestamp* as much as a
        # scoreboard. `hub.fetch.espn._get` degrades to its last-good cache on failure, which
        # is right for the dashboard and wrong here: the cached payload would arrive
        # indistinguishable from a live one, be stamped with this moment, and publish a
        # heartbeat saying the page is fresh while it shows scores nobody has refreshed. The
        # watchdog reads exactly that stamp, so it could never fire again -- issue #91.
        # Refusing the cache turns that into the failed fetch it actually is.
        rows = live_state(league, allow_cache=False)
    except Exception as e:
        print(f"  live: ESPN unavailable ({type(e).__name__}); leaving last-good in place")
        return None
    # `detail` empty rather than absent: the poller fills it with per-game win probability
    # and this writer has none, and a reader written against one document must be written
    # against the other. Same keys, not merely compatible ones.
    payload = jsonio.artifact("live", "espn_scoreboard", rows, league=league, detail={})
    _write(out, "live", payload)
    return payload


# --- the roster, and the lineup it implies --------------------------------

def roster(out: Path | None = None,
           path: Path | None = None) -> dict[str, Any] | Kept | None:
    """Serialise the roster and the lock decision. Does not *make* the decision.

    Reads `data/processed/roster.parquet` rather than ESPN, so the panel is publishable
    without a network round trip and shows last-good when a sync fails -- the same contract
    every other artifact here keeps.

    `season.roster.lock` computes set-versus-best -- the lineup it would set, the moves that
    reach it, and which players are withheld as unavailable. That arithmetic lived here for
    one evening, which meant the only way to ask the Sunday question was to publish a website;
    it is a decision, not a rendering. Nothing here re-derives any part of it.
    """
    src = path or ROSTER_PARQUET
    if not src.exists():
        return None
    df = pl.read_parquet(src)
    # No empty check here. There was one, justified by `lock` being unable to price an empty
    # pool -- and `lock` does no such thing: on a zero-row frame it returns a `Lock` of Nones
    # and empty lists. The excision harness found it, by deleting the branch and watching
    # `test_publish.py` stay green. An empty frame falls through to `_publish`, which is the
    # one place this rule is meant to live; `hub.season.roster.write` refuses to create the
    # empty parquet in the first place.
    lk = lock(df)
    # The lock's own lineup, read rather than rebuilt. This used to reconstruct it from the
    # two deltas -- the set starters, less those to sit, plus those to start -- which agrees
    # with the lock wherever it priced a comparison and says nothing at all where it did not.
    # Both of `lock`'s declining branches return empty deltas, so the subtraction and the
    # union each did nothing, `start` collapsed to the lineup as set, and every current
    # starter published `best_start: true` under a null `gain`. The page renders that as a
    # best-XI tick, so a roster that can field no legal lineup ticked everybody (issue #130).
    # An empty lineup ticks nobody, which is what declining means.
    best = set(lk.best_lineup)

    rows = []
    for r in df.iter_rows(named=True):
        rows.append({
            "player": r["player"], "pos": r["pos"], "nfl_team": r["nfl_team"],
            "mu": r["mu"], "sd": r["sd"], "projected": r["projected"],
            "starting": r["starting"], "best_start": r["player"] in best,
            "injury_status": r["injury_status"],
            "available": r.get("available", True),
            # Two neighbouring facts, and the panel needs both: `available` is whether we
            # expect him to play, `can_start` whether the league will let him.
            "can_start": r.get("can_start", True),
            "missing_games": r.get("missing_games", 0),
        })
    # Dated from the parquet, not from this run. The roster CLI serves last-good when ESPN is
    # unreachable and no longer rewrites the file, so this mtime is the last *successful*
    # sync -- and the page ages the panel from `generated_at`. Stamping now instead is how
    # last week's starters, withheld list and add/drops published as this week's, every
    # Sunday, on the one panel about the operator's own team (issue #122).
    payload = jsonio.artifact("roster", "roster.parquet", rows,
                        as_of=jsonio.file_stamp(src),
                        set_total=lk.set_total, optimal_total=lk.best_total, gain=lk.gain,
                        withheld=lk.withheld, start=lk.start, sit=lk.bench)
    return _publish(out or SITE, "roster", payload)


# --- everything, plus a manifest -----------------------------------------

def _generated_at(path: Path) -> str | None:
    """Last-good's timestamp, or None if the file cannot be read.

    The type branch is gone with #107. It existed for `draft_board.json` alone, which was a
    bare list of rows, so a bare `.get` on it raised -- one accommodation of four for the one
    artifact that skipped the envelope. It carries the envelope now, like every other file
    here, so there is one shape to read.
    """
    try:
        return json.loads(path.read_text()).get("generated_at")
    except (OSError, ValueError, AttributeError):
        # Unreadable, not JSON, or JSON that is not an object. The last is caught rather than
        # branched on: `isinstance(got, dict)` here meant "there is a second artifact shape",
        # and #107 removed the artifact that had one. What is left is a file that is damaged
        # or hand-edited, which is a failure to read rather than a shape to support.
        return None


class Artifact(NamedTuple):
    """One published file, and the freshness contract every one of them keeps.

    `CLAUDE.md`'s degradation rule -- *a panel whose data is missing says so and keeps
    rendering* -- used to be written three ways inside `publish_all`: a `record` closure for
    four artifacts, a hand-rolled dict for `draft_board`, and `survivor` returning its own
    manifest entry. The two that bypassed `record` were also the two that always reported
    `generated_at: null`, so the page could not age them.

    A producer returns its payload; a **`Kept`** meaning "I ran, and whatever is published
    stands -- here is why"; or **None**, meaning there is nothing fresh *and* nothing kept,
    so the panel's standing reason is the best sentence available. It never builds a manifest
    entry itself; that is this module's single job.

    `keeps` is whether this producer can answer `Kept` at all. Every one here can except
    `live`, which ADR-0018 keeps out of the last-good rule entirely; `read` refuses rather
    than rendering if that ever stops being true.
    """
    name: str
    produce: Callable[[], dict[str, Any] | Kept | None]
    reason: str
    keeps: bool = True

    def record(self, out: Path) -> dict[str, Any]:
        # Decoded by `read` and not here. This site and the three CLI branches were four
        # spellings of one cascade, and the manifest's `reason` is all a reader gets.
        seen = read(self.produce(), self.reason, keeps=self.keeps)
        path = out / f"{self.name}.json"
        present = path.exists()
        return {
            "name": self.name, "present": present, "stale": seen.stale,
            "reason": seen.reason,
            "generated_at": (seen.fresh.get("generated_at") if seen.fresh
                             else (_generated_at(path) if present else None)),
        }


def _board(out: Path) -> dict[str, Any] | None:
    """`draft_board.json` has no producer here -- `hub.draft.board` writes it.

    So this reports whether it is there and hands back what it says about itself. It used to
    return `{"generated_at": None}`, which is why the board panel could never be stale and
    could never be aged: the one artifact whose freshness matters most on draft night was the
    one the page could not date (issue #107). ADR-0020 is untouched -- what changed is the
    file's shape, not who writes it.
    """
    p = out / "draft_board.json"
    if not p.exists():
        return None
    try:
        got = json.loads(p.read_text())
    except ValueError:
        return None
    return got if isinstance(got, dict) else {"generated_at": None}


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
        # `keeps=False`: ADR-0018 keeps `live` out of the last-good rule, so a `Kept` from
        # it is a contract violation rather than a panel to render.
        Artifact("live", lambda: live(out=out), "ESPN scoreboard unavailable", keeps=False),
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


def survivor(season: int, out: Path | None = None,
             store: Path | None = None) -> dict[str, Any] | Kept | None:
    """The survivor plan, as its own artifact.

    Wrapped rather than inlined because it reaches the network for a schedule. A failing
    source marks the panel stale and leaves the last good plan in place -- taking the whole
    page down over one panel is the operator-dependence CLAUDE.md warns about.

    `store` is the processed store the pool host's last-known state is read from for the
    Ledger (#280); the default is the real one, and a test passes its own.
    """
    from hub.season import survivor as sv
    out = out or SITE
    # The pool's rules as a run actually resolves them -- `conf/` applied -- and not a bare
    # `PoolConfig()`. This was the one consumer that publishes and the one that built the
    # defaults by hand, so an override of the double-pick weeks or of the week buybacks run
    # through reached four other call sites and never the published Remaining plan (#160).
    # That falsified the survivor plan's own commitment that pool rules are configuration: a
    # rule correction was a re-run everywhere except the place a reader sees.
    pool = resolved_config().pool
    try:
        # The remaining plan *and its scope*, from `hub.season.survivor` rather than
        # assembled here. This function used to make nine `sv.*` calls orchestrating
        # survivor's own data, and the ones that matter were written out again in
        # `survivor.main`. `plan_remaining`'s own docstring enumerates that sequence and is
        # the one place it is written; restating it here is what put "six steps" over five
        # items into two modules at once (issue #53). That sequence is the whole of issue
        # #24's rule, so two copies is one plan silently wrong.
        got = sv.plan_remaining(sv.grid_from_schedule(season), season,
                                # The pool's own rules, so weeks 13-18 take two teams here
                                # even though a bare `solve` still takes one.
                                pool=pool,
                                # What this entry has already used: the remaining plan it
                                # published and the pool host's Ledger for it, through the
                                # one reading of both (#280).
                                prior=sv.prior_rows(season, path=out / "survivor.json",
                                                    store=store))
    except sv.Infeasible as e:
        # Caught apart from the failure below, because it is not one. The schedule answered;
        # what is missing is a *posted spread* for the weeks that remain, or a pool of teams
        # that cannot cover them. Under one `except Exception` the panel read "schedule
        # unavailable" in August, when nothing was unavailable and the betting market simply
        # had not posted -- the reader sent to fix a problem they do not have (issue #27).
        print(f"  survivor: no plan ({e})"[:160])
        return Kept(f"no plan for the weeks that remain: {e}")
    except Exception as e:
        # Reported the way `live` reports its own failure: printed for the operator, None for
        # the caller. It used to hand back a manifest entry of its own -- the only producer
        # that did -- which is why it could never be recorded like the rest.
        print(f"  survivor: schedule unavailable ({type(e).__name__}: {e})"[:160])
        return None
    # The rules this plan rests on that nobody has confirmed, carried beside the numbers
    # rather than left for a reader to know. `co_survivor_rule` decides how a shared pot
    # splits, so every dollar figure here is conditional on it -- publishing them without
    # saying so is the unlabelled claim `docs/method.md` exists to stop. Read from
    # `hub.config` so this cannot disagree with the settings it describes.
    art = jsonio.artifact("survivor", "hub.season.survivor", got.picks.to_dicts(),
                          unconfirmed=list(UNCONFIRMED_POOL_RULES),
                          # Which rules this plan was computed under, as a digest a reader can
                          # compare between two publishes. Built for exactly this and, until
                          # #160, called by nothing in production -- so two runs under
                          # different rules produced identical artifacts *and* identical
                          # provenance.
                          pool_digest=pool_digest(pool),
                    season=season,
                    survival=got.survival, unpriced_weeks=got.coverage.missing,
                    # The remaining plan's own scope, said out loud. A survival probability means
                    # nothing without the weeks it is over, and a reader looking at a plan
                    # that starts in week 9 should not have to infer why.
                    #
                    # Every one of these is rendered by the survivor panel, and
                    # `test_every_field_the_survivor_artifact_publishes_is_read_by_the_page`
                    # is what keeps that true. `weeks_played` was published here and read by
                    # nothing at all for a while -- no page code, no test, no CLI -- which is
                    # a claim in a public artifact that no reader could ever check (#57).
                    weeks_remaining=got.coverage.covered, weeks_played=got.played,
                    spent=got.spent,
                    # Which weeks exist in this plan only because the store was read. They
                    # are the ones a reader should not expect to find on nflverse.
                    snapshot_only_weeks=got.snapshot_only)
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
    # Imported here rather than at module scope for the reason `live` imports `live_state`
    # inside the function: `hub.fetch.espn` opens its cache directory on import, and
    # `--help` must reach nothing. The names of the two boards have one owner, so `--league`
    # takes its choices from there rather than restating them.
    from hub.fetch.espn import LEAGUE_PATHS

    ap = argparse.ArgumentParser(
        prog="hub.publish", description="Write site/data/*.json from the processed store.")
    ap.add_argument("--all", action="store_true", help="every artifact plus the manifest")
    ap.add_argument("--predictions", action="store_true", help="one week's predictions")
    ap.add_argument("--track-record", action="store_true", help="calibration page data")
    ap.add_argument("--live", action="store_true", help="live scores overlay only")
    ap.add_argument("--league", default="nfl", choices=sorted(LEAGUE_PATHS),
                    help="which board the overlay is for; the window decides (--live only)")
    # The destination is a parameter of every producer in this module and was the one thing
    # only the CLI could not say, which is the seam `test_cli_surface` threaded `--store`
    # through the store-backed commands for: a command that can only write the real file can
    # only be tested by something that is not the command.
    ap.add_argument("--out", type=Path, default=None,
                    help="where the overlay is written; defaults to site/data (--live only)")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, default=None,
                    help="defaults to the latest week already predicted")
    a = ap.parse_args(argv)
    if a.week is None:
        a.week = default_week(a.season)

    # `read` rather than a truth test on any of these: a `Kept` is a non-empty tuple and so
    # is truthy, and indexing it for a row count raises. Each branch now writes only the
    # sentence for its own payload; the other two states carry their own words.
    if a.predictions:
        seen = read(predictions(a.season, a.week), "nothing in the store; left as-is")
        print(f"  {preds_name(a.season, a.week)}: "
              + seen.sentence(lambda got: f"{got['n']} games"))
        return 0
    if a.live:
        seen = read(live(out=a.out, league=a.league), "unavailable; last-good kept",
                    keeps=False)
        print("  live: " + seen.sentence(lambda got: f"{got['n']} games"))
        # Three outcomes, two exit codes, and the split is the one the refresher acts on:
        # games and no games are both ESPN answering, and both are worth publishing
        # (ADR-0018). Not reaching ESPN at all is the third, and nothing about it should
        # reach the page -- no deploy, and no new `generated_at` over scores nobody
        # refreshed. `NOTHING_FRESH` rather than 1 because this is not a failure: the
        # overlay is intact, last-good stands, and the caller is being told which of the
        # three happened. `.github/scripts/live-loop.sh` is the caller that branches on it.
        return NOTHING_FRESH if seen.stale else 0
    if a.track_record:
        seen = read(track_record(), "nothing to score; last-good kept")
        print("  track_record: " + seen.sentence(
            lambda got: f"{got['n_scored']} scored, "
                        f"{got['n_preregistered']} pre-registered"))
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
