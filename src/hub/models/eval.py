"""Comparing two models honestly.

`docs/foundation-plan.md` 3.4. The job is not to produce a number that flatters whatever was
just built -- it is to make "is this better than the market?" answerable *with an interval*,
so that the answer can be no.

The correctness test the plan names is the boring one, and it is the right one: score a model
against itself and the delta must be exactly zero with an interval containing zero. A harness
that cannot report "no difference" when there is none will cheerfully report an edge that is
not there, and everything it says afterwards is worthless.

Splits are temporal by default. A random split leaks -- a prediction for week 10 that had
week 12 in its fit is not a prediction -- and that is `docs/track-record.md` rule 1 one layer
up.

    uv run python -m hub.models.eval --compare market_baseline,ratings_v2
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from hub.models.experiment import BOOTSTRAP
from hub.models.margin import home_won  # the repo's one tie convention -- issue #64
from hub.models.scoring_rules import brier, log_loss, reliability

DEFAULT_HOLDOUT = 0.3


class NoOverlap(Exception):
    """The two models never predicted the same game."""


class OutcomesUnavailable(Exception):
    """The realised outcomes could not be read, so there is nothing to score against.

    Distinct from `NoOverlap`, which is about the predictions: two models sharing no games
    is a comparison that *cannot* be made, and an absent outcome source is one that could
    not be *attempted*. Different fixes, so a different exception -- and the two causes
    inside this one (schedules unreachable, schedules answering without `result`) carry
    different messages for the same reason. Two messages rather than two more types
    because the CLI's job with either is to print it; what a reader needs is which of the
    two happened, and that is a sentence. Issue #63.
    """


def _schedules() -> pl.DataFrame:                                    # pragma: no cover
    """nflverse's schedules, fetched only when the caller did not supply them.

    A named function rather than an inline import so a test can drive the CLI against a
    temporary store without touching the network. `nflreadpy` is configured to cache in
    memory, so the two `load_predictions` calls one comparison makes cost one fetch.
    """
    import nflreadpy as nfl
    return nfl.load_schedules()


def _nothing_predicted() -> pl.DataFrame:
    """The frame for a model the store holds no predictions for.

    Named for what returning it asserts rather than for what it is made of -- it was
    `empty_shape`, which described the columns and not the fact. The fact is the useful
    part: this model has predicted nothing, so the frame flows through to `_paired`, whose
    NoOverlap message is the true answer for it.

    A function rather than a local built at the top of `load_predictions`, which paid to
    construct it on every call including the ones that go on to fetch and join.
    """
    return pl.DataFrame(schema={"game_id": pl.Utf8, "season": pl.Int32,
                                "week": pl.Int32, "model": pl.Utf8,
                                "home_win_prob": pl.Float64, "home_won": pl.Int64})


def load_predictions(model: str, base: Path | None = None,
                     schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """One model's predictions with the realised outcome joined on.

    This used to raise `NoOverlap(f"{model} has no scored outcomes yet")` for any frame
    without `home_won`, and the store has never carried that column -- so the store path was
    unreachable and `--compare` could not score against a real store at all. Every test in
    this module handed `compare` a frame that already had `home_won`, which is why the dead
    path read as working. Issue #59.

    `preds` records what was predicted and never what happened: `hub.publish` writes a row
    before kickoff and does not revisit it. The outcome lives in nflverse's schedules as
    `result`, home score minus away. So scoring a prediction is a join, not a column -- the
    same shape `hub.models.conformal.load_scored` settled on for the realised margin -- and
    an unplayed game does not survive it. That matters more here than in conformal: a
    prediction for an unplayed game defaulting to a home loss is scored by log loss exactly
    as confidently as a real one, and nothing downstream can tell the two apart.

    A join means a fetch, and a fetch can fail: `OutcomesUnavailable` rather than whatever
    `nflreadpy` raised, so the CLI can degrade on it without also swallowing every unrelated
    error underneath. A model the store holds nothing for returns before the fetch happens.

    The outcome comes from `hub.models.margin.home_won`, which is the repo's one answer to
    what a tied game means: a tie is neither a home win nor an away win, so it is not scored.
    This used to restate that test inline while *citing* `margin.DROP_TIES` in this
    paragraph, and `hub.publish._scored` took the other convention -- so the two disagreed on
    tied games and only those, with the disagreeing one feeding the public record. Issue #64.
    """
    from hub import store
    # One row per game. `_paired` joins on (game_id, season, week), so two versions on each
    # side is a fourfold cross product -- and the comparison silently becomes mostly a model
    # against itself. A fresh clone has no `preds` view at all; `_nothing_predicted` says
    # what returning nothing means. Returning it before the fetch is deliberate: a model
    # with nothing stored must not cost a network call to say so, and a fresh clone has to
    # reach the NoOverlap message even with nflverse down.
    got = store.predictions(model=model, base=base)
    if got.is_empty():
        return _nothing_predicted()
    preds = got.select("game_id", "season", "week", "model", "home_win_prob")
    if schedules is None:
        try:
            schedules = _schedules()
        except Exception as e:
            # Broad, and deliberately so, on `hub.publish._scored`'s reasoning: this is a
            # network call inside a command that runs unattended, and the module rule is
            # that it produces a usable answer with no operator. Like `_scored` it names
            # the failure *type* rather than presenting every cause as one, and it chains
            # the original so the traceback is still there for whoever goes looking.
            # Issue #63: closing #59 added this call and `main` was not widened, so an
            # unreachable nflverse reached the operator as a traceback from a command that
            # previously could not fail that way.
            raise OutcomesUnavailable(
                f"could not fetch schedules ({type(e).__name__}); {model} has no "
                f"outcomes to score against") from e
    if "result" not in schedules.columns:
        raise OutcomesUnavailable("schedules is missing `result`, the realised margin")
    won = home_won(schedules.select("game_id", pl.col("result").cast(pl.Float64)))
    return preds.join(won.select("game_id", "home_won"), on="game_id", how="inner")


def _labels(a: pl.DataFrame, b: pl.DataFrame,
            given: tuple[str, str] | None) -> tuple[str, str]:
    """What to call the two frames in an error message.

    Read off the `model` column, so `compare` called directly on two frames still names them.
    `given` covers the case the CLI hits most often -- a model absent from the store reads
    back as an empty frame, and an empty frame has no row to carry its own name.
    """
    def one(df: pl.DataFrame, fallback: str) -> str:
        if "model" not in df.columns:
            return fallback
        got = df["model"].unique().to_list()
        return str(got[0]) if len(got) == 1 and got[0] else fallback

    return given if given else (one(a, "the first model"), one(b, "the second model"))


def _paired(a: pl.DataFrame, b: pl.DataFrame, na: str, nb: str) -> pl.DataFrame:
    """Games both models predicted, joined side by side.

    Comparing on the union would make the result partly a comparison of which games each
    model chose to touch, and easy games are not evenly spread.
    """
    keys = [c for c in ("game_id", "season", "week") if c in a.columns and c in b.columns]
    j = (a.select([*keys, "home_win_prob", "home_won"])
          .join(b.select([*keys, "home_win_prob"]), on=keys, how="inner",
                suffix="_b"))
    if j.height == 0:
        # Overlap, and only overlap. This message used to double as the report for "no
        # outcomes joined yet", which was the #59 defect wearing the right exception's name.
        raise NoOverlap(f"{na} and {nb} share no games in common")
    return j


def _holdout_weeks(weeks: Sequence[int], holdout: float) -> list[int]:
    uniq = sorted({int(w) for w in weeks})
    n = max(1, round(len(uniq) * holdout))
    return uniq[-n:]


def compare(a: pl.DataFrame, b: pl.DataFrame, split: str = "temporal",
            holdout: float = DEFAULT_HOLDOUT, bootstrap: int = BOOTSTRAP,
            seed: int = 0, names: tuple[str, str] | None = None) -> dict:
    """Log-loss delta between two models, with a bootstrap interval.

    Negative delta means `a` is better. The interval is bootstrapped over *paired* games --
    the same games for both models on every resample -- so it measures the difference rather
    than the sum of two models' sampling noise, which is the same common-random-numbers
    argument the draft optimizer uses.
    """
    na, nb = _labels(a, b, names)
    j = _paired(a, b, na, nb)
    weeks: list[int] = []
    if split == "temporal" and "week" in j.columns:
        weeks = _holdout_weeks(j["week"].to_list(), holdout)
        j = j.filter(pl.col("week").is_in(weeks))
        if j.height == 0:
            raise NoOverlap(f"{na} and {nb} share no games in the holdout window")

    pa = j["home_win_prob"].to_numpy()
    pb = j["home_win_prob_b"].to_numpy()
    y = j["home_won"].to_numpy().astype(int)
    la, lb = log_loss(pa, y), log_loss(pb, y)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(bootstrap, len(y)))
    deltas = np.array([log_loss(pa[i], y[i]) - log_loss(pb[i], y[i]) for i in idx])

    scored = pl.DataFrame({"home_win_prob": pa, "home_won": y})
    scored_b = pl.DataFrame({"home_win_prob": pb, "home_won": y})
    return {
        "delta": la - lb, "log_loss_a": la, "log_loss_b": lb,
        "brier_a": brier(pa, y), "brier_b": brier(pb, y),
        "ci95": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "n_scored": j.height, "holdout_weeks": weeks, "split": split,
        "reliability_a": reliability(scored), "reliability_b": reliability(scored_b),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.eval", description="Compare two models on held-out games.")
    ap.add_argument("--compare", required=True, help="two model names, comma separated")
    ap.add_argument("--split", default="temporal", choices=("temporal", "all"))
    ap.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    ap.add_argument("--store", default=None,
                    help="processed-store root; defaults to this repo's. Overridable "
                         "so the CLI can be driven against an empty or backup store")
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.compare.split(",")]
    if len(names) != 2:
        print("hub.models.eval: --compare takes exactly two model names", file=sys.stderr)
        return 2
    try:
        base = Path(args.store) if args.store else None
        # The names go in explicitly: a model with nothing in the store reads back empty, and
        # that is the frame whose error message a first run is most likely to see.
        got = compare(load_predictions(names[0], base), load_predictions(names[1], base),
                      split=args.split, holdout=args.holdout, names=(names[0], names[1]))
    except (NoOverlap, OutcomesUnavailable) as e:
        # Both exit the same way; the sentence is what differs, and `_paired`, the fetch and
        # the missing column each write their own. This wrapped `NoOverlap` alone until #63,
        # which is why the network call #59 introduced tracebacked -- an unreachable source,
        # schedules without a realised margin, and two models sharing no games are three
        # different fixes and the reader gets only this line to tell them apart.
        print(f"hub.models.eval: {e}", file=sys.stderr)
        return 1

    w = got["holdout_weeks"]
    print(f"  {names[0]} vs {names[1]} on {got['n_scored']} games"
          + (f", weeks {w[0]}-{w[-1]} held out" if w else " (all games)"))
    print(f"    log loss  {names[0]:<20} {got['log_loss_a']:.4f}")
    print(f"    log loss  {names[1]:<20} {got['log_loss_b']:.4f}")
    print(f"    brier     {names[0]:<20} {got['brier_a']:.4f}")
    print(f"    brier     {names[1]:<20} {got['brier_b']:.4f}")
    lo, hi = got["ci95"]
    verdict = ("no detectable difference" if lo <= 0.0 <= hi else
               (f"{names[0]} better" if got["delta"] < 0 else f"{names[1]} better"))
    print(f"    delta     {got['delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print("    negative delta favours the first model; an interval containing zero is")
    print("    the honest answer when there is nothing to choose between them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
