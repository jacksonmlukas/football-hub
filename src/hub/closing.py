"""The close of each game, with the lead time it was taken at, and the value of closing (#495).

**What "the close" means here, and what it does not.** A game's close is the last archived
capture at or before its kickoff. Whether that capture is *close to* kickoff is a separate
fact, and this module never lets the first pass for the second: every close row carries
`lead_minutes` (kickoff minus capture) and `is_true_close`, true only when the lead is within
`TRUE_CLOSE_LEAD_MIN`. A capture taken six hours early is returned as the game's last
pre-kickoff line with `lead_minutes` 360 and `is_true_close` false -- it is not "the close"
and no report may call it that. A capture stamped after kickoff is in-play, so it is never a
close at all (the scheduled polls land hours late, #468, and the Sunday one falls after the
early kickoffs).

**Capturing it.** The scheduled polls cannot do this: GitHub starts them 4-8.5h late. The game
windows already run a long loop (`live.yml`), so `capture` is called on every cycle of one and
spends a call only when a kickoff is within `CLOSE_WINDOW_MIN` minutes with no capture inside
that window yet. One call returns the whole season's lines, so it is **one call per distinct
kickoff time**, not per game: a slate of 1pm games is one call. Cost per call is
`len(odds.MARKETS) * len(odds.REGIONS)`; at three markets (spreads, totals, h2h) and at most
`CLOSE_CALLS_PER_WEEK` calls that is at most 30 credits a week; a normal week has six or
seven distinct kickoff times (Thu, Sun 1pm, 4:05, SNF, MNF, sometimes 9:30 and 4:25) and so
18-21. The credit floor
`odds.CREDIT_FLOOR` and the weekly cap both refuse before any request is formed.

**Closing-line value is descriptive.** `clv` compares the probability a prediction published
with the probability the close implies, on the side the prediction picked. It is a published
figure with its n and an interval and no verdict word; nothing pre-registered rests on it.

    uv run python -m hub.closing --report        # reads the archive, spends nothing
    uv run python -m hub.closing --capture       # one gated call if a kickoff is imminent
    uv run python -m hub.closing --due           # exit 0 if one is due; spends nothing
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic, store
from hub.config import SEASON_AHEAD
from hub.declare import not_an_input
from hub.fetch import odds
from hub.models.market import MARGIN_SD, normal_cdf
from hub.paths import ROOT, STATE_DIR

# Fire a capture when a kickoff is this close. Sized to the loop's cadence (a cycle every five
# minutes, so a window of fifteen gives at least two chances before the kickoff) and small
# enough that the lead is a close, not a morning line.
CLOSE_WINDOW_MIN = not_an_input(
    15, "decides when a capture is taken and archived; it changes no prediction, forecast or model")

# A capture is called a true close only within this many minutes of kickoff. Wider than the
# window so a cycle missed by one loop beat still counts, and no wider: past it the row says
# its lead and is not the close.
TRUE_CLOSE_LEAD_MIN = not_an_input(
    30, "only labels an archived capture as a close or not; no prediction reads it")

# The most close-window calls in any seven days. A normal NFL week has six or seven distinct
# kickoff times; ten is the cap with slack for an international game or a double-header.
CLOSE_CALLS_PER_WEEK = not_an_input(
    10, "a spending ceiling on the odds poller; it changes no prediction or model")


class CloseBudgetExceeded(Exception):
    """The weekly close-window call cap is spent; refused before any request is formed."""


# --- the archive side -------------------------------------------------------------------

_CLOSE_COLUMNS = {"game_id": pl.Utf8, "kickoff": pl.Datetime, "captured_at": pl.Datetime,
                  "close_spread": pl.Float64, "lead_minutes": pl.Float64,
                  "is_true_close": pl.Boolean}


def closes(archive: pl.DataFrame, kickoffs: pl.DataFrame) -> pl.DataFrame:
    """Each game's last capture at or before kickoff, with how long before kickoff it was.

    `archive` carries `game_id`, `captured_at`, `close_spread` (and optionally `close_total`,
    `ml_home_prob`); `kickoffs` carries `game_id`, `kickoff` (naive UTC, both). A game with no
    kickoff, or no capture at or before it, is absent: it has no close, and a capture after
    kickoff is never offered in its place. `lead_minutes` is exact, not rounded.
    """
    if archive.is_empty() or kickoffs.is_empty():
        return pl.DataFrame(schema=_CLOSE_COLUMNS)
    extra = [c for c in ("close_total", "ml_home_prob") if c in archive.columns]
    pre = (archive.select("game_id", "captured_at", "close_spread", *extra)
           .join(kickoffs.select("game_id", "kickoff").drop_nulls("kickoff"), on="game_id")
           .filter(pl.col("captured_at") <= pl.col("kickoff"))
           .drop_nulls("close_spread"))
    last = pre.sort("game_id", "captured_at").group_by("game_id", maintain_order=True).last()
    return last.with_columns(
        ((pl.col("kickoff") - pl.col("captured_at")).dt.total_seconds() / 60.0)
        .alias("lead_minutes")
    ).with_columns(
        (pl.col("lead_minutes") <= TRUE_CLOSE_LEAD_MIN).alias("is_true_close"))


def due(archive: pl.DataFrame, kickoffs: pl.DataFrame, now: datetime,
        attempts: Sequence[datetime] = ()) -> pl.DataFrame:
    """Games whose kickoff is within the window, still ahead, with nothing spent inside it yet.

    Strictly ahead: at or after kickoff nothing is due, because a line fetched then is not a
    pre-game line. A window is **covered once any capture of any game, or any recorded attempt,
    falls inside it** -- not only a row for that game. One call returns the whole season, so a
    game missing from the response (no line posted, or unmatched to a nflverse game) would
    otherwise stay due and re-spend every cycle; the window spends at most once. `attempts` are
    the moments a call was started (`record_attempt`), kept so a call whose capture never landed
    -- a lost push, a response that matched nothing -- still counts.
    """
    soon = (kickoffs.drop_nulls("kickoff")
            .filter((pl.col("kickoff") > now)
                    & (pl.col("kickoff") <= now + timedelta(minutes=CLOSE_WINDOW_MIN))))
    if soon.is_empty():
        return soon.select("game_id", "kickoff")
    stamps = list(attempts)
    if not archive.is_empty():
        stamps += archive["captured_at"].unique().to_list()
    window = timedelta(minutes=CLOSE_WINDOW_MIN)
    keep = [any(k - window <= t <= k for t in stamps) is False
            for k in soon["kickoff"].to_list()]
    return soon.filter(pl.Series(keep, dtype=pl.Boolean)).select("game_id", "kickoff")


def calls_in_week(archive: pl.DataFrame, kickoffs: pl.DataFrame, now: datetime,
                  attempts: Sequence[datetime] = ()) -> int:
    """Calls spent in the last seven days on close windows: attempts, plus unmatched captures.

    Every recorded attempt counts, whether or not a capture followed it. A capture taken inside
    some game's window counts too unless an attempt shortly precedes it (`_SAME_CALL_MIN`) --
    that is the attempt's own capture, and counting both would charge one call twice. A
    scheduled poll that happened to land in a window is such a capture: it spent the credits.
    """
    week_ago = now - timedelta(days=7)
    tried = sorted(t for t in attempts if t > week_ago)
    caps: list[datetime] = []
    if not archive.is_empty() and not kickoffs.is_empty():
        recent = archive.filter(pl.col("captured_at") > week_ago)
        inside = (recent.select("game_id", "captured_at")
                  .join(kickoffs.select("game_id", "kickoff").drop_nulls("kickoff"),
                        on="game_id")
                  .filter((pl.col("captured_at") <= pl.col("kickoff"))
                          & (pl.col("captured_at")
                             >= pl.col("kickoff") - timedelta(minutes=TRUE_CLOSE_LEAD_MIN))))
        caps = inside["captured_at"].unique().to_list() if inside.height else []
    gap = timedelta(minutes=_SAME_CALL_MIN)
    extra = [c for c in caps if not any(t <= c <= t + gap for t in tried)]
    return len(tried) + len(extra)


# --- the attempt record -----------------------------------------------------------------
#
# Written and committed BEFORE the request (`capture`), so a call whose capture is lost (the
# push fails, the response matches nothing) is still counted against the window and the weekly
# cap. Without it the next cycle pays again for a window it already paid for.

ATTEMPTS_FILE = STATE_DIR / "odds_close_attempts.json"
_SAME_CALL_MIN = not_an_input(
    10, "how far after an attempt its own capture may land; accounting only, no prediction")


class AttemptNotRecorded(Exception):
    """The attempt marker could not be committed, so no request was made."""


def read_attempts(path: Path | None = None) -> list[datetime]:
    try:
        doc = json.loads(Path(path or ATTEMPTS_FILE).read_text())
        return [datetime.fromisoformat(t) for t in doc.get("attempts", [])]
    except Exception:
        return []


def record_attempt(when: datetime, path: Path | None = None) -> None:
    keep = [t for t in read_attempts(path) if t > when - timedelta(days=30)]
    doc = {"attempts": [t.isoformat() for t in [*keep, when]]}
    atomic.write_text(Path(path or ATTEMPTS_FILE), json.dumps(doc, indent=1) + "\n")


def kickoff_table(season: int) -> pl.DataFrame:
    """`game_id`, `kickoff` for the season, from the schedule (a network read, no credits)."""
    from hub import schedule
    from hub.fetch import nflverse
    # Not `odds._schedule`, which refreshes: a close loop asks every few minutes and kickoff
    # times do not move between cycles, so the runner's cached read is the right one.
    sched = nflverse.load("schedules", [season]).filter(pl.col("season") == season)
    if not {"gameday", "gametime"} <= set(sched.columns):
        return pl.DataFrame(schema={"game_id": pl.Utf8, "kickoff": pl.Datetime})
    return sched.select("game_id", schedule.kickoff_expr())


def capture(season: int = SEASON_AHEAD, *, now: datetime | None = None,
            kickoffs: pl.DataFrame | None = None, archive: pl.DataFrame | None = None,
            state_path: Path | None = None, base: Path | None = None,
            floor: int = odds.CREDIT_FLOOR,
            fetch: Callable[..., Any] | None = None,
            attempts_path: Path | None = None,
            publish_attempt: Callable[[], None] | None = None) -> int:
    """One gated close capture. Returns the number of calls made: 0 or 1.

    Spends nothing unless a game is `due`. The weekly cap and the credit floor are checked
    before the request is formed; both raise rather than skip, so the loop's log says why.
    **The attempt is recorded, and published if `publish_attempt` is given, before the request**:
    if it cannot be published no request is made (`AttemptNotRecorded`), and if the capture is
    lost afterwards the window is still marked spent. `fetch` is `odds.snapshot` unless a test
    substitutes the reach.
    """
    when = now or datetime.now(UTC).replace(tzinfo=None)
    kicks = kickoffs if kickoffs is not None else kickoff_table(season)
    arc = archive if archive is not None else store.lines(season, base=base)
    tried = read_attempts(attempts_path)
    pending = due(arc, kicks, when, tried)
    if pending.is_empty():
        print("  close capture: no kickoff inside the window without a capture; nothing spent")
        return 0
    # GUARD close-calls-capped-per-week [unit/test_closing.py]: a runaway loop cannot drain the
    # month through this door; the credit floor below is the second, independent stop
    used = calls_in_week(arc, kicks, when, tried)
    if used >= CLOSE_CALLS_PER_WEEK:
        raise CloseBudgetExceeded(
            f"{used} close-window calls in the last 7 days, cap is {CLOSE_CALLS_PER_WEEK}. "
            f"Refusing before a request is formed.")
    # /GUARD
    have = odds.known_balance(state_path, when, floor=floor)
    if have is not None and have < floor:
        raise odds.QuotaFloor(f"{have} credits left, floor is {floor}. Refusing before "
                              f"recording an attempt or spending one.")
    cost = len(odds.MARKETS) * len(odds.REGIONS)
    print(f"  close capture: {pending.height} game(s) kick off within {CLOSE_WINDOW_MIN} min; "
          f"one call, {cost} credits ({used + 1} of {CLOSE_CALLS_PER_WEEK} this week)")
    # GUARD attempt-recorded-before-the-request [unit/test_closing.py]: the spend is on the
    # record before it happens, so a lost capture cannot be paid for twice
    record_attempt(when, attempts_path)
    if publish_attempt is not None:
        try:
            publish_attempt()
        except Exception as e:
            raise AttemptNotRecorded(
                f"the attempt marker could not be committed ({type(e).__name__}: {e}); "
                f"no request was made") from e
    # /GUARD
    (fetch or odds.snapshot)(season, state_path=state_path, base=base, floor=floor, now=when)
    return 1


def is_due(season: int = SEASON_AHEAD, *, now: datetime | None = None,
           base: Path | None = None) -> bool:
    """Whether a close capture is due now. Reads the archive and schedule; spends nothing."""
    when = now or datetime.now(UTC).replace(tzinfo=None)
    return not due(store.lines(season, base=base), kickoff_table(season), when,
                   read_attempts()).is_empty()


# --- closing-line value -----------------------------------------------------------------

def _implied_home(spread: Any) -> Any:
    """P(home wins) from a home-positive spread: `MarketBaseline`'s own conversion."""
    return pl.Series([normal_cdf(float(s) / MARGIN_SD) for s in spread], dtype=pl.Float64)


def _mean_interval(x: Sequence[float]) -> dict[str, Any]:
    """Mean with a normal-approximation 95% interval; None where n is too small to have one."""
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m = sum(x) / n
    if n < 2:
        return {"n": n, "mean": m, "lo": None, "hi": None}
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": m, "lo": m - half, "hi": m + half}


def _median(col: pl.Series) -> float | None:
    got = col.drop_nulls()
    return float(statistics.median(got.to_list())) if len(got) else None


def _log_loss(p: Sequence[float], y: Sequence[int]) -> float | None:
    if not p:
        return None
    eps = 1e-12
    return -sum(math.log(max(eps, min(1 - eps, q if w else 1 - q)))
                for q, w in zip(p, y, strict=True)) / len(p)


def clv(preds: pl.DataFrame, close: pl.DataFrame) -> dict[str, Any]:
    """Published-time probability against the close's, on the side each prediction picked.

    `preds` carries `game_id`, `home_win_prob`, `predicted_at` (and `home_won` where the game
    has been scored); `close` is `closes()`. The pick is the side the published probability
    favours; a published 0.5 has no side and is counted apart, never given one. A positive
    value means the close moved toward our side by that many probability points.

    **Published probabilities are the spread's normal conversion**, so the primary figure
    converts the close's spread the same way: like for like, no de-vig method in the
    difference. The moneyline close, where archived, is reported beside it as a count and a
    log loss, not mixed into the difference.

    Descriptive. No verdict word, no threshold, no pre-registration: the figure is what it is
    and carries its n and interval. Rows are reported as `n_true_close` (lead within
    `TRUE_CLOSE_LEAD_MIN`) and, apart, as the weaker last-pre-kickoff group with their median
    lead -- the latter is not a close and is labelled so. A prediction stamped after the
    capture it is compared with is dropped: it could not have been published before that line.
    """
    base: dict[str, Any] = {
        "descriptive": True,
        "definition": ("close_prob(pick) - published_prob(pick), in probability points, on the "
                       "side the published probability favours; the close is the last capture "
                       "at or before kickoff, with its lead time stated"),
        "close_probability": "normal conversion of the close spread at the published margin sd",
        "true_close_lead_minutes": TRUE_CLOSE_LEAD_MIN,
    }
    if preds.is_empty() or close.is_empty():
        return {**base, "n": 0, "n_true_close": 0, "n_early_close": 0, "n_no_pick": 0,
                "n_published_after_close": 0, "true_close": _mean_interval([]),
                "early_close": {**_mean_interval([]), "median_lead_minutes": None},
                "log_loss": None, "n_h2h_close": 0}
    p = (preds.sort("predicted_at").group_by("game_id", maintain_order=True).first()
         .join(close, on="game_id"))
    if p.is_empty():
        return clv(preds.clear(), close.clear())
    # Published at or after the capture it is compared with: it could not have preceded it.
    if p.schema["predicted_at"] == pl.Utf8:
        p = p.with_columns(pl.col("predicted_at").str.to_datetime(strict=False))
    after = p.filter((pl.col("predicted_at") >= pl.col("captured_at")).fill_null(False))
    p = p.filter(~(pl.col("predicted_at") >= pl.col("captured_at")).fill_null(False))
    no_pick = p.filter(pl.col("home_win_prob") == 0.5)
    p = p.filter(pl.col("home_win_prob") != 0.5)
    if p.is_empty():
        return {**base, "n": 0, "n_true_close": 0, "n_early_close": 0,
                "n_no_pick": no_pick.height, "n_published_after_close": after.height,
                "true_close": _mean_interval([]),
                "early_close": {**_mean_interval([]), "median_lead_minutes": None},
                "log_loss": None, "n_h2h_close": 0}
    close_home = _implied_home(p["close_spread"])
    pick_home = p["home_win_prob"] > 0.5
    pub_pick = pl.select(pl.when(pick_home).then(p["home_win_prob"])
                         .otherwise(1 - p["home_win_prob"])).to_series()
    close_pick = pl.select(pl.when(pick_home).then(close_home)
                           .otherwise(1 - close_home)).to_series()
    value = (close_pick - pub_pick)
    true_mask = p["is_true_close"]
    early = p.filter(~true_mask)
    out: dict[str, Any] = {
        **base,
        "n": p.height,
        "n_true_close": int(true_mask.sum()),
        "n_early_close": early.height,
        "n_no_pick": no_pick.height,
        "n_published_after_close": after.height,
        "true_close": _mean_interval(value.filter(true_mask).to_list()),
        "early_close": {**_mean_interval(value.filter(~true_mask).to_list()),
                        "median_lead_minutes": _median(early["lead_minutes"])},
    }
    if "home_won" in p.columns:
        won = p.drop_nulls("home_won")
        cw = _implied_home(won["close_spread"]).to_list()
        out["log_loss"] = {
            "n": won.height,
            "published": _log_loss(won["home_win_prob"].to_list(), won["home_won"].to_list()),
            "close": _log_loss(cw, won["home_won"].to_list()),
        }
    else:
        out["log_loss"] = None
    if "ml_home_prob" in p.columns:
        ml = p.drop_nulls("ml_home_prob")
        out["n_h2h_close"] = ml.height
        if "home_won" in ml.columns:
            m = ml.drop_nulls("home_won")
            out["h2h_log_loss"] = {"n": m.height,
                                   "close": _log_loss(m["ml_home_prob"].to_list(),
                                                      m["home_won"].to_list())}
    else:
        out["n_h2h_close"] = 0
    return out


# --- CLI ---------------------------------------------------------------------------------

def report(season: int = SEASON_AHEAD, base: Path | None = None) -> int:
    """Print the close table's shape: how many games have a close, at what lead. No credits."""
    arc = store.lines(season, base=base)
    got = closes(arc, kickoff_table(season))
    if got.is_empty():
        print("  closes: none archived yet")
        return 0
    print(f"  closes: {got.height} games with a last pre-kickoff capture; "
          f"{int(got['is_true_close'].sum())} within {TRUE_CLOSE_LEAD_MIN} min of kickoff; "
          f"median lead {got['lead_minutes'].median():.0f} min")
    return 0


def _commit_marker() -> None:
    """Commit and push `state/` through the shared script; raises if that fails."""
    import subprocess
    subprocess.run([str(ROOT / ".github" / "scripts" / "commit-state.sh"),
                    "odds: close attempt"], check=True, timeout=300)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hub.closing", description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true", help="summarise closes; spends nothing")
    ap.add_argument("--capture", action="store_true",
                    help="one gated odds call if a kickoff is within the window")
    ap.add_argument("--commit-attempt", action="store_true",
                    help="commit the attempt marker before the request (the workflow's mode)")
    ap.add_argument("--due", action="store_true",
                    help="exit 0 if a capture is due, 1 if not, 2 if that cannot be told; "
                         "spends nothing")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    a = ap.parse_args(argv)
    if a.due:
        try:
            return 0 if is_due(a.season) else 1
        except Exception as e:
            print(f"hub.closing: cannot tell whether a capture is due ({type(e).__name__}: {e})",
                  file=sys.stderr)
            return 2
    if a.capture:
        try:
            capture(a.season, publish_attempt=_commit_marker if a.commit_attempt else None)
        except (odds.QuotaFloor, odds.MultiplierRefused, CloseBudgetExceeded,
                odds.SnapshotIncomplete, AttemptNotRecorded) as e:
            print(f"hub.closing: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            from hub.cli import unavailable
            return unavailable("hub.closing", "the betting market's closing prices", e)
        return 0
    return report(a.season)


if __name__ == "__main__":
    sys.exit(main())
