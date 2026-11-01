"""The interim leaderboard: the weekly, descriptive read of the LLM-forecaster study (#453).

`docs/prereg/llm-forecasters.md`, *Reporting and deviations*: "A scheduled job scores the
finished week and updates the leaderboard page, labeled 'Interim: descriptive, not a test.'"
The confirmatory H1 and H2 verdicts run once, after Week 18, so this file carries **estimates
and intervals only** -- no p-value, no verdict sentence -- and says `interim: true`.

It is built from `hub.forecasters.score`'s seams (`require_pinned`, `load_burn_in` and
`h1_family`, the same pinned-settings and burn-in refusals, `games_from_schedule`, `describe`),
so the figures are the ones the final analysis will report, not a second implementation. That
module is not edited here: the H2 pin names its commit.

**What is published.** Per contestant: the Delta log loss, Delta Brier and mean absolute gap
from the betting market (each with n and a 95% interval, weekly means as in `score`), its
missing count, and the game ids it missed. The week-level `missing_games` list names every
scorable game some contestant has no forecast for. Market snapshots appear as their ids,
which end in a prefix of the payload's SHA-256; the raw Odds API payload is never read here,
and the output is assembled from named fields only, so nothing else in a record can leak into
it (the prereg's *Reporting*: raw odds are published only as a hash).

**Refusals** (exit codes): 2 an input that cannot be read; 3 the analysis is not pinned; 4 the
burn-in estimability record (#451) is absent or does not fit the pin; 5 no scored week has
finished yet -- the first is the week after the burn-in (Week 10). The workflow
(`forecast-score.yml`) treats 3, 4 and 5 as a clean "nothing to publish yet", not a failure.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.forecasters import config as study_config
from hub.forecasters import score
from hub.forecasters.config import Analysis, StudyConfig
from hub.forecasters.packet import Condition
from hub.forecasters.runner import NotPinned

LABEL = "Interim: descriptive, not a test."
OK, UNREADABLE, NOT_PINNED, NO_BURN_IN, NO_SCORED_WEEK = 0, 2, 3, 4, 5


def _interval(w: score.WeeklyMean) -> dict[str, Any]:
    return {"estimate": score._num(w.estimate), "lo": score._num(w.lo), "hi": score._num(w.hi),
            "weeks": w.weeks, "games": w.games}


def build(games: Sequence[score.Game], forecasts: score.Forecasts, *, analysis: Analysis,
          burn_in: score.BurnIn, snapshot_ids: Sequence[str], refused: int,
          seed: int) -> dict[str, Any]:
    """The leaderboard document for `games` (the scored window) against `forecasts`."""
    played = [g for g in games if g.outcome is not None]
    rows: list[dict[str, Any]] = []
    missing_by_game: dict[str, list[str]] = defaultdict(list)
    for c in sorted(forecasts):
        s = score.describe(games, forecasts, c, seed=seed)
        missed = sorted(g.game_id for g in played if g.game_id not in forecasts[c])
        for gid in missed:
            missing_by_game[gid].append(c)
        rows.append({"contestant": c, "log_loss_diff": _interval(s.log_loss),
                     "brier_diff": _interval(s.brier), "abs_gap": _interval(s.abs_gap),
                     "missing": s.missing, "scorable": s.scorable, "missing_games": missed})
    week = {g.game_id: g.week for g in played}
    return {
        "interim": True,
        "label": LABEL,
        "season": burn_in.season,
        "first_scored_week": burn_in.first_scored_week,
        "through_week": max((g.week for g in played), default=None),
        "scored_weeks": sorted({g.week for g in played}),
        "h2_implementation": analysis.h2_implementation,
        "contestants": rows,
        "missing_games": [{"game_id": gid, "week": week[gid], "contestants": cs}
                          for gid, cs in sorted(missing_by_game.items())],
        "market_snapshot_ids": sorted(set(snapshot_ids)),
        "packet_refused": refused,
    }


def _snapshot_ids(out_dir: Path) -> list[str]:
    return [str(r["market_snapshot_id"]) for r in score._records(out_dir)
            if r.get("market_snapshot_id")]


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print,
         config: StudyConfig | None = None, analysis: Analysis | None = None,
         schedule: Callable[[int], pl.DataFrame] = score.cached_schedule) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.leaderboard")
    ap.add_argument("--forecasts", type=Path, required=True, help="the forecasts checkout")
    ap.add_argument("--out", type=Path, required=True, help="where to write the leaderboard JSON")
    ap.add_argument("--burn-in", type=Path, default=None,
                    help=f"the burn-in estimability record (default: {score.BURN_IN.name})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = config or study_config.load()
    an = analysis or cfg.analysis
    try:
        score.require_pinned(an, cfg)
    except NotPinned as exc:
        echo(f"refused: {exc}")
        return NOT_PINNED
    if not args.forecasts.is_dir():
        echo(f"refused: cannot read {args.forecasts}")
        return UNREADABLE
    try:
        burn_in = score.load_burn_in(args.burn_in)
        score.h1_family(burn_in, [f"{m.name}:{Condition.LINE_AWARE}" for m in cfg.llms],
                        float(an.h1_min_deviation_sd or 0.0))
    except score.BurnInRefused as exc:
        echo(f"refused: {exc}")
        return NO_BURN_IN
    markets = score.load_markets(args.forecasts)
    refused = score.refused_games(args.forecasts)
    games = score.games_from_schedule(schedule(burn_in.season), markets.home,
                                      season=burn_in.season, first_week=burn_in.first_scored_week,
                                      refused=refused)
    if not any(g.outcome is not None for g in games):
        echo(f"refused: no scored week has finished yet (the first is Week "
             f"{burn_in.first_scored_week})")
        return NO_SCORED_WEEK
    doc = build(games, score.load_forecasts(args.forecasts), analysis=an, burn_in=burn_in,
                snapshot_ids=_snapshot_ids(args.forecasts), refused=len(refused), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(args.out, json.dumps(doc, sort_keys=True, indent=1) + "\n")
    echo(f"wrote {args.out}: interim leaderboard through Week {doc['through_week']}, "
         f"{len(doc['contestants'])} contestants")
    return OK


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
