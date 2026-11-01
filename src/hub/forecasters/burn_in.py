"""The burn-in report: which line-aware models can enter H1, and what power H2 has (#451).

`docs/prereg/llm-forecasters.md`, *Burn-in* and *Before v1*. After the burn-in week (Week 9)
this reads that week's forecast records -- forecasts and the forecast-time market only, never
an outcome -- and does two things:

1. **Writes the estimability record** `docs/prereg/llm-forecasters.burn-in.json`, in the
   schema `hub.forecasters.score.load_burn_in` reads: `{season, burn_in_week,
   h1_min_deviation_sd, deviation_sd: {contestant: sd}}`. `deviation_sd` is the sample SD of
   each line-aware LLM's log-odds deviations from the de-vigged market on the burn-in week;
   the threshold is the pinned `[analysis].h1_min_deviation_sd`, read from
   `conf/forecasters.toml`, never an argument. The record is a `prereg-llm-v1` artifact: an
   existing file with different content is refused, never replaced, and the same content is
   left untouched. It is written atomically.
2. **Prints the burn-in report**: each model's SD against the threshold (estimable, or "not
   estimable (copies the betting market)"), and H2's power at the pinned delta for the
   remaining scored weeks. The power is a **simulation**, labelled as one: games draw their
   market probabilities from the burn-in week's, a model's log-odds deviation is Normal with
   its own burn-in SD, outcomes follow the betting market, and each simulated season is
   analysed with `score.noninferiority` on weekly means. Two true-Delta scenarios are shown:
   the model's deviations are noise (its expected cost against the betting market), and
   exact parity.

**Refusal.** One sentence, nothing written, when the burn-in week's records are missing or
incomplete: every game of the week (from nflverse's schedule, less those the runner logged as
refused) needs a scorable record and a market probability for every line-aware LLM.

Exit codes: 0 ok, 2 an input that cannot be read, 3 the analysis is not pinned, 4 the week's
records are missing or incomplete, 5 a record already exists with different content; 1 the
schedule cannot be read (`hub.cli.unavailable`).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from hub import atomic, cli
from hub.forecasters import config as study_config
from hub.forecasters import score
from hub.forecasters.config import Analysis, StudyConfig
from hub.forecasters.packet import Condition
from hub.forecasters.runner import NotPinned

OK, UNREADABLE, NOT_PINNED, INCOMPLETE, EXISTS = 0, 2, 3, 4, 5
SEASON = 2026
BURN_IN_WEEK = 9
SIMULATIONS = 2000
SIM_SEED = 0
PRECISION = 6                                      # digits kept in the record's SDs


class Refused(RuntimeError):
    """The burn-in cannot be reported; the message is the one sentence."""


class RecordExists(Refused):
    """A different burn-in record is already committed."""


def aware_names(config: StudyConfig) -> list[str]:
    return [f"{m.name}:{Condition.LINE_AWARE}" for m in config.llms]


def burn_in_games(frame: pl.DataFrame, *, season: int, week: int,
                  refused: set[str]) -> list[str]:
    """The regular-season game ids of the burn-in week, less those the runner refused."""
    return sorted(
        str(r["game_id"]) for r in frame.select("game_id", "season", "game_type", "week")
        .iter_rows(named=True)
        if r["game_type"] == "REG" and int(r["season"]) == season and int(r["week"]) == week
        and r["game_id"] not in refused)


def deviations(forecasts: score.Forecasts, markets: Mapping[str, float], contestant: str,
               game_ids: Sequence[str]) -> npt.NDArray[np.float64]:
    """Log-odds deviation of the (clipped) forecast from the de-vigged market, per game."""
    p = score.clip([forecasts[contestant][g] for g in game_ids])
    m = np.array([markets[g] for g in game_ids], dtype=float)
    return score.logit(p) - score.logit(m)


def deviation_sds(forecasts: score.Forecasts, markets: Mapping[str, float],
                  contestants: Sequence[str], game_ids: Sequence[str], week: int
                  ) -> dict[str, float]:
    """Each contestant's deviation SD on the week, or `Refused` in one sentence if the week
    is empty or a contestant lacks a forecast, or a market, on any of its games."""
    if len(game_ids) < 2:
        raise Refused(f"refused: Week {week} has {len(game_ids)} scheduled games, too few "
                      f"to measure a deviation SD")
    for c in contestants:
        mine = forecasts.get(c, {})
        absent = [g for g in game_ids if g not in mine or g not in markets]
        if absent:
            raise Refused(f"refused: the Week {week} records are incomplete, {c} has no "
                          f"scorable forecast and market for {len(absent)} of {len(game_ids)} "
                          f"games (first: {absent[0]}).")
    return {c: round(float(np.std(deviations(forecasts, markets, c, game_ids), ddof=1)),
                     PRECISION) for c in contestants}


def record(*, season: int, week: int, threshold: float, sds: Mapping[str, float]) -> dict[
        str, object]:
    return {"season": season, "burn_in_week": week, "h1_min_deviation_sd": threshold,
            "deviation_sd": dict(sorted(sds.items()))}


def render(rec: Mapping[str, object]) -> str:
    return json.dumps(rec, sort_keys=True, indent=1) + "\n"


def write_record(path: Path, rec: Mapping[str, object]) -> bool:
    """Write the v1 artifact atomically. True if written, False if the file already holds
    exactly this content; `Refused` if it holds anything else."""
    if path.exists():
        try:
            same = json.loads(path.read_text()) == json.loads(render(rec))
        except (OSError, ValueError):
            same = False
        if same:
            return False
        raise RecordExists(f"refused: {path} already exists with different content; it is a "
                      f"prereg-llm-v1 artifact and is never overwritten.")
    atomic.write_text(path, render(rec))
    return True


# --- H2's power, by simulation ------------------------------------------------------------

def simulate_power(markets: Sequence[float], sds: Mapping[str, float], *, delta: float,
                   games_per_week: int, weeks: int, simulations: int = SIMULATIONS,
                   seed: int = SIM_SEED) -> dict[str, dict[str, float]]:
    """P(reject Delta <= -delta) per model, and for the intersection-union claim (all
    reject), over `simulations` simulated scored windows of `weeks` weeks.

    Market probabilities are drawn from `markets` (the burn-in week's); a model's forecast is
    the betting market's log-odds plus Normal(0, its SD) noise, clipped as the study clips;
    outcomes follow the betting market. Scenario "noise": the true Delta is that noise's
    expected cost against the betting market. Scenario "parity": the same draws re-centred to
    a true Delta of exactly 0.
    Each simulated window is tested by `score.noninferiority` on its weekly means."""
    rng = np.random.default_rng(seed)
    m = np.asarray(markets, dtype=float)
    shape = (simulations, weeks, games_per_week)
    mk = rng.choice(m, size=shape)
    y = (rng.random(shape) < mk).astype(float)
    base = score.log_loss_terms(mk, y)
    out: dict[str, dict[str, float]] = {}
    passes: dict[str, dict[str, npt.NDArray[np.bool_]]] = {"noise": {}, "parity": {}}
    for name, sd in sds.items():
        p = score.clip(1.0 / (1.0 + np.exp(-(score.logit(mk) + rng.normal(0, sd, shape)))))
        diff = (score.log_loss_terms(p, y) - base).mean(axis=2)          # (sim, week)
        for scenario, weekly in (("noise", diff), ("parity", diff - diff.mean())):
            passes[scenario][name] = _rejects(weekly, delta)
        out[name] = {s: float(passes[s][name].mean()) for s in passes}
    out["all"] = {s: float(np.logical_and.reduce(list(passes[s].values())).mean())
                  for s in passes} if sds else {"noise": 0.0, "parity": 0.0}
    return out


def _rejects(weekly: npt.NDArray[np.float64], delta: float) -> npt.NDArray[np.bool_]:
    weeks = weekly.shape[1]
    est = weekly.mean(axis=1)
    se = weekly.std(axis=1, ddof=1) / math.sqrt(weeks)
    nan = float("nan")
    res = []
    for e, s in zip(est, se, strict=True):
        wm = score.WeeklyMean(float(e), float(s), nan, nan, weeks, 0, nan, nan)
        scored = score.Scored("sim", wm, wm, wm, 0, 0)
        res.append(score.noninferiority(scored, delta).noninferior)
    return np.array(res, dtype=bool)


# --- the report ---------------------------------------------------------------------------

def report_lines(rec: Mapping[str, object], power: Mapping[str, Mapping[str, float]], *,
                 delta: float, weeks: int, games_per_week: int, simulations: int,
                 written: bool, path: Path) -> list[str]:
    threshold = float(str(rec["h1_min_deviation_sd"]))
    sds = rec["deviation_sd"]
    assert isinstance(sds, dict)
    lines = [f"{'wrote' if written else 'already holds this exact record:'} {path}",
             f"Burn-in report, Week {rec['burn_in_week']} of {rec['season']} "
             f"(forecasts only; no outcome is read)",
             f"H1 estimability: SD of log-odds deviations from the betting market vs the threshold "
             f"{threshold:g}"]
    for name, sd in sds.items():
        verdict = "estimable" if sd >= threshold else score.NOT_ESTIMABLE
        lines.append(f"  {name}: SD {sd:.4f}  {'>=' if sd >= threshold else '<'} "
                     f"{threshold:g}  {verdict}")
    n_in = sum(1 for sd in sds.values() if sd >= threshold)
    lines.append(f"  H1 family fixed at {n_in} of {len(sds)} line-aware models")
    lines.append(f"H2 power at delta {delta:g} nats/game: SIMULATION, not a measurement and "
                 f"not pre-registered ({simulations} simulated windows of {weeks} weeks x "
                 f"{games_per_week} games; deviations ~ Normal(0, burn-in SD), outcomes follow the betting market)")
    for name in sds:
        p = power[name]
        lines.append(f"  {name}: power {p['noise']:.2f} if its deviations are noise, "
                     f"{p['parity']:.2f} at exact parity")
    lines.append(f"  all {len(sds)} reject (the H2 claim): {power['all']['noise']:.2f} noise, "
                 f"{power['all']['parity']:.2f} parity")
    return lines


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print,
         config: StudyConfig | None = None, analysis: Analysis | None = None,
         schedule: Callable[[int], pl.DataFrame] = score.cached_schedule,
         simulations: int = SIMULATIONS) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.burn_in")
    ap.add_argument("--forecasts", type=Path, required=True, help="the forecasts checkout")
    ap.add_argument("--week", type=int, default=BURN_IN_WEEK, help="the burn-in week")
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--out", type=Path, default=score.BURN_IN,
                    help=f"the record to write (default: {score.BURN_IN.name})")
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
    threshold, delta = an.h1_min_deviation_sd, an.delta
    assert threshold is not None and delta is not None           # require_pinned held
    try:
        frame = schedule(args.season)
    except Exception as exc:                 # the nflverse cache or wire: a sentence, exit 1
        return cli.unavailable("hub.forecasters.burn_in", f"the {args.season} schedule", exc)
    try:
        games = burn_in_games(frame, season=args.season, week=args.week,
                              refused=score.refused_games(args.forecasts))
        markets = score.load_markets(args.forecasts).home
        sds = deviation_sds(score.load_forecasts(args.forecasts), markets, aware_names(cfg),
                            games, args.week)
        rec = record(season=args.season, week=args.week, threshold=threshold, sds=sds)
        written = write_record(args.out, rec)
        power = simulate_power([markets[g] for g in games], sds, delta=delta,
                               games_per_week=len(games), weeks=score.LAST_WEEK - args.week,
                               simulations=simulations)
    except Refused as exc:
        echo(str(exc))
        return EXISTS if isinstance(exc, RecordExists) else INCOMPLETE
    for line in report_lines(rec, power, delta=delta, weeks=score.LAST_WEEK - args.week,
                             games_per_week=len(games), simulations=simulations,
                             written=written, path=args.out):
        echo(line)
    return OK


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
