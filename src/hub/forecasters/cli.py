"""The commands `.github/workflows/forecast.yml` runs, in order: `should-run`, `run`, `mark-done`.

`should-run` prints `run=true|false` and `day=YYYY-MM-DD` for `$GITHUB_OUTPUT`. `run` forecasts
the day's games into `--out` (the checked-out `forecasts` branch) through
`hub.forecasters.pipeline`. `mark-done` writes the day's marker once `run` has succeeded; a run
that fails leaves none, so the next trigger in the window retries and the runner's idempotency
fills only the gaps.

Exit codes: 0 ok (including nothing due and nothing left), 2 bad arguments, 3 the config is not
pinned for v0, 4 a source failed with no last-good copy, 5 a provider's key is missing,
6 partial (no market snapshot: the blind half ran, the line-aware half waits for the next
trigger), 7 another run holds the lock, 8 the spending guard refused. Each refusal is one
sentence, and only 6 follows any model call.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

from hub import atomic
from hub.forecasters import config as study_config
from hub.forecasters import days, last_good, pipeline, providers, runner

# The refusals `run` answers with one sentence and an exit code, never a traceback.
REFUSALS: tuple[tuple[type[Exception], int], ...] = (
    (last_good.NoLastGood, 4), (providers.MissingKey, 5), (runner.Locked, 7),
    (runner.BudgetExhausted, 8))


def _marker(out: Path, day: date) -> Path:
    return out / f"{day:%Y}" / f"{day:%Y-%m-%d}.done"


def done_days(out: Path) -> set[date]:
    return {date.fromisoformat(p.stem) for p in out.glob("*/*.done")}


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print,
         deps: pipeline.Deps | None = None,
         config: study_config.StudyConfig | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gate = sub.add_parser("should-run")
    gate.add_argument("--out", type=Path, required=True)
    gate.add_argument("--now", help="ISO time with offset (default: now)")
    for name in ("run", "mark-done"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--day", type=date.fromisoformat, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "should-run":
        now = datetime.fromisoformat(args.now) if args.now else datetime.now().astimezone()
        try:
            go, day = days.should_run(now, done_days=done_days(args.out))
        except ValueError as exc:
            echo(f"error={exc}")
            return 2
        echo(f"run={'true' if go else 'false'}")
        echo(f"day={day:%Y-%m-%d}")
        return 0

    if args.cmd == "mark-done":
        # Through `hub.atomic`, as every writer in the tree is
        # (`tests/contracts/test_writers_are_crash_atomic.py`).
        atomic.write_text(_marker(args.out, args.day),
                          f"{datetime.now().astimezone().isoformat()}\n")
        return 0

    cfg = config or study_config.load()
    try:
        runner.require_pinned(cfg)
    except runner.NotPinned as exc:
        echo(f"refused: {exc}")
        return 3
    try:
        outcome = pipeline.run_day(
            args.day, args.out, config=cfg,
            deps=deps or pipeline.production(run_id=os.environ.get("GITHUB_RUN_ID", "local")))
    except tuple(kind for kind, _ in REFUSALS) as exc:
        echo(f"refused: {exc}")
        return next(code for kind, code in REFUSALS if isinstance(exc, kind))
    echo(pipeline.summary(outcome))
    return 6 if outcome.status == "partial" else 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
