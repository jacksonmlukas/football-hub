"""The commands `.github/workflows/forecast.yml` runs, in order: `should-run`, `run`, `mark-done`.

`should-run` prints `run=true|false` and `day=YYYY-MM-DD` for `$GITHUB_OUTPUT`. `run` forecasts
the day's games into `--out` (the checked-out `forecasts` branch). `mark-done` writes the day's
marker once `run` has succeeded; a run that fails leaves none, so the next trigger in the window
retries and the runner's idempotency fills only the gaps.

Exit codes: 0 ok, 2 bad arguments, 3 the config is not pinned for v0 (no model was called),
4 the pipeline is not assembled yet (item 6 of the build; no model was called).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

from hub.forecasters import config as study_config
from hub.forecasters import days, runner


def _marker(out: Path, day: date) -> Path:
    return out / f"{day:%Y}" / f"{day:%Y-%m-%d}.done"


def done_days(out: Path) -> set[date]:
    return {date.fromisoformat(p.stem) for p in out.glob("*/*.done")}


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print) -> int:
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
        m = _marker(args.out, args.day)
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(f"{datetime.now().astimezone().isoformat()}\n")
        return 0

    try:
        runner.require_pinned(study_config.load())
    except runner.NotPinned as exc:
        echo(f"refused: {exc}")
        return 3
    echo("refused: the forecast pipeline (market snapshot -> packets -> runner) is not "
         "assembled yet; no model was called")
    return 4


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
