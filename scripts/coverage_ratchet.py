"""Fail when a module leaves more statements untested than it is committed to.

Run after the suite, against the JSON coverage report:

    uv run pytest tests/unit tests/contracts --cov=hub --cov-report=json:cov.json
    uv run python scripts/coverage_ratchet.py cov.json

`--update` rewrites the committed floor to match the report, which is how an improvement is
recorded. It never raises a floor: a module that got worse has to be argued for in a diff,
which is the whole point -- see `tests/coveragefloor.py` for why a total floor could not see
the loss that prompted this.

A separate script rather than a test, because a test cannot read the coverage of the run it is
part of. What *is* a test is that every module has an entry at all
(`tests/contracts/test_coverage_floor.py`), so a new module cannot be silently exempt while
this script is only looking at the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR_FILE = ROOT / "tests" / "coveragefloor.py"
sys.path.insert(0, str(ROOT / "tests"))


def _measured(report: Path) -> dict[str, int]:
    got = json.loads(report.read_text())
    return {k: v["summary"]["missing_lines"] for k, v in got["files"].items()}


def _rewrite(now: dict[str, int]) -> None:
    src = FLOOR_FILE.read_text()
    head = src[: src.index("FLOOR: dict[str, int] = {")]
    body = "\n".join(f'    "{k}": {n},' for k, n in sorted(now.items()))
    FLOOR_FILE.write_text(f"{head}FLOOR: dict[str, int] = {{\n{body}\n}}\n")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    update = "--update" in args
    rest = [a for a in args if not a.startswith("-")]
    if not rest:
        print("usage: coverage_ratchet.py <coverage.json> [--update]", file=sys.stderr)
        return 2

    from coveragefloor import FLOOR

    now = _measured(Path(rest[0]))
    if update:
        _rewrite(now)
        moved = {k: (FLOOR.get(k), v) for k, v in now.items() if FLOOR.get(k) != v}
        print(f"  floor updated: {len(moved)} modules changed")
        return 0

    worse = {k: (FLOOR[k], v) for k, v in now.items() if k in FLOOR and v > FLOOR[k]}
    unlisted = sorted(k for k in now if k not in FLOOR)
    better = {k: (FLOOR[k], v) for k, v in now.items() if k in FLOOR and v < FLOOR[k]}

    for module, (was, is_now) in sorted(worse.items()):
        print(f"  WORSE  {module}: {was} statements untested -> {is_now}", file=sys.stderr)
    for module in unlisted:
        print(f"  UNLISTED  {module}: no committed floor", file=sys.stderr)
    if better:
        print(f"  {len(better)} module(s) improved; run with --update to record it")
    if worse or unlisted:
        print("\n  A module stopped being exercised. That is what this gate is for: the "
              "total\n  floor cannot see it -- see tests/coveragefloor.py. Add tests, or run "
              "with\n  --update if the loss is intended and say why in the commit.",
              file=sys.stderr)
        return 1
    print(f"  coverage floor held for {len(FLOOR)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
