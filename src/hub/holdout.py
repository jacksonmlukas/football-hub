"""A per-season constant set, fitted without that season, that a gate can replay under (#294).

`hub.draft.backtest.LIMITATIONS` names the defect: the simulator's constants were fitted on
the seasons the gate replays, so arm B carries in-sample constants on every held-out season
and the headline is a lower bound on how badly it loses out of sample. The fix is
leave-one-season-out, in three parts:

* every fitting script under `scripts/` takes `--exclude-season N` and, with `--record`,
  writes what it fitted into **one committed file per season**, `conf/holdout/N.json`,
  under the constant's digest key (`predict.WEEKLY_K`, `availability.PICK_NOISE_SLOPE`) with
  the command that produced it;
* a gate told to replay under hold-out wraps each season in `applied(season)`, which
  rebinds the module attributes to that season's set for the duration and restores the
  shipped values after -- so the shipped constants, and every digest on the default path,
  are untouched;
* the run line says, per season, which constants were refitted and which stayed shipped
  and why, so a hold-out replay that could only refit some of the eleven is read as that.

**What is held out is the list, not "everything fitted".** `HELD_OUT` holds eleven keys:
the eight constants the LIMITATIONS entry names -- `TALENT_CV`, `TALENT_CV_BY_POS`,
`IMPUTE_CV`, `WEEKLY_K`, `WEEKLY_SKEW`, `TEAMMATE_RHO`, `PICK_NOISE_INTERCEPT`,
`PICK_NOISE_SLOPE` -- and the three companions they ship beside and are read with,
`IMPUTE_CV_BY_POS`, `WEEKLY_K_POOLED` and `WEEKLY_SKEW_POOLED`, which a set that refits
the constant refits in the same run. `record` refuses any other key: a set that quietly
carried `MIN_SKEW` would be a second mechanism for the thing `hub.declare` made one.

**A missing key is recorded, not assumed.** Three of the eight (`TALENT_CV`,
`PICK_NOISE_INTERCEPT`, `PICK_NOISE_SLOPE`) are fitted on this league's ESPN draft history
and cannot be refitted without a session; a script that cannot run records `why_not` for
the key and the run line carries it. A season
with no file at all is refused: replaying "under hold-out" on a season nobody fitted for
would be the shipped run wearing the hold-out's name.

Values are stored as JSON. `TEAMMATE_RHO` is keyed by position pairs, which JSON cannot
key, so a tuple key is written `"QB|WR"` and read back as the tuple.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from hub import atomic, declare
from hub.paths import ROOT

# Where the sets live. `HUB_HOLDOUT_SETS` points a run at another directory -- a set fitted
# elsewhere, or a test's -- and reaches a spawned worker, which imports this module afresh
# and would not see a rebinding of the name in the parent.
SETS = Path(os.environ["HUB_HOLDOUT_SETS"]) if os.environ.get("HUB_HOLDOUT_SETS") \
    else ROOT / "conf" / "holdout"

# The constants a hold-out set may carry: the eight `backtest.LIMITATIONS` names plus the
# three companions (`IMPUTE_CV_BY_POS`, `WEEKLY_K_POOLED`, `WEEKLY_SKEW_POOLED`) -- eleven
# keys. A gate under hold-out reads these and nothing else.
HELD_OUT: tuple[str, ...] = (
    "predict.TALENT_CV", "predict.TALENT_CV_BY_POS",
    "predict.IMPUTE_CV", "predict.IMPUTE_CV_BY_POS",
    "predict.WEEKLY_K", "predict.WEEKLY_K_POOLED",
    "predict.WEEKLY_SKEW", "predict.WEEKLY_SKEW_POOLED",
    "predict.TEAMMATE_RHO",
    "availability.PICK_NOISE_INTERCEPT", "availability.PICK_NOISE_SLOPE",
)

PAIR_SEP = "|"


@dataclass(frozen=True)
class ConstantSet:
    """One season's set: what was refitted without it, how, and what could not be."""
    season: int
    values: dict[str, Any]
    commands: dict[str, str]
    missing: dict[str, str] = field(default_factory=dict)   # key -> why it was not refitted
    path: Path | None = None


def _path(season: int, root: Path | None) -> Path:
    return (root or SETS) / f"{season}.json"


def _to_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {(PAIR_SEP.join(k) if isinstance(k, tuple) else str(k)): _to_json(x)
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json(x) for x in v]
    return v


def _from_json(key: str, v: Any) -> Any:
    if isinstance(v, dict):
        return {(tuple(k.split(PAIR_SEP)) if PAIR_SEP in k else k): _from_json(key, x)
                for k, x in v.items()}
    if isinstance(v, list):
        # A tuple constant (`PICK_NOISE_SLOPE_CI`-shaped) round-trips as a list; the held-out
        # keys are scalars and dicts, so a list here is read back as a tuple.
        return tuple(_from_json(key, x) for x in v)
    return v


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def record(season: int, key: str, value: Any, *, command: str, root: Path | None = None,
           why_not: str | None = None) -> Path:
    """Write one constant into the season's set, keeping every other key the file holds.

    `value=None` with `why_not` records that the key could not be refitted and why; the
    run line carries the reason. A key outside `HELD_OUT` is refused.
    """
    if key not in HELD_OUT:
        raise KeyError(f"{key!r} is not one of the held-out constants: {', '.join(HELD_OUT)}")
    path = _path(season, root)
    raw = _read(path)
    raw.setdefault("excluded_season", season)
    constants = raw.setdefault("constants", {})
    stamp = datetime.now(UTC).date().isoformat()
    if value is None:
        if not why_not:
            raise ValueError(f"{key}: a missing value needs why_not")
        constants[key] = {"value": None, "command": command, "why_not": why_not,
                          "recorded": stamp}
    else:
        constants[key] = {"value": _to_json(value), "command": command, "recorded": stamp}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return path


def load(season: int, root: Path | None = None) -> ConstantSet:
    """The season's set. Refused, not defaulted, when no file exists."""
    path = _path(season, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no hold-out constant set for {season} at {path}: fit one with the scripts "
            f"under scripts/ and --exclude-season {season} --record")
    raw = _read(path)
    if raw.get("excluded_season") != season:
        raise ValueError(f"{path} says excluded_season={raw.get('excluded_season')!r}, "
                         f"not {season}")
    values: dict[str, Any] = {}
    commands: dict[str, str] = {}
    missing: dict[str, str] = {}
    for key, entry in (raw.get("constants") or {}).items():
        if key not in HELD_OUT:
            raise KeyError(f"{path} carries {key!r}, which is not a held-out constant")
        if entry.get("value") is None:
            missing[key] = entry.get("why_not") or "no reason recorded"
        else:
            values[key] = _from_json(key, entry["value"])
        commands[key] = entry.get("command", "")
    return ConstantSet(season, values, commands, missing, path)


def _shown(path: Path | None) -> str:
    """A set's path as a run line spells it: repo-relative when it is under the repo."""
    if path is None:
        return "?"
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def describe(cs: ConstantSet) -> str:
    """The run line: which of the held-out constants this set refits, and which a replay
    under it still reads shipped, with why."""
    refitted = [k for k in HELD_OUT if k in cs.values]
    shipped = [k for k in HELD_OUT if k not in cs.values]
    reasons = sorted({cs.missing[k] for k in shipped if k in cs.missing})
    where = _shown(cs.path)
    out = (f"season {cs.season}: constants from {where} (fitted without {cs.season}); "
           f"refitted: {', '.join(refitted) if refitted else 'none'}; "
           f"shipped (not refitted): {', '.join(shipped) if shipped else 'none'}")
    if reasons:
        out += " -- " + "; ".join(reasons)
    return out


def run_lines(seasons: Sequence[int], root: Path | None = None) -> list[str]:
    """One run line per season naming the set it will read. Loading every set first is the
    point: a season with no set raises here, before a gate plays a draft."""
    return [describe(load(season, root)) for season in seasons]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """The two flags every fitting script takes: which season to hold out, and whether to
    write what it fitted into that season's set."""
    ap.add_argument("--exclude-season", type=int, default=None, metavar="N",
                    help="hold season N out of the fit (#294)")
    ap.add_argument("--record", action="store_true",
                    help="write the fitted values into conf/holdout/N.json for the excluded "
                         "season; needs --exclude-season")


def recording(args: argparse.Namespace, command: str) -> Callable[..., None]:
    """A recorder bound to the script's flags: `note(key, value)` writes into the excluded
    season's set when `--record` was given and is a no-op otherwise. `note(key, None,
    why_not=...)` records that a key could not be refitted."""
    if args.record and args.exclude_season is None:
        raise SystemExit("--record needs --exclude-season N: a set is one season's")

    def note(key: str, value: Any, *, why_not: str | None = None) -> None:
        if not args.record:
            return
        path = record(args.exclude_season, key, value, command=command, why_not=why_not)
        print(f"  recorded {key} into {_shown(path)}")
    return note


def command_line(script: str, args: argparse.Namespace) -> str:
    """The fitting command a set records: the script and its hold-out flag, nothing else."""
    return f"uv run python {script} --exclude-season {args.exclude_season} --record"


def _target(key: str) -> tuple[Any, str]:
    name = key.split(".", 1)[1]
    for d in declare.declarations():
        if d.key == key:
            return import_module(d.module), name
    raise KeyError(f"{key} is not a declared constant")


@contextmanager
def applied(season: int, root: Path | None = None) -> Iterator[str]:
    """Rebind each constant the season's set holds for the duration of the block, and put
    the shipped value back after -- on success and on an exception alike. Yields the run
    line, so the caller prints what the block ran under."""
    cs = load(season, root)
    saved: list[tuple[Any, str, Any]] = []
    try:
        for key, value in cs.values.items():
            mod, name = _target(key)
            saved.append((mod, name, getattr(mod, name)))
            setattr(mod, name, value)
        yield describe(cs)
    finally:
        for mod, name, before in reversed(saved):
            setattr(mod, name, before)
