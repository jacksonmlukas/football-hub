"""Is week N played *and* scored? A check, where #325 and #326 had a sentence (#417).

**The failure this is for.** Audit V's precondition read "`preds_2026_wk04.json` exists and
`survivor.json` `weeks_played` contains 4". Both held on Saturday 2026-10-03 with 1 of the 16
week-4 games scored. A week's preds file is written *before* the week is played, and since #263
`survivor.played` counts a week once any game in it kicks off -- right for locking a pick,
wrong as an audit gate. The prose test went stale silently; nothing could notice.

**What this reads.** Only `site/data`: the week's preds file, for the games predicted, and
`track_record.json`, whose per-season `scored_game_ids` (written by `publish._curve` from the
join `publish._finished` makes) are the games the published record scored. It never fetches
nflverse and never opens `data/`, so it answers the same on a machine with the network down.

**What it answers.** Exit 0 only when every predicted `game_id` is among the scored ids.
Otherwise non-zero, naming each unscored game. There is no postponed special case: a game that
was moved is named like any other, and what to do about it is the maintainer's call on the list.

**A refusal is not a pass.** A missing or unreadable artifact, or a record that predates the
field, exits 2 with a sentence. A record that cannot say which games it scored has established
nothing, and reading that as "all scored" is the failure this exists to remove.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hub import publish
from hub.paths import ROOT

REFUSED = 2
UNSCORED = 1


class Refusal(Exception):
    """The check could not be made. Distinct from "made, and the answer is no"."""


def _read(path: Path, what: str) -> dict[str, Any]:
    try:
        got = json.loads(path.read_text())
    except OSError as e:
        raise Refusal(f"{what} is not readable ({path}: {type(e).__name__})") from e
    except ValueError as e:
        raise Refusal(f"{what} is not valid JSON ({path})") from e
    if not isinstance(got, dict):
        raise Refusal(f"{what} is not an artifact object ({path})")
    return got


def predicted(preds: dict[str, Any], where: str) -> list[str]:
    rows = preds.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Refusal(f"{where} carries no predicted games, so there is nothing to check")
    ids = [r.get("game_id") if isinstance(r, dict) else None for r in rows]
    if not all(isinstance(i, str) for i in ids):
        raise Refusal(f"{where} has a row with no game_id")
    return sorted(set(ids))  # type: ignore[arg-type]


def scored_ids(record: dict[str, Any], season: int) -> set[str]:
    """The game_ids the published record scored for `season`.

    Refuses when any season entry lacks `scored_game_ids`: a record written before the field
    existed cannot say which games it scored. A record that does carry it for other seasons
    and has no entry for this one has scored none of this one's games.
    """
    seasons = record.get("seasons")
    if not isinstance(seasons, list) or not seasons:
        raise Refusal("track_record.json carries no per-season entries to read scored games "
                      "from")
    got: set[str] = set()
    for entry in seasons:
        if not isinstance(entry, dict) or not isinstance(entry.get("scored_game_ids"), list):
            raise Refusal("track_record.json predates `scored_game_ids`: it counts scored "
                          "games per season but does not say which. The next slate run "
                          "publishes the field; until then this check cannot answer")
        if entry.get("season") == season:
            got.update(entry["scored_game_ids"])
    return got


def check(season: int, week: int, site: Path) -> list[str]:
    """The predicted games of this week the published record has not scored (empty = ready)."""
    name = publish.preds_name(season, week) + ".json"
    ids = predicted(_read(site / name, name), name)
    scored = scored_ids(_read(site / "track_record.json", "track_record.json"), season)
    return [g for g in ids if g not in scored]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.audit_ready",
        description="Exit 0 only when every game in the week's published predictions has a "
                    "scored result in the published record; otherwise name the unscored "
                    "games. Reads site/data only.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--site", type=Path, default=None,
                    help="the published artifacts directory (default: site/data)")
    a = ap.parse_args(argv)
    site = a.site if a.site is not None else ROOT / "site" / "data"
    tag = f"{a.season} week {a.week}"
    try:
        missing = check(a.season, a.week, site)
    except Refusal as e:
        print(f"audit_ready: REFUSED for {tag}: {e}", file=sys.stderr)
        return REFUSED
    if missing:
        print(f"audit_ready: NOT READY for {tag}: {len(missing)} predicted game(s) have no "
              "scored result in the published record:", file=sys.stderr)
        for g in missing:
            print(f"  unscored: {g}", file=sys.stderr)
        return UNSCORED
    print(f"audit_ready: READY for {tag}: every predicted game is scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
