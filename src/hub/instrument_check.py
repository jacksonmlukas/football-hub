"""The week-4 instrument check (#389): prove the pipeline, the persistence and the ledger, and
publish no verdict.

Audit V is pre-registered against a scored week 4, and the combined rule's k is seasons, so four
scored weeks add no cluster: a verdict taken then reads SHOW whatever is true, and a SHOW that
arrives because the rule cannot see is indistinguishable from one that arrives because nothing
is there. This makes the week useful without making it a verdict. **No ADOPT, no REMOVE, no SHOW
is produced here, no published number moves, and nothing in this module imports a gate's
decision.** It reads four facts and says whether each holds.

Four criteria, each `PASS`, `FAIL` or `NOT-YET`:

1. **pipeline** -- a gate run over real scored rows completed and stamped `cfg_digest` and
   `data_digest` (read off the stamped paired parquet that `weekly_gate --run --out` writes).
2. **persistence** -- every `hub.fetch.odds` capture the window should have produced is in the
   archive **and readable from a fresh checkout**, not from the runner that wrote it. This is the
   criterion #383 exists to satisfy. It is checked in a clone of the repo's committed `HEAD` into a
   scratch directory, read with *that clone's* `hub.store` and no `data/`: a file present in the
   working tree and never committed is invisible to it, which is the point.
3. **ledger** -- the run is recorded in `state/gate-width.json`, and the entry's comparison
   partner shares its digest pair (`WidthEntry.comparable`), never a different one.
4. **k** -- clusters (seasons) and within-season rows per cluster, *reported*, so the entry reads
   against the horizon table in `docs/gate-power.md`. Never a pass or a fail on its own.

**`NOT-YET` is not `PASS`.** A criterion that has nothing to look at -- no scheduled poll has
passed its grace, no paired run was handed over -- says so and the run exits 3, not 0. A
persistence check that passes with nothing to persist is the same defect class as a gate that
cannot see what it gates (rule 18); the positive control (`tests/unit/test_instrument_check.py`)
removes one capture from a real committed archive and requires criterion 2 to go red, and
separately leaves one *untracked* in the working tree and requires the same.

Exit 0 only when all four are `PASS`; 1 when any is `FAIL`; 3 when none failed but one is not
yet checkable. Reads only the repo's committed tree and, if given, one parquet; it fetches
nothing and spends no credit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from hub.config import SEASON_AHEAD
from hub.fetch.odds import POLL_SCHEDULE
from hub.ledger import WIDTH_STATE, Ledger, WidthEntry
from hub.paths import ROOT

PASS, FAIL, NOT_YET = "PASS", "FAIL", "NOT-YET"
EXIT = {PASS: 0, FAIL: 1, NOT_YET: 3}

# A cron is a request for a start, not a guarantee of one -- `live.yml` records GitHub delivering
# runs 97-126 minutes late. A slot counts as checkable only once this has passed, and a capture
# counts for a slot if it landed inside it.
GRACE = timedelta(hours=6)

# The instant #383 (`f79bef4`) landed, 2026-10-06 15:24 EDT. No scheduled poll before it was
# ever committed, whatever it captured: the runner discarded it.
PERSISTENCE_BEGINS = datetime(2026, 10, 6, 19, 24, 48, tzinfo=UTC)

# The polls #428 back-filled from the maintainer's local store, one stamp each, all 18 lookahead
# weeks. They predate #383, so they are the only pre-persistence captures anyone can read.
BACKFILLED = ("20260825T045821", "20260827T202705", "20260827T212531", "20260828T020137",
              "20260904T151730", "20260904T152145", "20260904T155136", "20260906T114421")
BACKFILL_WEEKS = 18


@dataclass
class Criterion:
    name: str
    status: str
    detail: str
    facts: dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        return f"  [{self.status:>7}] {self.name}: {self.detail}"


# --- 2. persistence -----------------------------------------------------------------------

def expected_polls(start: datetime, until: datetime) -> list[datetime]:
    """Every scheduled poll instant in `[start, until]`, UTC, from `POLL_SCHEDULE`.

    Cron numbers Sunday 0; Python's `weekday()` numbers Monday 0, so `(w - 1) % 7`.
    """
    out = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= until:
        for p in POLL_SCHEDULE:
            if day.weekday() == (p.weekday - 1) % 7:
                slot = day.replace(hour=p.hour, minute=p.minute)
                if start <= slot <= until:
                    out.append(slot)
        day += timedelta(days=1)
    return sorted(out)


_READER = """
import json, sys
from pathlib import Path
had_data = Path("data").exists()    # before the reader can create one as a side effect
from hub import store
season = int(sys.argv[1])
print(json.dumps({"rows": store.lines(season).height, "has_data_dir": had_data}))
"""


def read_from_fresh_checkout(repo: Path, season: int) -> dict[str, object]:
    """Clone `repo`'s committed HEAD into a scratch directory and read the odds archive there.

    `git clone` copies commits and nothing else, so an untracked file, an ignored directory or a
    `data/` the writer happened to have are all absent. The rows are counted by the *clone's*
    `hub.store.lines`, not this process's, and the snapshot files are listed from the clone's own
    tree. Returns `{"captures": {week: [stamp, ...]}, "rows": int, "clone_has_data": bool}`.
    """
    tmp = tempfile.mkdtemp(prefix="389-fresh-")
    try:
        clone = Path(tmp) / "checkout"
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)],
                       check=True, capture_output=True, text=True)
        got = subprocess.run(
            [sys.executable, "-c", _READER, str(season)], cwd=clone, capture_output=True,
            text=True, env={**os.environ, "PYTHONPATH": str(clone / "src")}, timeout=300)
        if got.returncode != 0:
            raise RuntimeError(f"the fresh checkout's own reader failed: {got.stderr[-400:]}")
        read = json.loads(got.stdout.strip().splitlines()[-1])
        captures: dict[int, list[str]] = {}
        for f in sorted((clone / "state" / "odds" / str(season)).glob("wk*/snap-*.json")):
            captures.setdefault(int(f.parent.name[2:]), []).append(f.stem.removeprefix("snap-"))
        return {"captures": captures, "rows": read["rows"], "clone_has_data": read["has_data_dir"]}
    finally:
        # `rm`, not `shutil.rmtree`: the clone has a `site/data`, and a Python walk opens it by
        # the bare name `data` relative to the cwd, which the suite's real-data guard (#410)
        # cannot tell from the repo's own `data/` and rightly refuses.
        subprocess.run(["rm", "-rf", tmp], check=False)


def _stamp(s: str) -> datetime:
    return datetime.strptime(s + "+0000", "%Y%m%dT%H%M%S%z")


def persistence(repo: Path, season: int, week: int, start: datetime, until: datetime,
                *, now: datetime, grace: timedelta = GRACE) -> Criterion:
    """Criterion 2: every scheduled poll in `[start, until]` has its capture in the committed
    archive for `week`, as read from a fresh checkout.

    A poll slot is *checkable* once `slot + grace <= now`; the rest are counted and left out of
    the verdict, and a window with no checkable slot is `NOT-YET`, never `PASS`. A slot is
    *missing* when no committed snapshot for `week` is stamped inside `[slot, slot + grace]`.
    Slots before `PERSISTENCE_BEGINS` are named in the detail as unrecoverable-by-construction
    when missing: nothing was committed before #383 and a market price cannot be re-captured.
    """
    fresh = read_from_fresh_checkout(repo, season)
    stamps = [_stamp(s) for s in fresh["captures"].get(week, [])]  # type: ignore[attr-defined]
    slots = expected_polls(start, until)
    checkable = [s for s in slots if s + grace <= now]
    missing = [s for s in checkable if not any(s <= t <= s + grace for t in stamps)]
    pre = [s for s in missing if s < PERSISTENCE_BEGINS]
    facts = {"week": week, "window": [start.isoformat(), until.isoformat()],
             "scheduled": len(slots), "checkable": len(checkable),
             "missing": [s.isoformat() for s in missing],
             "missing_before_persistence": len(pre),
             "snapshots_for_week_in_fresh_checkout": len(stamps),
             "season_rows_read_by_fresh_checkout": fresh["rows"],
             "fresh_checkout_had_a_data_dir": fresh["clone_has_data"]}
    if fresh["clone_has_data"]:
        return Criterion("persistence", FAIL, "the scratch checkout contained data/, so what it "
                         "read proves nothing about a fresh one", facts)
    if not checkable:
        return Criterion("persistence", NOT_YET,
                         f"{len(slots)} poll(s) scheduled in the window and none past its "
                         f"{grace} grace yet, so there is nothing to have persisted", facts)
    if missing:
        why = (f"; {len(pre)} fall before #383 landed ({PERSISTENCE_BEGINS:%Y-%m-%d %H:%MZ}) "
               f"and were never committed (their runner discarded them) and cannot be re-captured"
               if pre else "")
        return Criterion("persistence", FAIL,
                         f"{len(missing)} of {len(checkable)} checkable poll(s) have no "
                         f"committed capture for week {week} in a fresh checkout: "
                         f"{', '.join(s.strftime('%m-%d %H:%MZ') for s in missing)}{why}", facts)
    return Criterion("persistence", PASS,
                     f"all {len(checkable)} checkable poll(s) are in a fresh checkout "
                     f"(week {week}, {fresh['rows']:,} season rows read by its own reader)", facts)


def backfill_readable(repo: Path, season: int) -> Criterion:
    """The #428 back-fill, asserted separately: every one of its 8 polls in all 18 weeks, from a
    fresh checkout. Reported beside criterion 2, not part of it -- these are the only
    pre-persistence captures there are."""
    fresh = read_from_fresh_checkout(repo, season)
    caps: dict[int, list[str]] = fresh["captures"]  # type: ignore[assignment]
    lacking = [(w, s) for w in range(1, BACKFILL_WEEKS + 1) for s in BACKFILLED
               if s not in caps.get(w, [])]
    ok = not lacking
    return Criterion("back-fill", PASS if ok else FAIL,
                     f"{len(BACKFILLED)} polls x {BACKFILL_WEEKS} weeks "
                     + ("all readable from a fresh checkout" if ok
                        else f"{len(lacking)} snapshot(s) missing, first {lacking[0]}")
                     + f"; {fresh['rows']:,} rows", {"rows": fresh["rows"]})


# --- 1, 3, 4: what a gate run leaves ---------------------------------------------------------

def _stamps(paired: pl.DataFrame) -> tuple[str | None, str | None]:
    def one(col: str) -> str | None:
        if col not in paired.columns:
            return None
        vals = paired[col].drop_nulls().unique().to_list()
        return vals[0] if len(vals) == 1 and vals[0] else None
    return one("cfg_digest"), one("data_digest")


def pipeline(paired: pl.DataFrame | None, season: int) -> Criterion:
    """Criterion 1, over the stamped paired frame a gate run wrote."""
    if paired is None:
        return Criterion("pipeline", NOT_YET,
                         "no stamped paired run was supplied (--paired); a run needs the "
                         "pinned nflverse data and cannot be made from a fresh checkout")
    cfg, data = _stamps(paired)
    rows = paired.filter(pl.col("season") == season).height if "season" in paired.columns else 0
    if paired.is_empty() or not rows:
        return Criterion("pipeline", FAIL, f"the run carries no rows for {season}: the pipeline "
                         f"did not reach the scored season", {"rows": paired.height})
    if cfg is None or data is None:
        return Criterion("pipeline", FAIL,
                         f"cfg_digest={cfg!r} data_digest={data!r}: a run that does not say "
                         f"which model and which bytes produced it is not stamped",
                         {"cfg_digest": cfg, "data_digest": data})
    return Criterion("pipeline", PASS, f"{rows:,} {season} rows; cfg_digest {cfg}, "
                     f"data_digest {data}", {"cfg_digest": cfg, "data_digest": data, "rows": rows})


def ledger(paired: pl.DataFrame | None, gate: str, path: Path | None = None) -> Criterion:
    """Criterion 3: the run is in the ledger, and its comparison partner shares its digest pair.

    The partner is the newest *earlier* entry of the gate that `WidthEntry.comparable` accepts --
    the ledger's own rule, not a second one -- and the check asserts that partner's digests equal
    the run's. `review_width` comparing across a pair is the defect of 2026-09-21 this guards.
    """
    if paired is None:
        return Criterion("ledger", NOT_YET, "no paired run was supplied, so there is no digest "
                         "pair to find in the ledger")
    cfg, data = _stamps(paired)
    entries = Ledger(path or WIDTH_STATE, write=False)._read()
    if entries is None:
        return Criterion("ledger", FAIL, "the ledger is on disk and does not parse")
    mine = [e for e in entries
            if e.name == gate and (e.config_digest, e.data_digest) == (cfg, data)]
    if not mine:
        return Criterion("ledger", FAIL,
                         f"no `{gate}` entry at cfg_digest {cfg}, data_digest {data}: the run "
                         f"was not recorded", {"entries": len(entries)})
    latest = mine[-1]
    earlier = entries[:len(entries) - 1 - entries[::-1].index(latest)]
    partner: WidthEntry | None = next((e for e in reversed(earlier) if e.name == gate
                                       and e.comparable(latest)), None)
    if partner is not None and (partner.config_digest, partner.data_digest) != (cfg, data):
        return Criterion("ledger", FAIL, "the comparison partner has another digest pair")
    said = (f"compared against the entry of {partner.timestamp} at the same digest pair"
            if partner else "no earlier entry at this digest pair, so nothing was compared")
    return Criterion("ledger", PASS, f"recorded ({len(mine)} at this digest pair); {said}",
                     {"verdict_recorded": latest.verdict, "width": latest.width})


def k_report(paired: pl.DataFrame | None, season: int) -> Criterion:
    """Criterion 4: the available k, reported. `season` is the one still being scored."""
    if paired is None or paired.is_empty() or "season" not in paired.columns:
        return Criterion("k", NOT_YET, "no paired run was supplied, so k is unreported")
    per = paired.group_by("season").agg(pl.len().alias("rows")).sort("season")
    bits = []
    for r in per.iter_rows(named=True):
        sub = paired.filter(pl.col("season") == r["season"])
        extra = "".join(f" x {sub[c].n_unique()} {c}s" for c in ("roster", "week")
                        if c in sub.columns)
        bits.append(f"{r['season']}: {r['rows']:,} rows{extra}")
    facts: dict[str, object] = {"k": per.height, "rows_by_season": dict(zip(per["season"].to_list(),
                                                        per["rows"].to_list(), strict=True))}
    return Criterion("k", PASS, f"k = {per.height} seasons ({'; '.join(bits)}); {season} is the "
                     f"partial one. Reported, not judged", facts)


# --- the run -------------------------------------------------------------------------------

def run(repo: Path, season: int, week: int, start: datetime, until: datetime, *,
        paired_path: Path | None, now: datetime, gate: str = "weekly") -> list[Criterion]:
    paired = pl.read_parquet(paired_path) if paired_path else None
    return [pipeline(paired, season),
            persistence(repo, season, week, start, until, now=now),
            ledger(paired, gate),
            k_report(paired, season),
            backfill_readable(repo, season)]


def _iso(s: str) -> datetime:
    """An instant from the command line; a naive one is read as UTC, the schedule's clock."""
    got = datetime.fromisoformat(s)
    return got if got.tzinfo else got.replace(tzinfo=UTC)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.instrument_check",
        description="Week-4 instrument check (#389). Produces no verdict.")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--from", dest="start", required=True, type=_iso,
                    help="start of the window the week's polls are expected in (UTC)")
    ap.add_argument("--until", required=True, type=_iso, help="end of that window (UTC)")
    ap.add_argument("--paired", type=Path, default=None,
                    help="the stamped paired parquet of a gate run (`--out`)")
    ap.add_argument("--gate", default="weekly", help="the ledger's name for that gate")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--now", type=_iso, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(list(argv) if argv is not None else None)
    now = a.now or datetime.now(UTC)
    try:
        results = run(a.repo, a.season, a.week, a.start, a.until, paired_path=a.paired,
                      now=now, gate=a.gate)
    except (subprocess.CalledProcessError, RuntimeError, OSError, pl.exceptions.PolarsError) as e:
        # An input that could not be read is not a criterion that failed: say which, exit
        # non-zero, no traceback (`tests/contracts/test_cli_surface.py`).
        from hub.cli import unavailable
        return unavailable("hub.instrument_check", "the repository or the paired run", e)
    print(f"instrument check, {a.season} week {a.week} -- no verdict is produced")
    for c in results:
        print(c.line())
    gating = [c for c in results if c.name != "back-fill" or c.status == FAIL]
    worst = (FAIL if any(c.status == FAIL for c in gating)
             else NOT_YET if any(c.status == NOT_YET for c in gating) else PASS)
    print(f"  => {worst}")
    return EXIT[worst]


if __name__ == "__main__":
    sys.exit(main())
