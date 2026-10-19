"""The league's free agents, fetched once and kept as last-good (#493).

One bulk ESPN request per run (`hub.fetch.espn.fetch_free_agents`), the same cookie path as the
roster, never a request per player, position or team. The frame is validated against
`ESPN_FREE_AGENTS` where it enters and written to `data/processed/free_agents.parquet`; a small
JSON beside it says when it was fetched, whether the pool came back at the cap, and -- if the
latest attempt failed -- why the parquet on disk is the one being served.

**A failed fetch serves last-good, with the reason.** The parquet is not rewritten, so its own
modification time stays the date of the last good pool; the reason lands in the sidecar, which
is what the published waiver block reads. With nothing on disk the answer is a sentence and a
non-zero exit.

    uv run python -m hub.fetch.free_agents            # fetch and write
    uv run python -m hub.fetch.free_agents --out PATH # write somewhere else (tests, trials)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.paths import FREE_AGENTS_PARQUET

Fetch = Callable[[], tuple[pl.DataFrame, bool]]


def meta_path(parquet: Path) -> Path:
    return parquet.with_suffix(".json")


def read_meta(parquet: Path) -> dict[str, Any]:
    """The sidecar, or `{}` when it is absent or unreadable: a missing note is not an error."""
    try:
        got = json.loads(meta_path(parquet).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def read(parquet: Path | None = None) -> tuple[pl.DataFrame | None, dict[str, Any]]:
    """The last good pool and its sidecar; `(None, {})` when none has ever been fetched."""
    path = parquet or FREE_AGENTS_PARQUET
    if not path.exists():
        return None, {}
    return pl.read_parquet(path), read_meta(path)


def refresh(fetch: Fetch, parquet: Path | None = None,
            now: datetime | None = None) -> tuple[str, pl.DataFrame | None]:
    """Fetch and write, or serve last-good. Returns `(status line, the frame served)`.

    The frame is None only when the fetch failed and nothing is on disk -- the caller turns
    that into a non-zero exit. A fetch that fails leaves the parquet untouched, so the file's
    own date stays the date of the pool it holds.
    """
    path = parquet or FREE_AGENTS_PARQUET
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    try:
        df, capped = fetch()
    except Exception as e:                      # graceful degradation, per CLAUDE.md
        # The full message stays on local stderr: it can carry request detail (URLs, league
        # ids). What is written beside the pool, and so published, is the exception type and a
        # fixed sentence.
        print(f"hub.fetch.free_agents: fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
        reason = f"{type(e).__name__}: the latest ESPN free-agent fetch failed"
        if not path.exists():
            return (f"no free agents: the fetch failed ({type(e).__name__}) and none are on "
                    "disk"), None
        meta = {**read_meta(path), "last_error": reason, "last_error_at": stamp}
        atomic.write_text(meta_path(path), json.dumps(meta, indent=1, sort_keys=True) + "\n")
        return (f"ESPN unreachable ({type(e).__name__}); serving last-good {path.name}",
                pl.read_parquet(path))
    atomic.write_parquet(df, path)
    atomic.write_text(meta_path(path), json.dumps(
        {"fetched_at": stamp, "n": df.height, "capped": bool(capped), "last_error": None},
        indent=1, sort_keys=True) + "\n")
    note = "; the pool came back at the cap, so its least-owned tail may be cut" if capped else ""
    return f"fetched {df.height} free agents{note}", df


def summary(df: pl.DataFrame) -> str:
    """Counts by position -- a summary, never the rows."""
    by = df.group_by("pos").len().sort("pos")
    return ", ".join(f"{r['pos']} {r['len']}" for r in by.iter_rows(named=True))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.free_agents",
        description="Fetch the league's free agents (one ESPN request) and keep them last-good.")
    ap.add_argument("--out", default=None, help="override the parquet path, read and written")
    a = ap.parse_args(list(argv) if argv is not None else None)
    path = Path(a.out) if a.out else FREE_AGENTS_PARQUET

    def live() -> tuple[pl.DataFrame, bool]:   # pragma: no cover - network
        from hub.fetch.espn import fetch_free_agents
        return fetch_free_agents()

    line, df = refresh(live, path)
    if df is None:
        print(f"hub.fetch.free_agents: {line}", file=sys.stderr)
        return 1
    print(f"  {line}")
    print(f"  by position: {summary(df)}")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
