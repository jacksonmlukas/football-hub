"""A file lands whole or not at all.

`CLAUDE.md` promises last-good state when a fetch fails, and until #258 every writer of that
state -- the store's partitions, the fetchers' cache files and `captured_at` stamps, the
site's artifacts -- wrote straight to its final path in one call. `write_text`, `write_bytes`
and `write_parquet` all truncate first and fill second, so a process killed between the two
(a runner cancelled, a laptop asleep) leaves a file that is shorter than it says it is. The
next read either refuses it through its contract or parses it as garbage, and the fallback
that exists to survive an outage is the thing the outage destroyed.

The fix is the old one: write to a scratch name beside the target and rename over it in one
`os.replace`, which POSIX makes atomic on one filesystem. A reader sees the old bytes or the
new bytes and never a mixture; an interrupted write leaves the old file exactly as it was.

**One helper, and a scan that holds every writer to it.** `tests/contracts/
test_writers_are_crash_atomic.py` parses the modules that serve last-good state and refuses
a `.write_text`, `.write_bytes` or `.write_parquet` call that is not this module's. The
three functions here carry the pathlib and polars names on purpose: the call site reads
the same, with `atomic.` in front.

A leaf: `os`, `pathlib`, `contextlib` and polars, nothing from `hub`, so the store and every
fetcher can take it without either importing the other -- the same reason `hub.jsonio` and
`hub.paths` are leaves.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

# The scratch name: hidden, suffixed so no `*.parquet` or `*.json` glob ever lists a partial,
# and carrying the pid so two processes writing one file do not share a scratch. A `.part`
# left behind by a hard kill is inert -- nothing reads the suffix -- and the next write of the
# same file from the same pid truncates it.
SUFFIX = ".part"


@contextmanager
def replacing(path: Path) -> Iterator[Path]:
    """Yield a scratch path beside `path`; on a clean exit it becomes `path` in one rename.

    On any exception the scratch is removed and `path` is untouched, which is the whole
    property: a writer that dies with half its bytes on disk leaves the previous file
    served and no partial anywhere. Parents are created here, once, rather than at
    every call site that used to do it on its own line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.{os.getpid()}{SUFFIX}")
    try:
        yield scratch
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise
    os.replace(scratch, path)


def write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """`Path.write_text`, landing whole."""
    with replacing(path) as scratch:
        scratch.write_text(text, encoding=encoding)
    return Path(path)


def write_bytes(path: Path, data: bytes) -> Path:
    """`Path.write_bytes`, landing whole."""
    with replacing(path) as scratch:
        scratch.write_bytes(data)
    return Path(path)


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """`DataFrame.write_parquet`, landing whole.

    Polars writes the footer last, so a truncated parquet is one the reader cannot open at
    all -- which is how `hub.store.write` came to name one as "different data" (#258).
    """
    with replacing(path) as scratch:
        df.write_parquet(scratch)
    return Path(path)
