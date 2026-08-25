"""Summarize a dataset without pulling it into context.

`.claude/hooks/guard_data_reads.py` blocks raw reads of anything under `data/` and points
here instead. That makes this module the escape hatch for CLAUDE.md rule 1, and it has one
hard obligation: **never become the expensive read it exists to prevent.**

Two things enforce that.

*Lazy by construction.* Every mode goes through `pl.scan_parquet`. Nothing here calls
`read_parquet`, so `--schema` on full-width play-by-play costs parquet metadata rather than
40k tokens of rows. A test makes `read_parquet` raise, so a future edit that reaches for it
fails loudly rather than quietly costing a session its context.

*Capped centrally.* One `cap()` on the way out, applied to every mode. Per-mode limits are
how one path quietly grows past the budget later, and a 300-column `--schema` is exactly the
case that would.

    uv run python -m hub.inspect draft_board --schema
    uv run python -m hub.inspect draft_board --head 5 --cols player,pos,adp
    uv run python -m hub.inspect preds --nulls
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import polars as pl

from hub.store import DATA

# The budget. Output is for an agent's context window, not a terminal, so this is a hard
# ceiling rather than a page size.
MAX_LINES = 40

# A 300-column row is one line but still unreadable, so width is bounded too.
MAX_COLS = 8
MAX_CELL = 18


class DatasetNotFound(Exception):
    """Named dataset, column, or path that does not resolve."""


def _available(base: Path) -> list[str]:
    """Dataset names a user could have meant, for the error message."""
    if not base.exists():
        return []
    names = {p.stem for p in base.glob("*.parquet")}
    names |= {d.name for d in base.iterdir() if d.is_dir() and any(d.rglob("*.parquet"))}
    return sorted(names)


def resolve(name: str, base: Path | None = None) -> list[Path]:
    """Turn a dataset name into the parquet files behind it.

    Three shapes, in order: an explicit path; a bare name under the processed root; or a
    Hive-partitioned directory of the kind `hub.store.write` produces. The last returns
    many files, which `scan_parquet` handles as one frame.
    """
    base = base or DATA

    p = Path(name)
    if p.is_file():
        return [p]

    flat = base / f"{name}.parquet"
    if flat.is_file():
        return [flat]

    tree = base / name
    if tree.is_dir():
        parts = sorted(tree.rglob("*.parquet"))
        if parts:
            return parts

    known = _available(base)
    hint = ", ".join(known[:12]) if known else "(none found)"
    raise DatasetNotFound(f"no dataset {name!r} under {base}. Available: {hint}")


def _scan(paths: Sequence[Path]) -> pl.LazyFrame:
    return pl.scan_parquet([str(p) for p in paths])


# Aggregation key separator. A NUL cannot occur in a column name, so `col + SEP + stat`
# round-trips unambiguously however the column is spelled.
#
# It is a named constant rather than an inline "\x00" because a backslash escape inside an
# f-string *expression* is Python 3.12 syntax, and this project supports 3.11 -- so the
# inline version made `hub.inspect` a SyntaxError on the floor version. That matters more
# here than anywhere else: this module is what the data-read guard tells every agent to use
# instead of `cat`, so on 3.11 the recommended escape hatch failed to import.
SEP = chr(0)
_STATS = ("count", "mean", "std", "min", "max")


def _schema(paths: Sequence[Path]) -> dict[str, pl.DataType]:
    """Column types without touching a single row."""
    return dict(_scan(paths).collect_schema())


def cap(lines: list[str], limit: int = MAX_LINES) -> list[str]:
    """Truncate, and say how much was hidden rather than trailing off silently."""
    if len(lines) <= limit:
        return lines
    shown = limit - 1
    return [*lines[:shown], f"  ... {len(lines) - shown} more (use --cols to narrow)"]


def overview_lines(paths: Sequence[Path]) -> list[str]:
    schema = _schema(paths)
    rows = cast(int, _scan(paths).select(pl.len()).collect().item())
    kinds = Counter(str(d) for d in schema.values())
    out = [
        f"  {paths[0] if len(paths) == 1 else f'{len(paths)} files'}",
        f"  {rows:,} rows | {len(schema)} columns",
        "  dtypes: " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common(6)),
    ]
    return out


def schema_lines(paths: Sequence[Path]) -> list[str]:
    return [f"  {name:<32} {dtype}" for name, dtype in _schema(paths).items()]


def head_lines(paths: Sequence[Path], n: int = 5,
               cols: Sequence[str] | None = None) -> list[str]:
    """First n rows. Formatted by hand rather than via the polars repr, which is wide."""
    schema = _schema(paths)
    if cols:
        missing = [c for c in cols if c not in schema]
        if missing:
            raise DatasetNotFound(
                f"no column {missing!r}. Available: {', '.join(list(schema)[:12])}")
        keep = list(cols)
    else:
        keep = list(schema)

    hidden = max(0, len(keep) - MAX_COLS)
    keep = keep[:MAX_COLS]
    frame = _scan(paths).select(keep).head(n).collect()

    def cell(v: object) -> str:
        if v is None:
            return "-"
        # Full float precision is noise in a summary: `1.0686149314137383` says nothing
        # `1.0686` does not, and it crowds out a column that might.
        s = f"{v:.4g}" if isinstance(v, float) else str(v)
        return s[:MAX_CELL]

    out = ["  " + " | ".join(f"{c[:MAX_CELL]:<{MAX_CELL}}" for c in keep)]
    for row in frame.iter_rows():
        out.append("  " + " | ".join(f"{cell(v):<{MAX_CELL}}" for v in row))
    if hidden:
        out.append(f"  (+{hidden} more columns; use --cols to choose)")
    return out


def null_lines(paths: Sequence[Path]) -> list[str]:
    """Only columns that actually have nulls. The clean ones are noise."""
    schema = _schema(paths)
    counts = (_scan(paths)
              .select([pl.col(c).null_count().alias(c) for c in schema])
              .collect().row(0, named=True))
    rows = cast(int, _scan(paths).select(pl.len()).collect().item())
    out = [f"  {c:<32} {n:,} null ({n / rows:.1%})"
           for c, n in counts.items() if cast(int, n)]
    return out or ["  no nulls"]


def describe_lines(paths: Sequence[Path]) -> list[str]:
    """count / mean / std / min / max for numeric columns.

    Bounded to the first MAX_LINES numeric columns: the output is capped anyway, and
    building five aggregations for each of 300 columns is real work for lines nobody sees.
    """
    schema = _schema(paths)
    numeric = [c for c, d in schema.items() if d.is_numeric()][:MAX_LINES]
    if not numeric:
        return ["  no numeric columns"]

    aggs = []
    for c in numeric:
        aggs += [getattr(pl.col(c), stat)().alias(c + SEP + stat) for stat in _STATS]
    got = _scan(paths).select(aggs).collect().row(0, named=True)

    def num(key: str) -> str:
        v = got.get(key)
        return "-" if v is None else f"{float(cast(float, v)):.2f}"

    out = [f"  {'column':<24} {'count':>8} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"]
    for c in numeric:
        cells = " ".join(f"{num(c + SEP + stat):>10}" for stat in _STATS[1:])
        out.append(f"  {c[:24]:<24} {num(c + SEP + 'count'):>8} {cells}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.inspect",
        description="Summarize a dataset without reading it into context.")
    ap.add_argument("dataset", help="name (draft_board, preds) or path to a parquet file")
    ap.add_argument("--schema", action="store_true", help="column names and dtypes")
    ap.add_argument("--head", type=int, default=None, metavar="N", help="first N rows")
    ap.add_argument("--cols", default=None, help="comma-separated columns for --head")
    ap.add_argument("--describe", action="store_true", help="numeric summary statistics")
    ap.add_argument("--nulls", action="store_true", help="null counts, non-zero only")
    ap.add_argument("--base", default=None,
                    help="override the processed-data root (mainly for tests)")
    a = ap.parse_args(argv)

    try:
        paths = resolve(a.dataset, base=Path(a.base) if a.base else None)
        cols = [c.strip() for c in a.cols.split(",") if c.strip()] if a.cols else None
        if a.schema:
            lines = schema_lines(paths)
        elif a.head is not None:
            lines = head_lines(paths, n=a.head, cols=cols)
        elif a.describe:
            lines = describe_lines(paths)
        elif a.nulls:
            lines = null_lines(paths)
        else:
            lines = overview_lines(paths)
    except DatasetNotFound as e:
        print(f"hub.inspect: {e}", file=sys.stderr)
        return 1

    print("\n".join(cap(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
