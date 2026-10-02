"""Replay: the nflverse seam's second adapter, serving a recorded set instead of the network.

`hub.fetch.nflverse.load` asks an `Adapter` for the bytes of one source. The network is the
first adapter and this is the second, which is what makes that seam a seam rather than a place
a test patches. Everything above the adapter -- narrowing, cleaning, the contract, the cache,
the pin, the run's record of what it read -- is production code running on recorded bytes, so
a run can be repeated offline and its data digest reproduced.

**It never stands in for a fetch the run did not make.** A source the set does not hold raises
`NotRecorded` rather than reaching the wire or returning an empty frame, because a source that
quietly came from somewhere else is a test that passes on a laptop and hangs in CI, and one
whose result nobody recorded.

Not a Snapshot (a dated capture of the betting market): a Replay is a recording of what one
run read. Select it with `hub.fetch.nflverse.serving(replay, cache=tmp)`.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

from hub.fetch.nflverse import SOURCES, Source

Edit = Callable[[pl.DataFrame], pl.DataFrame]
Table = pl.DataFrame | Callable[[Sequence[int | str]], pl.DataFrame]


class NotRecorded(Exception):
    """A source the recorded set does not hold was asked for. Loud, rather than a live fetch."""


def read_recording(path: Path) -> pl.DataFrame:
    """One recorded table, in the dtypes it was recorded with.

    The dtype map is part of the recording and not decoration. JSON has three scalar types and
    nflverse has a dozen; inferred, a column of whole numbers comes back `Int64` and a column
    that happened to be null on every recorded row comes back `Null`, and either one changes
    what the assembly does with it. Recording the dtypes keeps the file a statement about the
    frame nflverse returned rather than about what JSON could carry.
    """
    blob = json.loads(path.read_text())
    schema = {c: getattr(pl, t) for c, t in blob["dtypes"].items()}
    return pl.DataFrame(blob["rows"], schema=schema)


def _seasons(keys: Sequence[int | str]) -> list[int]:
    return [int(k) for k in keys]


class Replay:
    """Serves each recorded source as recorded, and keeps a log of what it served.

    `tables` maps a source name to a frame, served as given, or to a callable taking the
    partition key and returning one -- which is how a test records a call, or makes one fail.
    `served` is that log: `(source, keys)` in order, so "the second build fetched nothing" is
    a fact a test reads rather than a count it patches in.
    """

    def __init__(self, tables: Mapping[str, Table], *,
                 absent: Mapping[str, str] | None = None) -> None:
        self._tables = dict(tables)
        self._absent = dict(absent or {})
        self.served: list[tuple[str, tuple[int | str, ...]]] = []

    def add(self, **tables: Table) -> None:
        """Hold more sources, replacing any of the same name."""
        self._tables.update(tables)

    def read(self, source: Source, keys: Sequence[int | str]) -> pl.DataFrame:
        if source.name not in self._tables:
            why = self._absent.get(source.name, "")
            raise NotRecorded(
                f"{source.name} is not in this recorded set (it holds "
                f"{sorted(self._tables)}). A Replay never reaches the wire for a source it "
                f"does not hold. {why}".strip())
        self.served.append((source.name, tuple(keys)))
        table = self._tables[source.name]
        return table(keys) if callable(table) else table

    @classmethod
    def recorded(cls, directory: Path, sources: Sequence[str], *,
                 edits: Mapping[str, Edit] | None = None,
                 absent: Mapping[str, str] | None = None) -> Replay:
        """A set read from `<directory>/<source>.json`, served the way the wire serves it.

        The wire picks the season *files* a request names, so a season-keyed source is
        narrowed to the seasons asked for here, on the integer the key names -- `ff_opportunity`
        records `season` as a string, and the comparison is made after a cast. `edits` maps a
        source to a transform applied on the way out, which is how a leakage rule is asserted
        behaviourally: change one player-week's realised play, rebuild, and see what moves.
        """
        edits = dict(edits or {})

        def serve(name: str) -> Table:
            def table(keys: Sequence[int | str]) -> pl.DataFrame:
                df = read_recording(directory / f"{name}.json")
                if name in edits:
                    df = edits[name](df)
                source = SOURCES[name]
                if source.key == "seasons" and not source.whole:
                    df = df.filter(pl.col("season").cast(pl.Int64).is_in(_seasons(keys)))
                return df
            return table

        return cls({name: serve(name) for name in sources}, absent=absent)
