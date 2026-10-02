"""The two readers of "every poll this season" answer the same frames before and after #399.

`hub.fetch.odds._archive` and `hub.models.starter_change.archive` asked the `lines` table the
same question with different projections, and each repeated the fresh-clone guard and wrote the
empty schema by hand. #399 gave the question one owner, `store.lines`. A refactor of a reader
may not change a frame it hands back, so this file was written before it: the `EXPECTED` digests
were measured on the code where each reader held its own query, on a store built under
`tmp_path` that carries every state the real one has been in --

* a fresh clone, where there is no `lines` table at all and the answer is a typed empty frame;
* an archive written wholly before #211, with no `spread_price` column in any partition;
* an archive that mixes those partitions with ones that have it, which is the live shape;
* another league and another season beside the one asked for, which must not leak in.

A digest of the frame (schema and content, row order excluded: the store gives none).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import polars as pl
import pytest

from hub import store
from hub.fetch import nflverse as nv
from hub.fetch import odds
from hub.models import starter_change

# Measured on 2026-10-02 with each reader still holding its own query.
EXPECTED = {
    ("fresh", "odds"): "81207b871a0a045e",
    ("fresh", "starter_change"): "d05de48317bba255",
    ("pre211", "odds"): "23ca8cb3651d9ed9",
    ("pre211", "starter_change"): "815e0e6500e83527",
    ("mixed", "odds"): "f179f78d1759503d",
    ("mixed", "starter_change"): "f9717de90fcb8d10",
}


def _poll(rows, *, price: bool) -> pl.DataFrame:
    data = {"game_id": [r[0] for r in rows], "close_spread": [r[1] for r in rows],
            "captured_at": [r[2] for r in rows]}
    schema = {"game_id": pl.Utf8, "close_spread": pl.Float64, "captured_at": pl.Datetime}
    if price:
        data["spread_price"] = [r[3] for r in rows]
        schema["spread_price"] = pl.Float64
    return pl.DataFrame(data, schema=schema)


def _store(kind: str, base: Path) -> None:
    if kind == "fresh":
        return
    t0 = dt.datetime(2025, 9, 1, 12)
    old = [(f"g{i}", -3.0 + i, t0 + dt.timedelta(hours=i), None) for i in range(4)]
    store.write(_poll(old, price=False), "lines", "nfl", 2025, 1, name="a", base=base)
    store.write(_poll([(f"g{i}", -2.5 + i, t0 + dt.timedelta(days=1, hours=i), None)
                       for i in range(4)], price=False),
                "lines", "nfl", 2025, 2, name="a", base=base)
    if kind == "mixed":
        store.write(_poll([(f"g{i}", -2.0 + i, t0 + dt.timedelta(days=2, hours=i), -110.0 - i)
                           for i in range(4)], price=True),
                    "lines", "nfl", 2025, 3, name="b", base=base)
    # What must not leak in: another league this season, and this league another season.
    store.write(_poll([("x1", 9.0, t0, None)], price=False), "lines", "ncaa", 2025, 1, name="c",
                base=base)
    store.write(_poll([("y1", 7.0, t0, None)], price=False), "lines", "nfl", 2024, 1, name="d",
                base=base)


def _digest(frame: pl.DataFrame) -> str:
    schema = ";".join(f"{c}:{t}" for c, t in frame.schema.items())
    return hashlib.sha256(
        (schema + "\n" + nv.content_digest(frame, ())).encode()).hexdigest()[:16]


READERS = {"odds": lambda base: odds._archive(2025, base),
           "starter_change": lambda base: starter_change.archive(2025, base)}


@pytest.mark.parametrize(("kind", "reader"), sorted(EXPECTED))
def test_a_reader_of_every_poll_hands_back_the_frame_it_always_did(kind, reader, tmp_path):
    _store(kind, tmp_path)
    got = _digest(READERS[reader](tmp_path))
    assert got == EXPECTED[(kind, reader)], (
        f"{reader} read the {kind} store as a different frame ({got}) than it did when it held "
        f"its own query")


def test_the_two_readers_are_not_vacuous(tmp_path):
    """The premise: the stores above hold polls, and the readers return them. A fixture that
    wrote nothing readable would make every digest the digest of an empty frame."""
    _store("mixed", tmp_path)
    assert READERS["odds"](tmp_path).height == 12
    assert READERS["starter_change"](tmp_path).height == 12
