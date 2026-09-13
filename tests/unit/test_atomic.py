"""`hub.atomic`: a writer that dies mid-write leaves the previous file served (#258)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from hub import atomic


class Died(RuntimeError):
    """The process going away with half the bytes on disk, as an exception."""


def _partials(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.name.endswith(atomic.SUFFIX)]


def test_an_interrupted_write_leaves_the_previous_file_and_no_partial(tmp_path):
    """The acceptance criterion in one test: bytes partly on disk, then death. The file the
    reader sees is the one from before; the partial is gone, not lying beside it."""
    target = tmp_path / "state.json"
    target.write_text('{"captured_at": "before"}')
    with pytest.raises(Died):
        with atomic.replacing(target) as scratch:
            with scratch.open("w") as fh:
                fh.write('{"captured_at": "af')
                fh.flush()
                assert scratch.stat().st_size > 0, "the partial has to be on disk to count"
                raise Died()
    assert target.read_text() == '{"captured_at": "before"}'
    assert _partials(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_a_clean_write_replaces_the_file_whole_and_leaves_no_scratch(tmp_path):
    target = tmp_path / "deep" / "er" / "state.json"      # parents made by the helper
    atomic.write_text(target, "one")
    assert target.read_text() == "one"
    atomic.write_text(target, "two")
    assert target.read_text() == "two"
    assert sorted(p.name for p in target.parent.iterdir()) == ["state.json"]


def test_bytes_and_parquet_land_the_same_way(tmp_path):
    b = atomic.write_bytes(tmp_path / "blob.csv", b"a,b\n1,2\n")
    assert b.read_bytes() == b"a,b\n1,2\n"
    df = pl.DataFrame({"x": [1, 2]})
    p = atomic.write_parquet(df, tmp_path / "part.parquet")
    assert pl.read_parquet(p).equals(df)
    assert _partials(tmp_path) == []


def test_an_interrupted_parquet_write_serves_the_old_partition(tmp_path, monkeypatch):
    """The store's case: polars writes the footer last, so a truncated parquet is one the
    reader cannot open at all. After the interruption the old partition still reads."""
    p = tmp_path / "part.parquet"
    atomic.write_parquet(pl.DataFrame({"x": [1]}), p)

    def die(self, path, *a, **k):
        Path(path).write_bytes(b"PAR1\x00\x00")               # a header and no footer
        raise Died()

    monkeypatch.setattr(pl.DataFrame, "write_parquet", die)
    with pytest.raises(Died):
        atomic.write_parquet(pl.DataFrame({"x": [2]}), p)
    monkeypatch.undo()
    assert pl.read_parquet(p)["x"].to_list() == [1]
    assert _partials(tmp_path) == []


@pytest.mark.parametrize("name, payload, die_in", [
    ("write_text", "after", "write_text"),
    ("write_bytes", b"after", "write_bytes"),
])
def test_the_conveniences_go_through_the_scratch_too(tmp_path, monkeypatch, name, payload, die_in):
    """`write_text` and `write_bytes` are the routed spellings every fetcher uses, so each
    has to die the same way `replacing` does: the pathlib call they wrap is made to write
    half and raise, and the target is still the old file with no partial beside it."""
    target = tmp_path / "state.json"
    target.write_bytes(b"before")
    real = getattr(Path, die_in)

    def die(self, data, *a, **k):
        real(self, data[:2], *a, **k)
        raise Died()

    monkeypatch.setattr(Path, die_in, die)
    with pytest.raises(Died):
        getattr(atomic, name)(target, payload)
    monkeypatch.undo()
    assert target.read_bytes() == b"before"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]
