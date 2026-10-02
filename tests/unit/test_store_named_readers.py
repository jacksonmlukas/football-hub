"""The readers of `prop_lines`, `prop_log`, `pool_state` and the journal answer what they always did.

#400 took each of these tables' reads out of the module that wanted them and gave them a name in
`hub.store`, which owns the columns, the filters and the fresh-clone case. A change of where a
query lives may not change the frame (or the exception, or the printed report) the caller gets,
so this file was written before it: the `EXPECTED` digests were measured with each caller still
holding its own `store.sql`, on stores built under `tmp_path` in the states the real one has been
in -- a fresh clone, a populated store with another league and another season beside the one
asked for (which must not leak in), and a journal written before a column existed.

What is digested is the caller's own answer: the frame with its schema, the exception type and
text, a list of states, or everything a command prints. Row order is not compared where the
store gives none.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hub import store
from hub.contracts import PROP_LOG
from hub.fetch import nflverse as nv
from hub.fetch import odds, pool
from hub.models import props
from hub.season import journal

# Measured on 2026-10-02 with every caller still holding its own query.
EXPECTED: dict[tuple[str, str], str] = {
    ("prop_archive", "fresh"): "4e17175c474de257",
    ("prop_archive", "populated"): "b82584915ac56acf",
    ("props_polls", "fresh"): "d30080c0186c5f03",
    ("props_polls", "populated"): "5c6d4cc3a9895c57",
    ("props_polls_week", "populated"): "e7689b4bfe181829",
    ("prop_log_report", "fresh"): "3f4d5cebc6f626b6",
    ("prop_log_report", "populated"): "c0bb9c21ccb79cf0",
    ("pool_archived", "fresh"): "4f53cda18c2baa0c",
    ("pool_archived", "populated"): "0d406a45ca0be082",
    ("pool_archived_week", "populated"): "6d628b8c1e7a0458",
    ("journal_read", "fresh"): "1e1ece72049553f8",
    ("journal_read", "decisions"): "eaef1c708d099389",
    ("journal_read", "settled"): "e3a6100b64c511c5",
    ("journal_read", "old_rows"): "5279348f876c32cb",
    ("journal_read_week", "settled"): "e71c2181c48906d5",
}

T0 = dt.datetime(2025, 9, 7, 12)


def _value(dtype: Any, i: int, tag: str) -> Any:
    """A deterministic value of `dtype` for row `i`, different per `tag` so a leak is visible."""
    if dtype == pl.Utf8:
        return f"{tag}-{i}"
    if dtype == pl.Boolean:
        return i % 2 == 0
    if dtype == pl.Datetime:
        return T0 + dt.timedelta(hours=i)
    if dtype == pl.Int64:
        return i + 1
    if dtype == pl.Float64:
        return 0.25 * (i + 1)
    raise AssertionError(dtype)


def _rows(schema: dict[str, Any], n: int, tag: str, **fixed: Any) -> pl.DataFrame:
    return pl.DataFrame(
        {c: [fixed[c] if c in fixed else _value(t, i, tag) for i in range(n)]
         for c, t in schema.items()},
        schema=schema)


def _write(df, table, league, season, week, name, base):
    store.write(df, table, league, season, week, name=name, base=base)


def _prop_lines(base: Path) -> None:
    for week in (1, 2):
        _write(_rows(odds._PROP_SCHEMA, 3, f"w{week}", week=week), "prop_lines", "nfl", 2025,
               week, f"p{week}", base)
    _write(_rows(odds._PROP_SCHEMA, 2, "ncaa", week=1), "prop_lines", "ncaa", 2025, 1, "c", base)
    _write(_rows(odds._PROP_SCHEMA, 2, "old", week=1), "prop_lines", "nfl", 2024, 1, "o", base)


def _prop_log(base: Path) -> None:
    schema = dict(PROP_LOG.required)
    for week in (1, 2):
        part = _rows(schema, 4, f"w{week}", market="player_receptions", stat="receptions",
                     status="priced", side="over", position="WR")
        _write(part, "prop_log", "nfl", 2025, week, f"d{week}", base)
    _write(_rows(schema, 2, "ncaa", market="player_receptions", stat="receptions",
                 status="priced", side="over", position="WR"), "prop_log", "ncaa", 2025, 1, "c",
           base)


def _pool(base: Path) -> None:
    def state(season: int, week: int, n: int) -> pool.PoolState:
        entries = tuple(pool.Entry(index=i, alive=i % 2 == 0, used=(f"T{i}", "ZZ"),
                                   last_week=week - 1, last_teams=(f"T{i}",)) for i in range(n))
        return pool.PoolState(season=season, week=week, field_size=n * 10, pot=100.0 + week,
                              entries=entries)

    for week in (3, 4):
        pool.write_state(state(2026, week, 3), base, when=dt.datetime(2026, 9, 10 + week,
                                                                    9, tzinfo=dt.UTC))
    pool.write_state(state(2027, 9, 2), base, when=dt.datetime(2027, 9, 20, 9, tzinfo=dt.UTC))


def _journal(base: Path, *, outcomes: bool, drop: tuple[str, ...] = ()) -> None:
    schema = {c: t for c, t in journal.SCHEMA.items() if c not in drop}
    for week in (1, 2):
        for k in range(2):
            row = _rows(schema, 1, f"j{week}{k}", key=f"k{week}{k}", season=2026, week=week,
                        kind="weekly")
            _write(row, journal.TABLE, "nfl", 2026, week, f"k{week}{k}", base)
    _write(_rows(journal.SCHEMA, 1, "other", key="z", season=2025, week=1, kind="weekly"),
           journal.TABLE, "nfl", 2025, 1, "z", base)
    if outcomes:
        out = _rows(journal.OUTCOME_SCHEMA, 1, "o", key="k11")
        _write(out, journal.OUTCOME_TABLE, "nfl", 2026, 1, "k11", base)


def _frame_digest(frame: pl.DataFrame) -> str:
    schema = ";".join(f"{c}:{t}" for c, t in frame.schema.items())
    return schema + "\n" + nv.content_digest(frame.select(sorted(frame.columns)), ())


def _observe(call) -> str:
    """The caller's answer as text: a frame, a list, an exception, or what a command printed."""
    try:
        got = call()
    except Exception as e:
        text = f"raises {type(e).__name__}: {e}"
    else:
        text = _frame_digest(got) if isinstance(got, pl.DataFrame) else repr(got)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _report(base: Path) -> str:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = props.main(["--report", "--season", "2025", "--base", str(base)])
    lines = sorted((out.getvalue() + "\n--\n" + err.getvalue()).replace(str(base), "<b>")
                   .splitlines())
    return f"exit {code}\n" + "\n".join(lines)


def _case(reader: str, kind: str, base: Path) -> str:
    if reader.startswith("prop_archive"):
        if kind == "populated":
            _prop_lines(base)
        return _observe(lambda: odds._prop_archive(2025, base))
    if reader.startswith("props_polls"):
        if kind == "populated":
            _prop_lines(base)
        week = 2 if reader.endswith("_week") else None
        return _observe(lambda: props._polls(2025, week, base))
    if reader == "prop_log_report":
        if kind == "populated":
            _prop_log(base)
        return _observe(lambda: _report(base))
    if reader.startswith("pool_archived"):
        if kind == "populated":
            _pool(base)
        week = 4 if reader.endswith("_week") else None
        return _observe(lambda: pool.archived(2026, week=week, base=base))
    if reader.startswith("journal_read"):
        if kind != "fresh":
            _journal(base, outcomes=kind == "settled",
                     drop=("pool_state_digest", "plan_source") if kind == "old_rows" else ())
        week = 2 if reader.endswith("_week") else None
        return _observe(lambda: journal.read(2026, week, base))
    raise AssertionError(reader)


@pytest.mark.parametrize(("reader", "kind"), sorted(EXPECTED))
def test_a_named_reader_answers_what_its_caller_always_got(reader, kind, tmp_path):
    got = _case(reader, kind, tmp_path)
    assert got == EXPECTED[(reader, kind)], (
        f"{reader} on the {kind} store answered differently ({got}) than it did while its "
        f"caller held its own query")


def test_the_populated_stores_are_not_vacuous(tmp_path):
    """The premise: what the fixtures write is what the readers find, so a digest is a digest of
    rows and not of an empty frame that every state shares."""
    _prop_lines(tmp_path)
    _pool(tmp_path)
    _journal(tmp_path, outcomes=True)
    assert odds._prop_archive(2025, tmp_path).height == 6
    assert props._polls(2025, 2, tmp_path).height == 3
    assert len(pool.archived(2026, base=tmp_path)) == 2
    assert journal.read(2026, base=tmp_path).height == 4
