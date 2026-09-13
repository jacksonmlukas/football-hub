"""`hub.fetch.cached`: one cached-file fetcher, and a source of this shape is an adapter (#255).

Everything here drives the shared module through a third source built from a fixture -- a
dict of rows behind an in-memory transport -- so what is proved is proved of the module and
not of either adapter. The two real adapters' own suites (`test_fetch_nfeloqb.py`,
`test_fetch_pool.py`) still hold their nouns, their stamps and the cookie scan.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from hub.contracts import Contract, ContractViolation
from hub.fetch import bigten, cached, cfbd, nfeloqb, pool

# --- the guard has one home and four readers ------------------------------------------


def _never(*a, **k):                                         # pragma: no cover - must not run
    raise AssertionError("a test reached the network")


@pytest.mark.parametrize("door", [
    pytest.param(lambda: nfeloqb._http_get(nfeloqb.URL), id="nfeloqb"),
    pytest.param(lambda: pool._http_get("https://pool.example.test/api/pools/1",
                                        "s%3A" + "0f1e2d3c4b5a" * 4), id="pool"),
    pytest.param(lambda: cfbd._http_get("/games", {"year": 2026, "week": 1}, "a-key"),
                 id="cfbd"),
    pytest.param(lambda: bigten._get(bigten.PAGE), id="bigten"),
])
def test_each_fetcher_refuses_a_live_call_under_pytest_through_the_one_guard(door, monkeypatch):
    """The four doors to the network, each refusing with the shared module's exception --
    and each module's `LiveCallRefused` *is* that class, so a caller catching one catches
    the other. The transports are made fatal so a guard that let the call through fails
    loudly rather than reaching for a host."""
    import urllib.request

    import requests
    monkeypatch.setattr(requests, "get", _never)
    monkeypatch.setattr(urllib.request, "urlopen", _never)
    with pytest.raises(cached.LiveCallRefused) as e:
        door()
    assert "test_each_fetcher_refuses_a_live_call" in str(e.value)
    assert cached.LIVE_TEST_SUITE in str(e.value)


def test_the_four_modules_share_the_one_exception_and_the_one_suite_name():
    assert (nfeloqb.LiveCallRefused is pool.LiveCallRefused is cfbd.LiveCallRefused
            is bigten.LiveCallRefused is cached.LiveCallRefused)
    assert {m.LIVE_TEST_SUITE for m in (nfeloqb, pool, cfbd, bigten)} == {cached.LIVE_TEST_SUITE}


def test_the_guard_names_the_test_the_patch_and_the_file_that_shows_how():
    with pytest.raises(cached.LiveCallRefused) as e:
        cached.refuse_live_call("would fetch x", patch="_get", tests="tests/unit/test_x.py")
    said = str(e.value)
    assert said.startswith("tests/unit/test_fetch_cached.py::test_the_guard_names")
    assert "would fetch x" in said and "`_get`" in said and "tests/unit/test_x.py" in said


def test_the_golden_suite_is_exempt_and_no_test_at_all_is_exempt(monkeypatch):
    monkeypatch.setenv(cached.PYTEST_NODE_ENV, "tests/golden/test_golden.py::t (call)")
    cached.refuse_live_call("would fetch", patch="_get", tests="t")
    monkeypatch.delenv(cached.PYTEST_NODE_ENV)
    cached.refuse_live_call("would fetch", patch="_get", tests="t")


# --- the stamp --------------------------------------------------------------------------

def test_a_stamp_reads_back_with_captured_at_first_and_an_unreadable_one_is_empty(tmp_path):
    p = cached.write_stamp(tmp_path / "s.json", "2026-09-12T12:00:00", rows=3, url="u")
    assert list(json.loads(p.read_text())) == ["captured_at", "rows", "url"]
    assert cached.read_stamp(p) == {"captured_at": "2026-09-12T12:00:00", "rows": 3, "url": "u"}
    p.write_text("{not json")
    assert cached.read_stamp(p) == {}
    p.write_text("[1, 2]")
    assert cached.read_stamp(p) == {}, "a record is a dict; anything else is no record"
    assert cached.read_stamp(tmp_path / "absent.json") == {}


# --- a third source is one adapter ---------------------------------------------------

WIDGETS = Contract(name="widgets", required={"id": pl.Int64, "size": pl.Float64},
                   non_null=("id",), unique=("id",), min_rows=1,
                   verified_against_live=False)
FILE, STAMP = "widgets.json", "widgets.stamp.json"


def _paths(cache: Path | None) -> tuple[Path, Path]:
    root = Path(cache) if cache is not None else Path("/nowhere")
    return root / FILE, root / STAMP


class Widgets:
    """A third source of the shape: an in-memory transport, a parser through a contract,
    a cache with a stamp, and a report. Built from a fixture, adapted in one declaration."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.transport = lambda: json.dumps(self.rows)

    def parse(self, body: str) -> pl.DataFrame:
        return WIDGETS.validate(pl.DataFrame(json.loads(body), schema=WIDGETS.required))

    def refresh(self, ns) -> pl.DataFrame:
        self.calls += 1
        body = self.transport()
        got = self.parse(body)
        path, stamp = _paths(ns.cache)
        cached.atomic.write_text(path, body)
        cached.write_stamp(stamp, "2026-09-12T12:00:00", rows=got.height)
        return got

    def read(self, cache: Path | None) -> pl.DataFrame | None:
        path, _ = _paths(cache)
        return self.parse(path.read_text()) if path.exists() else None

    def captured_at(self, cache: Path | None) -> str | None:
        return cached.read_stamp(_paths(cache)[1]).get("captured_at")

    def report(self, st: pl.DataFrame, *, captured: str | None = None) -> list[str]:
        return [f"  widgets: {st.height} rows" + (f" (pulled {captured})" if captured else "")]

    def adapter(self) -> cached.Adapter[pl.DataFrame]:
        return cached.Adapter(
            prog="hub.fetch.widgets", description="Widgets.",
            what="the widgets", cached="the cached widgets", kept="the last-good widgets pulled",
            cache_flag="--cache", cache_help="where", refresh_help="pull", status_help="print",
            refresh_hint="run --refresh",
            cache_path=lambda c: _paths(c)[0],
            refresh=self.refresh, read=self.read, captured_at=self.captured_at,
            report=self.report,
            describe=lambda e: (f"the widgets were refused: {e}" if isinstance(e, ContractViolation)
                                else f"{type(e).__name__}: {e}"),
        )


@pytest.fixture
def widgets():
    return Widgets([{"id": 1, "size": 2.0}, {"id": 2, "size": 3.5}])


def test_a_third_source_is_one_adapter_driven_by_the_shared_cli(widgets, tmp_path, capsys):
    a = widgets.adapter()
    cache = str(tmp_path / "w")
    assert cached.run(a, ["--status", "--cache", cache]) == 1
    err = capsys.readouterr().err
    assert "the widgets unavailable" in err and "run --refresh" in err and "Traceback" not in err

    assert cached.run(a, ["--refresh", "--cache", cache]) == 0
    assert capsys.readouterr().out == "  widgets: 2 rows (pulled 2026-09-12T12:00:00)\n"
    assert widgets.calls == 1

    assert cached.run(a, ["--status", "--cache", cache]) == 0
    assert capsys.readouterr().out == "  widgets: 2 rows (pulled 2026-09-12T12:00:00)\n"
    assert widgets.calls == 1, "--status reads the cache and nothing else"


def test_serves_last_good_on_a_refused_frame_and_names_why(widgets, tmp_path, capsys):
    """The policy, tested once against the shared module: a pull the contract refuses is
    said on stderr with the contract's own words, the last-good file is served on stdout
    with its capture time, and the refused pull did not overwrite it."""
    a = widgets.adapter()
    cache = str(tmp_path / "w")
    cached.run(a, ["--refresh", "--cache", cache])
    capsys.readouterr()
    widgets.transport = lambda: json.dumps([{"id": 1, "size": 2.0}, {"id": 1, "size": 9.0}])
    assert cached.run(a, ["--refresh", "--cache", cache]) == 0
    out = capsys.readouterr()
    assert out.err == ("hub.fetch.widgets: the widgets were refused: widgets: id not unique "
                       "(1/2). NOTE: this contract was written from documentation and has "
                       "never been checked against a live response -- suspect the "
                       "declaration before the source; serving the last-good widgets pulled "
                       "2026-09-12T12:00:00\n")
    assert out.out == "  widgets: 2 rows (pulled 2026-09-12T12:00:00)\n"
    kept = widgets.read(Path(cache))
    assert kept is not None and kept["size"].to_list() == [2.0, 3.5]


def test_a_transport_failure_serves_last_good_with_the_failure_named(widgets, tmp_path, capsys):
    a = widgets.adapter()
    cache = str(tmp_path / "w")
    cached.run(a, ["--refresh", "--cache", cache])
    capsys.readouterr()

    def down():
        raise OSError("no route")
    widgets.transport = down
    assert cached.run(a, ["--refresh", "--cache", cache]) == 0
    out = capsys.readouterr()
    assert out.err.startswith("hub.fetch.widgets: OSError: no route; serving the last-good")
    assert "2 rows" in out.out


def test_a_drifted_cache_is_refused_beside_the_failure_not_served(widgets, tmp_path, capsys):
    a = widgets.adapter()
    cache = tmp_path / "w"
    cached.run(a, ["--refresh", "--cache", str(cache)])
    (cache / FILE).write_text(json.dumps([{"id": 1, "size": 2.0}, {"id": 1, "size": 2.0}]))
    capsys.readouterr()

    def down():
        raise OSError("no route")
    widgets.transport = down
    assert cached.run(a, ["--refresh", "--cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert "the widgets (OSError: no route; and the cached widgets unavailable" in err
    assert "not unique" in err
    # `--status` refuses the same cache the same way, as a sentence and not a traceback.
    assert cached.run(a, ["--status", "--cache", str(cache)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("hub.fetch.widgets: the cached widgets unavailable: ContractViolation")


def test_a_fresh_clone_with_nothing_cached_is_a_sentence_and_a_non_zero_exit(widgets, tmp_path,
                                                                             capsys):
    a = widgets.adapter()

    def down():
        raise OSError("no route")
    widgets.transport = down
    assert cached.run(a, ["--refresh", "--cache", str(tmp_path / "none")]) == 1
    err = capsys.readouterr().err
    assert err == "hub.fetch.widgets: the widgets unavailable: RuntimeError: OSError: no route\n"


def test_an_adapter_can_add_a_branch_and_arguments_without_copying_the_tree(widgets, tmp_path,
                                                                            capsys):
    """The pool adapter's `--payload FILE`: an argument hook and a branch that runs first
    and, when it handled the call, is the whole call."""
    seen = []
    a = widgets.adapter()
    a = replace(a, arguments=lambda ap: ap.add_argument("--payload", default=None),
                before=lambda ns: (seen.append(ns.payload) or 7) if ns.payload else None)
    assert cached.run(a, ["--payload", "x.json", "--cache", str(tmp_path)]) == 7
    assert seen == ["x.json"] and widgets.calls == 0
    assert cached.run(a, ["--refresh", "--cache", str(tmp_path / "w")]) == 0
    assert widgets.calls == 1


def test_the_report_after_a_live_pull_carries_the_stamp_only_where_the_adapter_says(
        widgets, tmp_path, capsys):
    a = replace(widgets.adapter(), stamped_after_refresh=False)
    assert cached.run(a, ["--refresh", "--cache", str(tmp_path / "w")]) == 0
    assert capsys.readouterr().out == "  widgets: 2 rows\n"
