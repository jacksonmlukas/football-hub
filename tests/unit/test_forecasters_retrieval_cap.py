"""The retrieval arm's search cap (#511), standing alone.

`retrieval-search-cap` is a declared guard, and `tests/contracts/test_guards_are_load_bearing.py`
proves it by excising it and running this file against the mutant with only the mutant's `src`
on the path -- so this file imports nothing from `tests/` (the shared slate and packet
fixtures live there). The rest of the arm is `test_forecasters_retrieval.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hub.fetch import tavily
from hub.forecasters import retrieval

RUN_START = datetime(2026, 10, 18, 13, 31, tzinfo=UTC)


class Counting:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, days: int, max_results: int) -> tavily.Search:
        self.queries.append(query)
        return tavily.Search(query, (), 1)


def _packet() -> Any:
    game = SimpleNamespace(home_team="KC", away_team="BUF")
    return cast(Any, SimpleNamespace(game=game, run_start=RUN_START))


def test_a_query_list_over_the_cap_is_refused_before_any_search(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(retrieval, "QUERY_TEMPLATES",
                        (*retrieval.QUERY_TEMPLATES, "{team} beat writer"))
    client = Counting()
    r = retrieval.Retriever(client, full_names={}, state_path=tmp_path / "t.json",
                            season_credit_cap=1000, clock=lambda: RUN_START)
    with pytest.raises(retrieval.RetrievalFailed, match="exceed the cap") as exc:
        r.retrieve(_packet(), snapshot_id=None)
    assert client.queries == [] and exc.value.credits == 0
    assert retrieval.credits_used(tmp_path / "t.json") == 0


def test_the_fixed_queries_fit_the_cap_exactly():
    assert len(retrieval.queries_for(_packet(), {})) == retrieval.SEARCHES_PER_GAME == 4
