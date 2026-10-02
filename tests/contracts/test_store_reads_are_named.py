"""Reading the store's tables is `hub.store`'s business, and nobody else's.

Readers outside the store used to ask its tables their questions themselves -- the `lines`
polls three ways (#399), `prop_lines`, `prop_log`, `pool_state` and the journal each in the module
that wanted them (#400) -- and each repeated the fresh-clone guard and wrote the empty case by
hand. Every table now has a name in `hub.store` that owns its columns, its filters and its
fresh-clone answer. This holds them there: no module under `src/hub` other than the store calls
`store.sql`, on any table.

`store.sql` stays public -- `inspect` and the tooling use it from outside `src/hub`, and a query
the store has no name for yet has to be askable -- which is why this is a scan and not a rename.
It was scoped to `lines` when #399 wrote it; `TABLE` is `None` now that #400 has widened it.

The scan is `sql_callers`, in `tests/store_routing.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path

from store_routing import UNRESOLVED, sql_callers, sql_calls

SRC = Path(__file__).resolve().parents[2] / "src" / "hub"

TABLE = None            # every table; #399 held it to "lines"


def test_nothing_outside_the_store_queries_the_catalog():
    found = sql_callers(SRC, TABLE)
    assert not found, (
        "these modules query the catalog through `store.sql`, which is the store's to name: "
        + "; ".join(f"src/hub/{p} line(s) {lines}" for p, lines in sorted(found.items()))
        + ". Read it through the store's named reader for that table (`store.lines`, "
          "`store.lines_as_of`, `store.prop_lines`, `store.prop_log`, `store.pool_state`, "
          "`store.journal`, `store.journal_outcomes`), or add one.")


def test_the_scan_sees_the_store_reading_its_own_table():
    """The premise: the store does query `lines` through `sql`, and the scan reads it when it
    is asked to. A scan that matched no call at all would make the test above vacuous."""
    mine = [q for _, q in sql_calls(ast.parse((SRC / "store.py").read_text()))]
    assert any("lines" in q and q != UNRESOLVED for q in mine)


def test_a_planted_query_on_lines_is_found_and_a_neighbouring_table_is_not(tmp_path):
    """Rule 18. The shapes a query takes -- a literal, an f-string, a name bound to either, the
    name imported straight from the store -- planted in a tree shaped like the real one."""
    (tmp_path / "store.py").write_text("def sql(q): ...\n")
    plant = {
        "a_literal.py": "from hub import store\nstore.sql('SELECT * FROM lines')\n",
        "an_fstring.py": "from hub import store\nt = 'lines'\nstore.sql(f'SELECT * FROM {t}')\n",
        "a_bound_name.py": ("from hub import store\nq = 'SELECT 1 FROM lines WHERE a = ?'\n"
                            "store.sql(q, params=[1])\n"),
        "imported.py": "from hub.store import sql\nsql('SELECT * FROM lines')\n",
    }
    for name, body in plant.items():
        (tmp_path / name).write_text(body)
    (tmp_path / "neighbour.py").write_text(
        "from hub import store\nstore.sql('SELECT * FROM prop_lines')\n")
    found = set(sql_callers(tmp_path, "lines"))
    assert "neighbour.py" not in found, "`prop_lines` is not `lines`"
    assert set(plant) <= found, f"a shape of query on `lines` went unseen: {set(plant) - found}"
    # With every table in scope, the neighbour is caught too -- the widening #400 makes.
    assert "neighbour.py" in sql_callers(tmp_path, None)
