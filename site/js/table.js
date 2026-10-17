// The table helper: every cell and header in every panel is built, and escaped, here.

import { esc } from "./format.js";

// Escapes by default, so a column cannot forget. Getters return *text*, never markup --
// which is why the em dash below is the character rather than `&mdash;`. Relying on each
// call site to remember is how one of a dozen columns ends up unescaped.
//
// The marker below claims the single place and not the escaping, because the block is the
// smallest span that still parses and it carries the markup as well as the `esc(...)` calls:
// deleting it stops the table rendering at all, so its red would prove the helper throws.
// That the escaping is load-bearing is proved separately, by
// `test_a_table_that_stops_escaping_its_cells_is_caught`, which unwraps only the two `esc`
// calls and leaves the markup intact -- the change a new column would make by forgetting.
export function table(cols, rows) {
  // GUARD table-builds-every-cell-in-one-place [contracts/test_dashboard_escapes.py]: every cell and header the table renders is built by these two statements, so there is one place to escape at rather than one per column.
  const head = cols.map(c => `<th>${esc(c.label)}</th>`).join("");
  const body = rows.map(r =>
    `<tr>${cols.map(c => `<td class="num ${c.cls ? c.cls(r) : ""}">${esc(c.get(r))}</td>`).join("")}</tr>`
  ).join("");
  // /GUARD
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
