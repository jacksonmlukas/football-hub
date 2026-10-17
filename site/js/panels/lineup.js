// The lineup call (#352) and what changed since the last regen (#355).

import { ago, esc, fmt } from "../format.js";
import { STATE, staleNote } from "../state.js";

// Who a name is, where the reader cannot ask: the name with its position and NFL team beside
// it (#352). A second reader has no league context to supply either, and a bare name is the
// shortcut that is easy to take by accident. The name is looked up on the published roster
// rows; one the rows do not carry still renders, with `?` where the fact should be, so the
// gap is visible rather than silent.
function whoIs(name, byName) {
  const r = byName.get(name);
  return `<span class="who"><b>${esc(name)}</b> <span class="thin">${esc(r?.pos ?? "?")} &middot; ${esc(r?.nfl_team ?? "?")}</span></span>`;
}

// `sit` and `start` are two unordered lists; a swap is one out and one in at the same position.
// Pair by position first, then the leftovers in order (a flex slot swaps across positions), and
// a side with nothing to pair against is null.
function pairSwaps(sit, start, byName) {
  const posOf = n => byName.get(n)?.pos ?? "";
  const ins = [...start], rest = [], pairs = [];
  for (const out of sit) {
    const i = posOf(out) ? ins.findIndex(n => posOf(n) === posOf(out)) : -1;
    if (i >= 0) pairs.push([out, ins.splice(i, 1)[0]]); else rest.push(out);
  }
  while (rest.length || ins.length) pairs.push([rest.shift() ?? null, ins.shift() ?? null]);
  return pairs;
}

// "What changed since the last regen" (#355), under the lineup call. The publisher keeps the
// prior sync's decision beside this one (`ros.previous`: `generated_at`, `start`, `sit`,
// `withheld`) and this only diffs the two -- which run was previous is the publisher's fact.
// Four cases, each a sentence: moves, no change, no earlier decision, and a prior too old to be
// "the last regen". The line has its own stale state: when the roster artifact is stale or Kept
// it describes the run it actually came from, in the stale state with that run's stamp, and
// never reads as a current claim. A missing or stale prior degrades this line, never the call.
// More than this many days between the two syncs and a regen was missed in between (the roster
// regenerates Wednesday and Saturday: three and four days apart). Chosen, not fitted.
const PREVIOUS_MAX_GAP_DAYS = 5;
export function lineupChanges(ros, stale, byName) {
  const prev = ros?.previous;
  const lists = ["start", "sit", "withheld"];
  const usable = prev && typeof prev === "object" && lists.every(k => Array.isArray(prev[k]));
  const t0 = usable ? Date.parse(prev.generated_at) : NaN, t1 = Date.parse(ros?.generated_at);
  const gapDays = (t1 - t0) / 864e5;
  const wrap = (cls, inner) => `<p class="lineup-changes ${cls}">${inner}</p>`;
  const stamp = stale
    ? ` <span class="stamp">describes the run synced ${esc(ago(ros?.generated_at))}</span>` : "";
  const cls = stale ? STATE.stale : STATE.frozen;
  if (!usable) return wrap(cls, `No earlier decision to compare.${stamp}`);
  // GUARD changes-line-never-diffs-against-a-stale-prior [contracts/test_dashboard_escapes.py::test_a_prior_decision_too_old_to_be_the_last_regen_is_not_diffed_against]: a prior that is not the last regen is said so in the stale state, not diffed.
  if (!(gapDays > 0 && gapDays <= PREVIOUS_MAX_GAP_DAYS)) {
    return staleNote({ at: prev.generated_at,
      reason: "the earlier lineup call is not from the last regen, so what changed is not shown" });
  }
  // /GUARD
  const diff = (a, b) => b.filter(n => !a.includes(n));
  const named = ns => ns.map(n => whoIs(n, byName)).join(", ");
  const cur = { start: ros.start ?? [], sit: ros.sit ?? [], withheld: ros.withheld ?? [] };
  const parts = [];
  const ins = diff(prev.start, cur.start), outs = diff(prev.sit, cur.sit);
  if (ins.length) parts.push(`${named(ins)} in`);
  if (outs.length) parts.push(`${named(outs)} out`);
  // Called last week and not at all this week -- not a name already shown as in or out above.
  const callNow = [...cur.start, ...cur.sit];
  const dropped = [...prev.start, ...prev.sit].filter(n => !callNow.includes(n));
  if (dropped.length) parts.push(`${named(dropped)} no longer in the call`);
  const barred = diff(prev.withheld, cur.withheld), cleared = diff(cur.withheld, prev.withheld);
  if (barred.length) parts.push(`${named(barred)} now withheld`);
  if (cleared.length) parts.push(`${named(cleared)} no longer withheld`);
  const day = new Date(t0).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" });
  const said = parts.length ? `Since ${esc(day)}: ${parts.join("; ")}.` : `No change since ${esc(day)}.`;
  return wrap(cls, said + stamp);
}

// The lineup call, kept apart from the DOM so a test can import it: the gain at display
// size, the two totals under it, who was withheld, then the swaps and the caveat. A gain under
// the threshold and a null gain each say so in the same slot -- the headline never blanks.
export function lineupHeadline(ros, stale) {
  const rows = Array.isArray(ros?.rows) ? ros.rows : [];
  // Keyed on player name: two rostered players sharing a name would collide here.
  const byName = new Map(rows.map(r => [r.player, r]));
  const gain = ros?.gain;
  const changed = lineupChanges(ros, stale, byName);
  const totals = (gain != null && ros.set_total != null && ros.optimal_total != null)
    ? `<p class="lineup-totals">set lineup ${fmt(ros.set_total, 1)} &rarr; best available ${fmt(ros.optimal_total, 1)}</p>`
    : "";
  // Withheld players come before the swaps: the recommendation is only trustworthy if the
  // reader can see who was taken out of it. The unconfirmed state, as a chip's worth of prose.
  const heldHtml = (ros?.withheld ?? []).map(n => whoIs(n, byName)).join(", ");
  const chip = heldHtml
    ? `<p class="${STATE.unconfirmed}">Withheld as unavailable: ${heldHtml}. Not startable: ESPN's
         injury field does not carry suspensions; its games projection does, and these are short
         of a full slate. They are excluded from the lineup below.</p>`
    : "";
  if (gain == null) {
    return `<p class="lineup-call empty">No lineup could be filled from the available players.</p>` + changed + chip;
  }
  if (!(gain > 0.05)) {
    return `<p class="lineup-call">The set lineup is already the best available one.</p>` + changed + totals + chip;
  }
  const side = n => n == null ? "&mdash;" : whoIs(n, byName);
  const swaps = pairSwaps(ros.sit ?? [], ros.start ?? [], byName)
    .map(([out, inn]) => `<li>${side(out)}<span class="arrow">&rarr;</span>${side(inn)}</li>`).join("");
  return `<div class="lineup-gain">+${fmt(gain, 1)}<span class="unit"> pts a week</span></div>`
    + changed + totals + chip + `<ul class="swaps">${swaps}</ul>`
    + `<p class="legend">These projections are season-long and do not know who is hurt this week,
         so settle it at lock.</p>`;
}
