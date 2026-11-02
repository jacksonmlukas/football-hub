// The This week view's own parts (#484): the heading that states the decision, the best lineup
// drawn 20 ways, and the pipeline health that stands in for the footer on this view.
//
// Pure and DOM-free so a test can import it. Nothing here computes a number the artifacts did not
// publish: the dots are the published quantiles, the line is the published `mu`, the heading's
// count is the length of the published `swaps`. The only arithmetic is placing a value on the
// drawing's axis, which is a scale and not a fact. Every string off a feed goes through `esc`.

import { esc, fmt } from "../format.js";
import { chipFrozen, chipStale, chipUnconfirmed } from "../state.js";

// The shared axis of the 20-dot plot, in fantasy points. The same for every row so two rows can
// be read against each other; a value past the end sits on the end. A drawing choice, stated.
const AXIS_MAX = 40;
// Under this a gain is "already the best" (the same line `lineupHeadline` draws its call at).
const GAIN_THRESHOLD = 0.05;

const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const word = n => (Number.isInteger(n) && n >= 0 && n < WORDS.length ? WORDS[n] : String(n));

// "Week 6: make two changes", or "Week 6: your lineup is already the best one" when the gain is
// under the existing threshold. A gap and an unfillable lineup say so in the same slot. Plain
// text (the caller assigns it with `textContent`), so nothing here is markup.
export function weekHeading(ros, man) {
  const wk = ros?.week ?? man?.week ?? "?";
  const lead = `Week ${wk}: `;
  if (ros?.incomplete && typeof ros.incomplete === "object") {
    const empty = Array.isArray(ros.incomplete.empty_slots) ? ros.incomplete.empty_slots.length : 0;
    return lead + (empty === 1 ? "fill the empty slot" : `fill ${word(empty)} empty slots`);
  }
  if (ros?.gain == null) return lead + "no lineup could be filled";
  const n = Array.isArray(ros.swaps) ? ros.swaps.length : 0;
  const byRank = ros.lineup_rule?.source === "ecr";
  const already = byRank ? n === 0 : !(ros.gain > GAIN_THRESHOLD);
  if (already) return lead + "your lineup is already the best one";
  return lead + `make ${word(n)} change${n === 1 ? "" : "s"}`;
}

// A finite number placed on the axis, as a one-decimal string so a hostile value cannot reach
// the markup; null where the value is not a finite number.
const place = (v, width) => {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return (Math.min(Math.max(v, 0), AXIS_MAX) / AXIS_MAX * width).toFixed(1);
};

// The 20-dot quantile plot for one player: the published quantiles as dots, the projection as a
// line, on the shared 0-40 axis. A player with no distribution says why in the plot's place.
function plot(row) {
  const q = Array.isArray(row.quantiles) ? row.quantiles : null;
  if (!q) {
    return `<p class="bl-none thin">${esc(row.quantiles_reason ?? "no distribution published")}</p>`;
  }
  const W = 400;
  const dots = q.map(v => place(v, W)).filter(x => x !== null)
    .map(x => `<circle class="bl-dot" cx="${x}" cy="9" r="3"/>`).join("");
  const mu = place(row.mu, W);
  const line = mu === null ? "" : `<line class="bl-mu" x1="${mu}" x2="${mu}" y1="0" y2="18"/>`;
  const ticks = [0, 10, 20, 30, 40].map(t => {
    const x = place(t, W);
    return `<line class="bl-tick" x1="${x}" x2="${x}" y1="12" y2="18"/>`;
  }).join("");
  return `<svg class="bl-plot" viewBox="0 0 ${W} 18" preserveAspectRatio="none" role="img"
      aria-label="${esc(row.player)}: 20 weekly outcomes from 0 to ${AXIS_MAX} points, projection ${fmt(row.mu, 1)}">`
    + `<line class="bl-axis" x1="0" x2="${W}" y1="9" y2="9"/>${ticks}${dots}${line}</svg>`;
}

const lineRow = (row, incoming) => {
  const isNew = incoming.has(row.player);
  return `<li class="bl-row${isNew ? " bl-incoming" : ""}">`
    + `<span class="bl-pos">${esc(row.pos ?? "?")}</span>`
    + `<span class="bl-name"><b>${esc(row.player)}</b> <span class="thin">${esc(row.nfl_team ?? "?")}</span>`
    + `${isNew ? ' <span class="bl-new">in</span>' : ""}</span>`
    + `<span class="bl-proj">${row.projected ? fmt(row.mu, 1) : "—"}</span>`
    + `<span class="bl-dots">${plot(row)}</span></li>`;
};

// "Best lineup, 20 ways": one row per player in the best lineup (`best_start`), incoming players
// tinted and marked, the bench folded into a closed `<details>`, and the unprojected players
// listed with the reason the artifact gives for each.
export function bestLineupBody(ros) {
  const rows = Array.isArray(ros?.rows) ? ros.rows : [];
  const incoming = new Set((Array.isArray(ros?.swaps) ? ros.swaps : [])
    .map(s => s?.in).filter(n => typeof n === "string"));
  const best = rows.filter(r => r.best_start === true);
  const bench = rows.filter(r => r.best_start !== true && r.projected);
  const reasons = ros?.unprojected_reasons && typeof ros.unprojected_reasons === "object"
    ? ros.unprojected_reasons : {};
  const gone = (Array.isArray(ros?.unprojected) ? ros.unprojected : []);
  const axis = `<div class="bl-axis-head" aria-hidden="true"><span>0</span><span>10</span>`
    + `<span>20</span><span>30</span><span>${AXIS_MAX} pts</span></div>`;
  const legend = `<p class="legend">Each dot is one of 20 equally likely weeks for the player; the `
    + `vertical line is the projection.</p>`;
  const head = best.length
    ? `${axis}<ul class="bl-list">${best.map(r => lineRow(r, incoming)).join("")}</ul>`
    : `<p class="empty">No player in the best lineup has a projection.</p>`;
  const benchHtml = bench.length
    ? `<details class="bl-bench"><summary>Bench (${bench.length})</summary>`
      + `<ul class="bl-list">${bench.map(r => lineRow(r, incoming)).join("")}</ul></details>`
    : "";
  const noProj = gone.length
    ? `<p class="legend">Not projected:</p><ul class="bl-unproj">`
      + gone.map(n => `<li><b>${esc(n)}</b> <span class="thin">${esc(reasons[n] ?? "no reason given")}</span></li>`).join("")
      + `</ul>`
    : "";
  return head + legend + benchHtml + noProj;
}

// Pipeline health: every manifest artifact as a state chip with its age. A healthy artifact is
// frozen (committed, and it carries its age); a stale one is stale and says how old; one the page
// could not fetch is unconfirmed. `records` is `statusRecords(man)`: whole records, so the reason
// a chip is not frozen sits beside it. Replaces the footer's status line on this view.
export function healthBody(records) {
  const items = records.map(r => {
    const chip = !r.fetched ? chipUnconfirmed("not fetched")
      : r.stale ? chipStale(r.age, "stale")
      : chipFrozen(r.age);
    const why = !r.fetched || r.stale ? ` <span class="thin">${esc(r.reason ?? "no reason given")}</span>` : "";
    // The stale chip carries its age inside it and the frozen chip is labelled with it; the
    // unconfirmed chip says "not fetched", so the age rides beside it.
    const age = !r.fetched ? ` <span class="thin">${esc(r.age)}</span>` : "";
    return `<li><b>${esc(r.name)}</b> ${chip}${age}${why}</li>`;
  });
  return `<ul class="health">${items.join("")}</ul>`;
}
