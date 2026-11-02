// The Study view's body (#487): the LLM-vs-market leaderboard, or the designed empty state that
// stands in until `hub.forecasters.leaderboard` (#453) has published one.
//
// The page computes no score. Every estimate, interval and count drawn here is a field of the
// artifact (`contestants[].log_loss_diff`, `.brier_diff`, `.scorable`, `.missing`); what this file
// adds is layout: a shared horizontal scale for the whiskers, an order (sorted by the artifact's
// own estimate) and a rank (that order's position). The Delta is the contestant minus the betting
// market on the same games, so zero is the betting market and positive is worse than it.

import { esc } from "../format.js";
import { chipUnconfirmed } from "../state.js";

// The sentence the whole board carries while the artifact says it is interim. `studyBody` has no
// path around it: the only way to omit it is the artifact saying `interim: false`.
export const INTERIM_LABEL = "Interim: descriptive, not a test";

// The contestants as the pre-registration groups them (docs/prereg/llm-forecasters.md, Contestants,
// restated 2026-10-22 to twelve by #511). A static list on purpose: the leaderboard artifact only
// exists once a week has been scored, and the empty state exists for the weeks before that, so the
// roster has no published source yet except the prereg itself. The groups carry no result and no
// verdict -- only who is entered -- so nothing here can disagree with the artifact once it lands.
export const CONTESTANTS = [
  { group: "Four LLMs, blind", n: 4, note: "see the info packet without the betting line" },
  { group: "Four LLMs, line-aware", n: 4, note: "the same packet plus the betting market's probability and spread" },
  { group: "LLM + retrieval", n: 1, note: "paired with one line-aware LLM, plus news search (exploratory)" },
  { group: "Hub model", n: 1, note: "football-hub's own pick'em model (exploratory)" },
  { group: "Home constant", n: 1, note: "a fixed historical home win rate, every game" },
  { group: "Coin flip", n: 1, note: "50% for every game" },
];

// The three steps to the first score, each with the issue that builds it.
export const STEPS = [
  { n: 448, text: "Fill the v0 checklist and tag prereg-llm-v0, so the forecasts can start." },
  { n: 451, text: "Report the burn-in: whether the study can tell a model apart from the betting market." },
  { n: 453, text: "Score each finished week and publish this leaderboard." },
];

export function emptyStudy() {
  const who = CONTESTANTS.map(c =>
    `<li><b>${esc(c.group)}</b> <span class="lb-n">${c.n}</span><span class="lb-note">${esc(c.note)}</span></li>`
  ).join("");
  const steps = STEPS.map(s => `<li>${esc(s.text)} <span class="lb-issue">#${s.n}</span></li>`).join("");
  const total = CONTESTANTS.reduce((a, c) => a + c.n, 0);
  return `<p class="lb-chips">${chipUnconfirmed("not started")}</p>`
    + `<p class="lb-question">Can a language model forecast a college football game better than the betting market?</p>`
    + `<h4 class="lb-h">${total} contestants, one benchmark</h4><ul class="lb-list">${who}`
    + `<li class="lb-market"><b>Market (benchmark)</b><span class="lb-note">the de-vigged moneyline at forecast time; not a contestant</span></li></ul>`
    + `<h4 class="lb-h">Three steps to the first score</h4><ol class="lb-steps">${steps}</ol>`
    + `<p class="empty">Nothing is scored yet, so there is no board to show.</p>`;
}

// "llm1:line_aware" -> "line_aware"; the baselines carry no condition. The suffix is the
// condition's own spelling (hub.forecasters.packet.Condition).
export const conditionOf = id => {
  const m = /:(blind|line_aware)$/.exec(String(id ?? ""));
  return m ? m[1] : null;
};
const BASELINES = new Set(["home_constant", "coin_flip"]);

export function labelOf(id) {
  const s = String(id ?? "");
  const cond = conditionOf(s);
  const base = cond ? s.slice(0, -(cond.length + 1)) : s;
  if (base === "hub") return "Hub model (exploratory)";
  if (base === "home_constant") return "Home constant";
  if (base === "coin_flip") return "Coin flip";
  const m = /^llm(\d+)(\+retrieval)?$/.exec(base);
  return m ? `LLM ${m[1]}${m[2] ? " + retrieval" : ""}` : s;
}

const num = v => typeof v === "number" && Number.isFinite(v);
const hasIv = iv => Boolean(iv) && num(iv.estimate);

// "+0.012 [−0.004, +0.031]": the artifact's three numbers, signed, nothing else.
const sgn = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(3);
const cell = iv => hasIv(iv)
  ? `${sgn(iv.estimate)}${num(iv.lo) && num(iv.hi) ? ` [${sgn(iv.lo)}, ${sgn(iv.hi)}]` : ""}`
  : "—";

const W = 160, H = 16;
const plot = (inner, label) =>
  `<svg class="lb-plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${inner}</svg>`;
const zeroAt = m => {
  const x = (W / 2).toFixed(1);
  return `<line class="lb-zero" x1="${x}" x2="${x}" y1="0" y2="${H}"/>`;
};

// One dot and whisker about zero. `m` is the half-width of the shared scale.
function whisker(iv, m) {
  if (!hasIv(iv)) return plot(zeroAt(m), "no estimate");
  const x = v => (((v + m) / (2 * m)) * W).toFixed(1);
  const bar = num(iv.lo) && num(iv.hi)
    ? `<line class="lb-whisker" x1="${x(iv.lo)}" x2="${x(iv.hi)}" y1="${H / 2}" y2="${H / 2}"/>` : "";
  return plot(`${zeroAt(m)}${bar}<circle class="lb-dot" cx="${x(iv.estimate)}" cy="${H / 2}" r="3.5"/>`, cell(iv));
}

// The half-width of the shared axis: the widest bound any row draws, with a margin. A layout
// choice, so the dots keep their place when the condition toggles.
function halfWidth(rows) {
  let m = 0;
  for (const r of rows) {
    for (const v of [r.log_loss_diff?.estimate, r.log_loss_diff?.lo, r.log_loss_diff?.hi]) {
      if (num(v)) m = Math.max(m, Math.abs(v));
    }
  }
  return m > 0 ? m * 1.1 : 0.05;
}

const DIR = "lower is better";
// GUARD each-metric-header-carries-its-better-direction [contracts/test_dashboard_study.py::test_each_metric_header_carries_its_better_direction]: every metric column names which way is better, in the header itself.
function head() {
  return `<thead><tr><th>#</th><th>Contestant</th><th class="num">Resolved</th>`
    + `<th class="num">Missing ↓ <span class="lb-dir">(${DIR})</span></th>`
    + `<th class="num" title="the contestant's log loss minus the betting market's, same games">Δ log loss ↓ <span class="lb-dir">(${DIR})</span></th>`
    + `<th>vs market, 95% interval</th>`
    + `<th class="num" title="the contestant's Brier score minus the betting market's, same games">Δ Brier ↓ <span class="lb-dir">(${DIR})</span></th></tr></thead>`;
}
// /GUARD

function row(r, rank, m, prelim) {
  const base = BASELINES.has(String(r.contestant).split(":")[0]);
  const cls = ["lb-row", base ? "lb-base" : "", prelim ? "lb-prelim" : ""].filter(Boolean).join(" ");
  const mark = prelim ? ` ${chipUnconfirmed("preliminary")}` : "";
  return `<tr class="${cls}" data-contestant="${esc(r.contestant)}"><td class="num">${rank ?? "—"}</td>`
    + `<td>${esc(labelOf(r.contestant))}${mark}</td>`
    + `<td class="num">${esc(r.scorable ?? "—")}</td><td class="num">${esc(r.missing ?? "—")}</td>`
    + `<td class="num">${esc(cell(r.log_loss_diff))}</td><td>${whisker(r.log_loss_diff, m)}</td>`
    + `<td class="num">${esc(cell(r.brier_diff))}</td></tr>`;
}

// The benchmark: the zero every other row is measured from, so it has no figures of its own.
function marketRow(m) {
  return `<tr class="lb-row lb-market-row"><td class="num">—</td><td>Market (benchmark)</td>`
    + `<td class="num">—</td><td class="num">—</td><td class="num">zero</td>`
    + `<td>${plot(zeroAt(m), "the betting market, zero by definition")}</td><td class="num">zero</td></tr>`;
}

// One condition's table: the betting market, then the ranked rows by the artifact's own estimate, then
// the rows under the minimum game count (listed, unranked, marked preliminary). The baselines
// (home constant, coin flip) are drawn greyed. Rows with no condition (hub, baselines) appear in
// both tables.
function conditionTable(rows, min, cond, m) {
  const mine = rows.filter(r => [null, cond].includes(conditionOf(r.contestant)));
  const prelim = r => min !== null && !(r.log_loss_diff?.games >= min);
  const byEst = (a, b) => (a.log_loss_diff?.estimate ?? Infinity) - (b.log_loss_diff?.estimate ?? Infinity);
  const ranked = mine.filter(r => !prelim(r) && hasIv(r.log_loss_diff)).sort(byEst);
  const rest = mine.filter(r => !ranked.includes(r)).sort(byEst);
  return `<div class="scroll"><table class="lb-table">${head()}<tbody>${marketRow(m)}`
    + ranked.map((r, i) => row(r, i + 1, m, false)).join("")
    + rest.map(r => row(r, null, m, prelim(r))).join("") + `</tbody></table></div>`;
}

export function studyBody(lb) {
  const rows = Array.isArray(lb?.contestants) ? lb.contestants.filter(r => r && typeof r === "object") : [];
  if (!rows.length) return emptyStudy();
  const m = halfWidth(rows);
  const min = num(lb.min_games) ? lb.min_games : null;
  const through = num(lb.through_week) ? ` &middot; through week ${esc(lb.through_week)}` : "";
  // GUARD interim-label-while-the-artifact-says-interim [contracts/test_dashboard_study.py::test_the_interim_label_cannot_be_dropped_while_the_artifact_says_interim]: the board carries the interim label unless the artifact itself says `interim: false`.
  const label = lb.interim === false ? ""
    : `<p class="lb-interim" role="note"><b>${esc(INTERIM_LABEL)}</b>${through}</p>`;
  // /GUARD
  const toggle = `<div class="lb-toggle" role="group" aria-label="Condition">`
    + `<button type="button" data-cond="blind" aria-pressed="true">Blind</button>`
    + `<button type="button" data-cond="line_aware" aria-pressed="false">Line-aware</button></div>`;
  return label + toggle
    + `<div class="lb-cond" data-cond-body="blind">${conditionTable(rows, min, "blind", m)}</div>`
    + `<div class="lb-cond" data-cond-body="line_aware" hidden>${conditionTable(rows, min, "line_aware", m)}</div>`
    + `<p class="lb-foot">Each Δ is the contestant minus the betting market on the same games; `
    + `the whisker is its 95% interval about zero, which is the betting market.</p>`;
}

// The toggle's one effect, on the panel the button sits in.
export function chooseCondition(button) {
  const root = button.closest?.("#p-study");
  const cond = button.getAttribute("data-cond");
  if (!root) return;
  for (const b of root.querySelectorAll("[data-cond]")) {
    b.setAttribute("aria-pressed", String(b.getAttribute("data-cond") === cond));
  }
  for (const d of root.querySelectorAll("[data-cond-body]")) {
    d.hidden = d.getAttribute("data-cond-body") !== cond;
  }
}
