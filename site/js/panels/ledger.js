// The Ledger view (#488): one row per verdict in `ledger.json` (#483), drawn and nothing else.
//
// Pure and DOM-free so a test can import it. The page holds no verdict text of its own: the word
// on a row, the name, the figure and the two links all come off the published entry, and an
// empty or unreadable ledger says so without naming a verdict. What this module does own is
// *classification*, not wording: which colour a published verdict takes and where its class
// sorts. A verdict it has not heard of is shown as published, in the muted colour, last.
// Every string off a feed goes through `esc`.

import { esc } from "../format.js";

// Where the committed docs live. A link is this base plus a repo-relative path the publisher
// emitted (`adr`), so the page invents no path; a contract test checks each one is a real file.
const REPO = "https://github.com/jacksonmlukas/football-hub";
const blob = path => `${REPO}/blob/main/${path}`;

// Verdict classes in display order (Removed first), each with the colour class that draws it:
// loss for a removal, turf for what stayed on the page, muted for no effect, the unconfirmed
// ink for a claim that is not established. A contract test holds this set to the publisher's.
const CLASSES = [
  ["Removed", "lg-loss"],
  ["Demoted", "lg-kept"],
  ["Shown never ranked", "lg-kept"],
  ["Adopted", "lg-kept"],
  ["Inert", "lg-null"],
  ["Null", "lg-null"],
  ["Inconclusive", "lg-null"],
  ["Not established", "lg-unconfirmed"],
];
const rank = v => {
  const i = CLASSES.findIndex(c => c[0] === v);
  return i < 0 ? CLASSES.length : i;
};
const colourOf = v => CLASSES.find(c => c[0] === v)?.[1] ?? "lg-null";

// Verdict class, then `as_of` newest first (ISO dates compare as strings). `Array.sort` is
// stable, so equal rows keep the published order.
export function sortLedger(rows) {
  return rows.map((r, i) => [r, i]).sort(([a, i], [b, j]) =>
    rank(a.verdict) - rank(b.verdict)
    || String(b.as_of ?? "").localeCompare(String(a.as_of ?? ""))
    || i - j).map(([r]) => r);
}

const link = (href, text) => `<a href="${esc(href)}">${text}</a>`;

const row = e => {
  const links = [
    typeof e.adr === "string" && e.adr ? link(blob(e.adr), "ADR") : "",
    Number.isInteger(e.issue) ? link(`${REPO}/issues/${e.issue}`, `#${e.issue}`) : "",
  ].filter(Boolean).join(" ");
  return `<li class="lg-row">`
    + `<span class="lg-verdict ${colourOf(e.verdict)}">${esc(e.verdict ?? "?")}</span>`
    + `<span class="lg-name">${esc(e.name ?? "?")}</span>`
    + `<span class="lg-figure">${esc(e.figure ?? "")}</span>`
    + (links ? `<span class="lg-links">${links}</span>` : "")
    + `</li>`;
};

const INTRO = `<p class="lg-intro">Every model faced a rule written before its numbers. Most lost. `
  + `What survives is the draft market, corrected for three effects.</p>`;

const PREREG = `<h4 class="lg-sub">How a prediction becomes pre-registered</h4>`
  + `<ol class="lg-steps">`
  + `<li>The prediction file is written before the first kickoff.</li>`
  + `<li>It is committed, and the commit time is the pre-registration.</li>`
  + `<li>It is scored only after the games, against that commit.</li></ol>`
  + `<p class="legend">${link(blob("docs/track-record.md"), "The full rule, and what it does not prove")}</p>`;

const METHOD = `<h4 class="lg-sub">The method</h4>`
  + `<p class="lg-method">Fix the decision rule before looking at the numbers, in code, with a test. `
  + `Measure each predictor strictly before the outcome it is scored on, compare against the simplest `
  + `thing that works, and count every comparison tried. Each rule in the method names the mistake `
  + `that produced it. ${link(blob("docs/method.md"), "Read the method")}</p>`;

// The body of the Ledger panel. An artifact with no rows says why in the rows' place.
export function ledgerBody(art) {
  const rows = Array.isArray(art?.rows) ? art.rows : [];
  const list = rows.length
    ? `<ul class="lg-list">${sortLedger(rows).map(row).join("")}</ul>`
    : `<p class="empty">The ledger has no entries to show.</p>`;
  return INTRO + list + PREREG + METHOD;
}
