// The four-state vocabulary (#347) and what a panel is made of.
//
// Pure and DOM-free: `panelParts` is the data, `dom.js` draws it.

import { ago, cap, esc } from "./format.js";

// The four states, named once. `frozen` is what a panel is unless something says otherwise.
export const STATE = { frozen: "state-frozen", live: "state-live", stale: "state-stale",
                unconfirmed: "state-unconfirmed" };

// The four state chips (#478). The state is carried three ways so it survives colour blindness
// and a greyscale print: a glyph (lock, dot, clock, question mark), an outline (solid, none,
// dotted, dashed) and a fill (live is the only filled chip), with the word always spelled out.
// Colour is the fourth channel, not the first. The CSS keys on `chip-<state>`; a test keys on
// `data-state` and `data-glyph`.
//
// One builder per state rather than one builder with a flag, so the stale rule is a signature:
// `chipStale(age)` has no way to be called without an age (#351's rule: "stale" with no age
// cannot be told from a panel that went stale a minute ago), and a missing or blank age throws
// instead of drawing a chip that looks fine.
const GLYPHS = {
  lock: `<rect x="3.5" y="7" width="9" height="6.5" rx="1.2" fill="currentColor"/>
         <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" fill="none" stroke="currentColor" stroke-width="1.5"/>`,
  dot: `<circle cx="8" cy="8" r="4" fill="currentColor"/>`,
  clock: `<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M8 4.5V8l2.5 1.5" fill="none" stroke="currentColor" stroke-width="1.5"/>`,
  question: `<path d="M5.7 6.2a2.3 2.3 0 1 1 3.4 2c-.7.4-1.1.8-1.1 1.6" fill="none"
                   stroke="currentColor" stroke-width="1.6"/>
             <circle cx="8" cy="12.2" r="1" fill="currentColor"/>`,
};
const glyph = name =>
  `<svg class="glyph" data-glyph="${name}" viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">${GLYPHS[name]}</svg>`;
const chipOf = (kind, g, label, extra = "") =>
  `<span class="chip chip-${kind}" data-state="${kind}">${glyph(g)}`
  + `<span class="chip-label">${esc(label)}</span>${extra}</span>`;

export const chipFrozen = (label = "frozen") => chipOf("frozen", "lock", label);
export const chipLive = (label = "live") => chipOf("live", "dot", label);
export const chipUnconfirmed = (label = "unconfirmed") => chipOf("unconfirmed", "question", label);
export function chipStale(age, label = "stale") {
  if (typeof age !== "string" || !age.trim()) {
    throw new TypeError("a stale chip needs the age of what it describes");
  }
  return chipOf("stale", "clock", label, `<span class="chip-age">${esc(age)}</span>`);
}

// A stale panel's warning, in the stale state, with the artifact's age beside it. The age is
// required: "Stale" with no age cannot be told from a panel that went stale a minute ago, so
// an artifact that does not say when it was written says that instead of saying nothing.
// `stale` is `{reason, at}` -- the manifest's reason and the artifact's own `generated_at`.
// GUARD stale-warning-carries-the-state-and-the-age [contracts/test_dashboard_escapes.py::test_a_stale_artifact_renders_in_the_stale_state_with_its_age]: every stale warning renders in the stale state with the artifact's age beside it.
export function staleNote(stale) {
  const age = stale.at ? ago(stale.at) : "age unknown";
  return `<p class="warn ${STATE.stale}">Stale: ${esc(stale.reason ?? "unknown")} &middot;
          <span class="stamp">artifact ${esc(age)}</span></p>`;
}
// /GUARD

// What a panel is made of, kept apart from the DOM so a test can import it. A stale
// panel keeps every value it had -- the state dims it -- and a panel with no data renders
// its reason in the default frozen state, since there is nothing in it to be unsure of.
export function panelParts({ age, warn, html, stale }) {
  return {
    age: age ?? "",
    cls: stale ? STATE.stale : STATE.frozen,
    body: (stale ? staleNote(stale) : "")
          + (warn ? `<p class="warn">${esc(warn)}</p>` : "")
          + (html ?? ""),
  };
}

// The manifest entry's stale state, or null when the artifact is current. One reading of
// "is this stale" for every panel, so none of them can spell it differently.
export const staleOf = art => art?.stale ? { reason: art.reason, at: art.generated_at } : null;

// What a panel says when its artifact gave it nothing to draw: the manifest's own reason when
// it has one, else the panel's standing sentence. A failure is reported where it broke (#351),
// not only in the footer.
export const failureWarn = (art, fallback) =>
  art && (art.stale || art.present === false) && art.reason ? cap(art.reason) : fallback;
