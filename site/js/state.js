// The four-state vocabulary (#347) and what a panel is made of.
//
// Pure and DOM-free: `panelParts` is the data, `dom.js` draws it.

import { ago, cap, esc } from "./format.js";

// The four states, named once. `frozen` is what a panel is unless something says otherwise.
export const STATE = { frozen: "state-frozen", live: "state-live", stale: "state-stale",
                unconfirmed: "state-unconfirmed" };

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
