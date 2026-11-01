// The hash router (#478): five views, one visible at a time, switched in place.
//
// Views are sections of the one document, not pages. `render()` in `main.js` still draws every
// panel once, into the section its view owns; the router only decides which section is shown.
// A hash is the whole of the router's state, so the back button, a pasted link and a reload all
// land on the same view with no history code of ours.
//
// Pure up to `mount`: `viewFor` and `show` take the document as an argument so a node test can
// hand them a stub, and nothing here touches `window` until `mount` is called.

import { esc } from "./format.js";

// `week` is the default: an empty or unknown hash shows it. `ticket` names the issue that
// builds the view's own content (#484-#488); until it lands the view holds a placeholder.
export const VIEWS = [
  { id: "week", label: "This week", ticket: 484 },
  { id: "games", label: "Games", ticket: 485 },
  { id: "record", label: "Record", ticket: 486 },
  { id: "study", label: "Study", ticket: 487 },
  { id: "ledger", label: "Ledger", ticket: 488 },
];
export const DEFAULT_VIEW = "week";

// `#/games`, `#games` and `#/games/` all name Games. Anything else is the default view, never
// an error: a stale bookmark must still land on a page.
export function viewFor(hash) {
  const id = String(hash ?? "").replace(/^#\/?/, "").replace(/\/+$/, "");
  return VIEWS.some(v => v.id === id) ? id : DEFAULT_VIEW;
}

// The canonical link for a view, which is what the bar's anchors carry. The anchors are static
// markup in `index.html` (so the bar works before any script runs); a contract test holds them
// to this function and to VIEWS.
export const hrefFor = id => `#/${id}`;

// A view that has no content yet says so in one sentence and names where it will come from.
export const placeholderHtml = view =>
  `<p class="empty">The ${esc(view.label)} view is not built yet (issue #${view.ticket}).</p>`;

// Which views have their own renderer. The other three are drawn by `render()` into their
// panels, so showing them is only a matter of unhiding.
const RENDERERS = {
  study: el => { el.innerHTML = placeholderHtml(VIEWS[3]); },
  ledger: el => { el.innerHTML = placeholderHtml(VIEWS[4]); },
};

// Show exactly one view: hide the rest, mark the matching link current, and run the view's
// renderer into its `.view-body` if it has one. Returns the id shown.
export function show(doc, hash) {
  const id = viewFor(hash);
  for (const v of VIEWS) {
    const section = doc.getElementById(`v-${v.id}`);
    if (section) section.hidden = v.id !== id;
  }
  for (const v of VIEWS) {
    const link = doc.getElementById(`nav-${v.id}`);
    if (!link) continue;
    if (v.id === id) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const body = doc.getElementById(`v-${id}`)?.querySelector?.(".view-body");
  if (body && RENDERERS[id]) RENDERERS[id](body);
  return id;
}

// The same-document view transition, where the browser has one and the reader has not asked for
// less motion. `startViewTransition` snapshots, runs the callback, then crossfades; with
// `prefers-reduced-motion` the callback runs bare and nothing animates. The stylesheet also
// zeroes the transition pseudo-elements under that query, so a browser that ignores this check
// still does not animate.
export function swap(win, doc, change) {
  const reduced = win.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  if (!reduced && typeof doc.startViewTransition === "function") {
    doc.startViewTransition(change);
  } else {
    change();
  }
}

// Wire the router to a window. The first view is shown immediately and without a transition.
export function mount(win, doc) {
  show(doc, win.location.hash);
  win.addEventListener("hashchange", () => {
    swap(win, doc, () => {
      show(doc, win.location.hash);
      win.scrollTo?.(0, 0);
    });
  });
}
