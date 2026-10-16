// Loading artifacts and reading the manifest: `load`, the manifest lookups, `boardShows`,
// `predsArt`, the published-week list, and the frozen-predictions cache.

// Every fetch resolves. A panel with no data renders a reason; it never throws and never
// blanks the page. This is the whole reason the manifest exists.
export async function load(name) {
  try {
    const r = await fetch(`data/${name}.json?t=${Date.now()}`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

export function findArt(man, name) {
  return (man?.artifacts ?? []).find(a => a.name === name) ?? null;
}

// Whether the draft board has a place on the page (#348). It is a preseason artifact: once a
// slate has been published after it was built, the board is a stale answer to a question
// nobody is asking, and it retires. Derived, never deleted -- `draft_board.json` stays on disk,
// in the manifest and published; this reads two stamps the manifest already carries, so next
// July's rebuild brings the panel back with nobody remembering to re-add it.
//   - no `preds_*` entry at all (the July case): the board shows;
//   - any `preds_*` entry stamped strictly after the board's: it retires;
//   - a board entry missing, or stamped with nothing parseable: there is no board to show, so
//     nothing renders for it rather than a guess. A `preds_*` stamp that does not parse cannot
//     be called newer, so it does not retire the board either.
export function boardShows(man) {
  const built = Date.parse(findArt(man, "draft_board")?.generated_at);
  if (!Number.isFinite(built)) return false;
  return !(man?.artifacts ?? []).some(a =>
    String(a?.name ?? "").startsWith("preds_") && Date.parse(a.generated_at) > built);
}

// The manifest's entry for this week's predictions -- name, stale flag and reason in one
// lookup, because they are one fact.
//
// Read, never rebuilt. `publish.preds_name` owns the spelling and `store.week_key` owns the
// zero-pad inside it; this page can import neither, so building the name here kept a second
// copy of both rules in a language that cannot be held in step with the first. It had
// already drifted: `manifest.json` listed `preds_wk01` beside a committed
// `preds_2026_wk01.json`, so `findArt` matched nothing and the slate lost its stale and
// reason reporting with no error anywhere -- the file still loaded, because the *other*
// copy of the rule happened to be the right one. There is no fallback that spells a name
// this page could not read: a guessed name fetches nothing and reports the 404 as a week
// that was never published, which is a lie about the record.
//
// No week argument, deliberately. `publish.artifacts` emits exactly one `preds_`-prefixed
// entry per manifest, so there is nothing here to disambiguate; if it ever emitted several,
// the rule for choosing between them would be the writer's to state, and a page picking one
// on its own would be guessing again.
export function predsArt(man) {
  return (man?.artifacts ?? []).find(a => String(a?.name ?? "").startsWith("preds_")) ?? null;
}

// The published weeks a reader can reach other than the one on screen.
//
// The manifest describes the week it was written for and nothing else, so an older artifact
// is not discoverable from this page at all: `preds_2025_wk18.json` is committed, is the
// sixteen games the track record is scored from, and until now no reader could open it --
// issue #23 asks that the site render both. These are names the writer wrote, copied once;
// copying `preds_name`'s *output* is not the same as keeping a second copy of `preds_name`,
// and a name that stops resolving is a broken link rather than a silently wrong fetch.
// `tests/contracts/test_dashboard_escapes.py` pins each one to a file that is committed.
// When `publish` starts listing what it has published, this list goes.
export const ARCHIVE = [{ name: "preds_2025_wk18", label: "2025 week 18" }];

export function otherWeeks(man, shown) {
  // The current week's label is the manifest's claim about it, which is all there is before
  // its artifact has been fetched. The legend corrects it from the artifact on arrival.
  const art = predsArt(man);
  const current = art
    ? [{ name: art.name, label: `${man?.season ?? "?"} week ${man?.week ?? "?"}` }] : [];
  return [...current, ...ARCHIVE].filter(w => w.name !== shown);
}

// Predictions are fetched at most once per artifact per page load and kept here. The poll
// below must not be able to replace them even by accident: making the freeze structural
// rather than a convention is the point of 4.3, and a re-fetch on a timer is precisely how
// a "frozen" number quietly starts moving. Keyed by the artifact's own name, so that
// reading an archived week cannot disturb the current one and the key always names the
// artifact that was actually served.
const frozenPreds = new Map();

export async function frozen(key) {
  // The promise rather than its result: filling the map only after the await leaves a
  // window in which a second caller fetches the same artifact again, and "fetched once" is
  // the property this is for.
  if (!frozenPreds.has(key)) frozenPreds.set(key, load(key));
  return frozenPreds.get(key);
}
