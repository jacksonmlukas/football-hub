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
// The manifest's `weeks` (#480) is the publisher's own list of every `preds_*` artifact it
// holds, newest first; this page keeps no copy of it, and a name in it is a file the writer
// found on disk, so a link here cannot be a name the page invented. An older manifest has no
// `weeks`: it still offers the one week it describes and no archive links, because the page
// has no way to know what else was published.
export function otherWeeks(man, shown) {
  // The label is the manifest's claim about the week, which is all there is before its
  // artifact has been fetched. The legend corrects it from the artifact on arrival.
  const listed = Array.isArray(man?.weeks)
    ? man.weeks.filter(w => typeof w?.name === "string")
        .map(w => ({ name: w.name, label: `${w.season ?? "?"} week ${w.week ?? "?"}` }))
    : null;
  const art = predsArt(man);
  const week = listed ?? (art
    ? [{ name: art.name, label: `${man?.season ?? "?"} week ${man?.week ?? "?"}` }] : []);
  return week.filter(w => w.name !== shown);
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
