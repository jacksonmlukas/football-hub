// The slate panel: frozen model numbers beside live scores, and the archived weeks.

import { ago, esc } from "../format.js";
import { findArt, frozen, load, otherWeeks, predsArt } from "../artifacts.js";
import { panel } from "../dom.js";
import { chipFrozen, chipLive, staleOf } from "../state.js";
import { DEFAULT_SORT, feedState, liveSlot, matchupKey, gamesBody, sidesOf } from "./games.js";

// The ESPN-to-nflverse team alias lives with the tiles now (`matchupKey` in games.js): the live
// join is keyed by the matchup, and the tile is what reads it.

// The slate is the one panel mixing two clocks: model numbers frozen at lock, scores
// moving. Everything here exists to keep that boundary visible -- the frozen columns are
// captioned with when they were locked, the live columns are shaded and captioned with
// how old the scores are, and a game with no live match says so rather than showing a
// blank that could be read as 0-0.

// The archived week a reader asked for, and the manifest the slate last rendered against.
// Both live out here because the link that switches weeks is handled by delegation below:
// every panel is rebuilt through innerHTML, so a listener bound to the link itself would
// not survive the next render.
let viewing = null;
let slateMan = null;
// How the tiles are ordered, and the rows last drawn for the live week. The poll reads the rows
// to find the score slots; it never touches the frozen numbers beside them.
let sortBy = DEFAULT_SORT;
let drawn = null;

// The slate for a week that has rows, and the two facts about the live join that belong in
// the panel's frame beside it: how many games matched, and how old the scores are. Hence
// the `{age, warn, html}` where `trackRecordBody` returns markup alone -- the join happens
// here, so what it found is known here and nowhere else.
export async function slateBody(preds, art, archived, liveArt) {
  // Scores are joined only for the week the manifest names. The live map is keyed by the
  // matchup alone, and three of 2026 week 1's fixtures repeat 2025 week 18's -- so reaching
  // the older week put tonight's score beside a prediction made for a different season's
  // game, under a column captioned "live". An archived week gets no overlay at all rather
  // than a plausible wrong one.
  const live = archived ? null : await load("live");
  // One tile per prediction row (no cap): the view is the whole slate.
  const rows = preds.rows;
  const feed = feedState(live, liveArt, rows, archived);
  const matched = rows.filter(r => {
    const sides = sidesOf(r);
    return sides && feed.byGame.has(matchupKey(sides.away, sides.home));
  }).length;
  drawn = archived ? null : { rows, art: liveArt };

  const lockedAt = preds.rows[0]?.predicted_at ?? preds.generated_at;
  const liveAge = archived ? "not shown for a week that is over"
                : live ? ago(live.generated_at) : "unavailable";
  // Which week this slate is, stated in the panel rather than only in the page subtitle.
  // `site/data/` holds two published weeks -- 2025 week 18 and 2026 week 01 -- and a panel
  // headed "Weekly slate" showing "week 18" does not say which of them is on screen; the
  // subtitle is a page-level fact a reader scanning one panel need never have seen.
  //
  // Off the artifact and nothing else. The manifest names the week it was *written* for,
  // which is precisely the one week that can disagree with the rows on screen -- reading it
  // as a fallback would reintroduce the skew this panel exists to make visible, and an
  // archived week is not the manifest's week at all. An artifact that does not say which
  // week it is says so rather than borrowing an answer.
  const shownSeason = preds.season ?? preds.rows[0]?.season;
  const shownWeek = preds.week ?? preds.rows[0]?.week;
  const stamp = (shownSeason != null && shownWeek != null)
    ? `${shownSeason} week ${shownWeek}` : "an unrecorded season and week";
  const head = `
    <p class="legend">
      <b>${esc(stamp)}</b>.
      ${chipFrozen("frozen")} model numbers, locked ${ago(lockedAt)} — these
      do not move once games start.
      ${chipLive("live")} scores, ${liveAge}.
    </p>`;

  const body = gamesBody(rows, feed, preds, sortBy);

  return {
    age: `${preds.n} games · ${matched} live`,
    stale: staleOf(art),
    warn: archived ? "An earlier published week, kept exactly as it was committed."
        : (!live ? (liveArt?.reason ? `Live scores unavailable (${liveArt.reason}) — ` : "Live scores unavailable — ")
                      + "model numbers below are still correct."
                 : liveArt?.stale ? `Live scores are stale: ${liveArt.reason ?? "unknown"}.`
                 : (matched === 0 ? "No live game matches this slate; scores column is idle."
                                  : null)),
    html: head + body,
  };
}

export async function renderSlate(man) {
  slateMan = man;
  drawn = null;
  // The artifact this week was published as, named by whoever wrote it. See `predsArt`.
  const pArt = predsArt(man);
  const key = viewing ?? pArt?.name ?? null;
  // What the panel is showing, answered once. "Is this an archived week" decided the
  // manifest entry, the live join, the age of the scores and the warning, and each of the
  // four asked `viewing` again -- four readings of one question, and the live join is the
  // one that must never come out wrong: an archived week joined to tonight's scoreboard
  // puts a real score beside a prediction for a different season's game (issue #57).
  const archived = viewing != null;
  const preds = key ? await frozen(key) : null;
  // The manifest describes the week it published and not an archived one, so its staleness
  // and its reason belong to the panel only while that is what the panel is showing.
  const art = archived ? null : pArt;

  // The other published weeks, as links: issue #23 asks the site to render both, and the
  // only week reachable until now was whichever one the manifest happened to name.
  const links = otherWeeks(man, key)
    .map(w => `<a href="#" data-preds="${esc(w.name)}">${esc(w.label)}</a>`).join(", ");
  const also = links ? `<p class="legend">Also published: ${links}</p>` : "";

  // One exit, so the links are appended once. They were concatenated at each of the two
  // this render used to have, and the one that would eventually lose them is the empty
  // case -- which is exactly when a reader wants the week that did publish.
  const shown = preds?.rows?.length ? await slateBody(preds, art, archived, archived ? null : findArt(man, "live")) : {
    age: "",
    warn: art?.reason ? `No slate: ${art.reason}`
        : archived ? "That published week could not be loaded."
        : key ? "No predictions published yet."
              : "No slate: the manifest names no published week.",
    html: "",
  };
  panel("p-slate", { age: shown.age, stale: shown.stale, warn: shown.warn,
                     html: shown.html + also });
}


// Switching weeks. The click delegation in `main.js` calls this: choosing the manifest's own
// week is "back to now", which puts the live poll back in charge; choosing any other stops it,
// per `isArchived`.
export function chooseWeek(pick) {
  viewing = pick === predsArt(slateMan)?.name ? null : pick;
  return renderSlate(slateMan);
}

// Whether a reader is looking at an archived week. The poll asks, and does not redraw if so.
export const isArchived = () => Boolean(viewing);

// Choosing how the tiles are ordered. The predictions are in the frozen cache, so this redraws
// the same numbers in another order and fetches nothing it did not already hold but the scores.
export function chooseSort(how) {
  sortBy = how;
  return renderSlate(slateMan);
}

// The poll's whole job: put the current feed's state into each tile's score slot. Nothing else on
// the page is touched -- not a frozen number, not the order, not the popovers -- so a poll cannot
// replace a model number even by accident. An archived week has no slots and nothing to refresh.
// `root` is the document (a stub in the contract tests).
export async function refreshLive(man, root = document) {
  if (viewing != null || !drawn) return;
  const live = await load("live");
  const feed = feedState(live, findArt(man, "live"), drawn.rows, false);
  const byKey = new Map();
  for (const r of drawn.rows) {
    const sides = sidesOf(r);
    if (sides) byKey.set(matchupKey(sides.away, sides.home), r);
  }
  for (const slot of root.querySelectorAll("[data-live-slot]")) {
    const row = byKey.get(slot.getAttribute("data-game"));
    if (row) slot.innerHTML = liveSlot(row, feed);
  }
}
