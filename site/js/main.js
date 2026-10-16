// The entry point: `render`, the live poll and the click delegation.
//
// `index.html` loads this with `type="module"`; every other file under `site/js/` is
// imported from here. No bundler, no CDN: the browser fetches each one same-origin.

import { boardShows, findArt, load } from "./artifacts.js";
import { panel } from "./dom.js";
import { ago, cap, pct } from "./format.js";
import { boardBody } from "./panels/board.js";
import { statusLine, statusRecords } from "./panels/footer.js";
import { lineupHeadline } from "./panels/lineup.js";
import { trackRecordBody } from "./panels/record.js";
import { rosterBody } from "./panels/roster.js";
import { chooseWeek, isArchived, renderSlate } from "./panels/slate.js";
import { backtestStatement } from "./panels/statement.js";
import { eliminatedBody, survivorBody, unconfirmedNote, unreadStateNote } from "./panels/survivor.js";
import { STATE, failureWarn, staleOf } from "./state.js";

export async function render() {
  const man = await load("manifest");
  const wk = man?.week ?? "?";
  document.getElementById("subtitle").textContent =
    man ? `${man.season} · week ${wk} · published ${ago(man.generated_at)}`
        : "no manifest — run `make slate`";

  // The status line is drawn before any panel, so a panel that throws cannot take it with it.
  const [bigten, cfbd] = await Promise.all([load("bigten"), load("cfbd")]);
  // Guarded (#352): a record that makes the status line throw costs the footer and nothing
  // else -- the panels below it still draw.
  try {
    document.getElementById("footer").innerHTML = statusLine(statusRecords(man, { bigten, cfbd }));
  } catch {
    document.getElementById("footer").innerHTML =
      `<p class="status"><span class="down">status line unavailable: an artifact record could not be read</span></p>`;
  }

  // The statement is drawn before any panel for the reason the footer is: nothing below can take
  // it down. Its slot is never left empty, so a record it cannot read says why (#354).
  const tr = await load("track_record");
  const trArt = findArt(man, "track_record");
  const slot = document.getElementById("st-backtest");
  try {
    const said = backtestStatement(tr, trArt);
    slot.className = `statement ${said.cls}`;
    slot.innerHTML = said.html;
  } catch {
    slot.className = `statement ${STATE.unconfirmed}`;
    slot.innerHTML = `<p>The backtest statement could not be built from the track record.</p>`;
  }

  // --- draft board (retires itself when a newer slate exists, #348)
  const boardEl = document.getElementById("p-board");
  const shows = boardShows(man);
  boardEl.hidden = !shows;
  if (!shows) {
    boardEl.querySelector(".age").textContent = "";
    boardEl.querySelector(".body").innerHTML = "";
  }
  const boardArt = shows ? await load("draft_board") : null;
  // The envelope since #107. A clone on an older commit still has the bare list this used to
  // be, and reading both is one `??` -- cheaper than a page that renders nothing for anyone
  // who has not pulled.
  const board = Array.isArray(boardArt) ? boardArt : (boardArt?.rows ?? null);
  const bArt = findArt(man, "draft_board");
  if (!shows) {
    // Retired, not broken: the section is hidden above and has nothing to say.
  } else if (Array.isArray(board) && board.length) {
    const rows = board.filter(r => r.vor != null).slice(0, 12);
    panel("p-board", {
      // Dated, at last. This panel could never be stale and could never be aged, because the
      // producer reported no stamp -- on the artifact whose freshness matters most on draft
      // night. `built` rather than `synced`: a board is built, not fetched.
      age: `${board.length} players · built ${ago(boardArt?.generated_at)}`,
      stale: staleOf(bArt),
      html: boardBody(rows),
    });
  } else {
    // Backticks, not `<code>`: `warn` is escaped, and the manifest's own reason strings
    // ("run `python -m hub.season.roster --write`") already read this way.
    panel("p-board", { age: "", warn: failureWarn(bArt, "No draft board. Run `make draft`.") });
  }

  // --- my roster, and where the set lineup differs from the best one
  const ros = await load("roster");
  const rArt = findArt(man, "roster");
  if (ros && Array.isArray(ros.rows) && ros.rows.length) {
    // The call is the headline (#352); the table below it is the full roster. mu is the
    // board's season-long projection, so it does not know who is hurt this week.
    document.getElementById("p-lineup").hidden = false;
    panel("p-lineup", { age: `synced ${ago(ros.generated_at)}`, stale: staleOf(rArt),
                        html: lineupHeadline(ros, staleOf(rArt)) });
    panel("p-roster", {
      // Synced, not published. `generated_at` on this artifact is the roster parquet's own
      // date, because the CLI serves last-good without rewriting it -- so this is when ESPN
      // last answered, which on lineup day is the number that matters. It used to be the
      // time the page was built, which is always "just now" (issue #122).
      age: `${ros.n} rostered · synced ${ago(ros.generated_at)}`,
      stale: staleOf(rArt),
      html: rosterBody(ros.rows),
    });
  } else {
    document.getElementById("p-lineup").hidden = true;
    panel("p-roster", {
      age: "",
      warn: failureWarn(rArt, "No roster yet. Run `python -m hub.season.roster --write`."),
      html: `<p class="empty">Knowable only after the draft, which is why this panel was
             empty all summer.</p>`,
    });
  }

  // --- weekly slate, with the live overlay
  await renderSlate(man);

  // --- track record
  if (tr) {
    panel("p-record", { age: ago(tr.generated_at), stale: staleOf(trArt),
                        html: trackRecordBody(tr) });
  } else {
    panel("p-record", { age: "", warn: failureWarn(trArt, "No track record published yet.") });
  }

  // --- survivor
  const sArt = findArt(man, "survivor");
  const surv = await load("survivor");
  const sRows = Array.isArray(surv?.rows) ? surv.rows : [];
  if (surv && surv.status === "eliminated") {
    // No plan and no survival figure to caption the panel by (#380) -- the fact that
    // stands in their place is the elimination itself.
    panel("p-survivor", {
      age: "eliminated",
      stale: staleOf(sArt),
      html: eliminatedBody(surv),
    });
  } else if (sRows.length) {
    // The unpriced weeks matter more than the plan does. In August the board runs a few
    // weeks deep, so a plan covering 12 of 18 is the honest answer and a plan that did not
    // say so would read as a season. A week here is the entrant's to fill, which is why it
    // is the panel's warning and not part of the scope sentence below it.
    const missing = surv.unpriced_weeks ?? [];
    const from = sRows[0]?.week;
    const sWeeks = new Set(sRows.map(r => r.week)).size;
    panel("p-survivor", {
      // Weeks, not rows: a double-pick week contributes two rows and one week, so counting
      // rows read "24 weeks" for an eighteen-week plan. The pick count rides alongside
      // because it is the other thing a reader wants and the two are no longer the same.
      age: `${sWeeks} wk${sWeeks === 1 ? "" : "s"} from wk ${from}`
         + (sRows.length !== sWeeks ? ` · ${sRows.length} picks` : "")
         + ` · survives ${pct(surv.survival ?? 0)}`,
      stale: staleOf(sArt),
      warn: (missing.length
             ? `Weeks ${missing.join(", ")} have no posted spread yet and still need a pick.`
             : null),
      html: survivorBody(surv) + unconfirmedNote(surv) + unreadStateNote(surv),
    });
  } else {
    panel("p-survivor", {
      age: "",
      warn: sArt?.reason ? cap(sArt.reason) : "No survivor plan published.",
      html: `<p class="empty">The panel stays so the gap is visible rather than silently absent.</p>`,
    });
  }

}

// Poll only the slate. The other three panels read artifacts that change once a week, and
// re-fetching predictions on a timer is exactly the failure this design prevents: a model
// number that moves during a game is worthless as a pre-registered claim, however good it
// looks. `render()` runs once; only scores are allowed a clock.
const LIVE_POLL_MS = 45000;
let manifestCache = null;

export async function tick() {
  // Not while a reader is looking at an archived week. That week's games are over, so the
  // timer has nothing to add, and redrawing the panel under them would take their choice
  // away -- which is how a page teaches a reader not to click anything.
  if (!manifestCache || isArchived()) return;
  await renderSlate(manifestCache);
}

// Switching weeks. Delegated from the document because the panel is rebuilt through
// innerHTML on every render, so a listener bound to the link would not survive one.
export function onClick(e) {
  const link = e.target.closest?.("[data-preds]");
  if (!link) return;
  e.preventDefault();
  chooseWeek(link.getAttribute("data-preds"));
}

// The entry point. The notice in `index.html` that says the script did not run is shown until
// this is reached: a module that fails to parse or load stops the graph before it gets here,
// and the reader sees why the panels are empty instead of a blank page.
export function start() {
  document.getElementById("boot-status").hidden = true;
  document.addEventListener("click", onClick);
  render().then(async () => { manifestCache = await load("manifest"); });
  setInterval(tick, LIVE_POLL_MS);
}

// A real document has a `readyState`; the stub the contract tests install does not, and they
// import this module for `render` and `tick` without wanting the poll started.
if (typeof document !== "undefined" && document.readyState !== undefined) start();
