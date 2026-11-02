// The Games view's own parts (#485): one tile per prediction row, with the 80% margin range drawn
// on a field, a footer that keeps the frozen number and the live score in separate slots, and a
// popover of provenance.
//
// Pure and DOM-free so a test can import it. Nothing here computes a number the artifact did not
// publish: the favourite's win probability is the published `home_win_prob` (or its complement),
// the band runs from the published `margin_lo` to `margin_hi`, the tick is the published
// `margin_mean`. The only arithmetic is placing a margin on the strip's axis, which is a scale and
// not a fact. Every string off a feed goes through `esc`.

import { ago, esc, fmt, pct } from "../format.js";
import { chipFrozen, chipLive, chipStale, chipUnconfirmed } from "../state.js";

// ESPN to nflverse, and the direction matters. Compared against a live scoreboard and a
// published slate, exactly two abbreviations differ: ESPN writes LAR and WSH where
// nflverse writes LA and WAS. Predictions are keyed by nflverse game_id, so ESPN is the
// side that gets translated.
//
// This map was initially written the other way round, which silently dropped the Rams
// game from the overlay every week -- a missing row reads as "not on the board", which is
// a legitimate state, so nothing would have looked wrong.
const TEAM_ALIAS = { LAR: "LA", WSH: "WAS" };
const teamKey = t => TEAM_ALIAS[t] ?? t;
// The key a prediction row and a live row share: `AWAY_HOME`, nflverse spelling.
export const matchupKey = (away, home) => `${teamKey(away)}_${teamKey(home)}`;

// The strip's scale, in points of home margin. Fixed for every tile so two tiles can be read
// against each other; a value past the end sits on the end. A drawing choice, stated: five
// touchdowns either side of a tie is wider than any published range so far.
export const HALF = 35;
const W = 300;           // the viewBox's width
const EZ = 28;           // each end zone's width
const FIELD_L = EZ, FIELD_R = W - EZ;
const TD = 7;            // a gridline every 7 points

// A margin placed on the strip. A non-finite value has no place on it.
export function marginX(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  const clamped = Math.min(Math.max(v, -HALF), HALF);
  return FIELD_L + (clamped + HALF) / (2 * HALF) * (FIELD_R - FIELD_L);
}
const at = v => marginX(v).toFixed(1);
const num = v => typeof v === "number" && Number.isFinite(v);

// The two sides of a game, off the id `2026_08_ARI_DAL` (season, week, away, home).
export function sidesOf(row) {
  const m = /^\d+_\d+_([^_]+)_([^_]+)$/.exec(String(row?.game_id ?? ""));
  return m ? { away: m[1], home: m[2] } : null;
}

// Who the model favours and by how much, off the published numbers. `team` is the abbreviation,
// or null where the id does not name the sides; a toss-up has no favourite.
export function favourite(row) {
  const sides = sidesOf(row);
  const p = row?.home_win_prob;
  if (!num(p)) return { team: null, prob: null, tossup: false };
  if (p === 0.5) return { team: null, prob: 0.5, tossup: true };
  const home = p > 0.5;
  return { team: sides ? (home ? sides.home : sides.away) : (home ? "home" : "away"),
           prob: home ? p : 1 - p, tossup: false };
}

// "LA by 12.5": the expected margin, named by the side it favours.
function marginLabel(row) {
  const sides = sidesOf(row);
  const m = row?.margin_mean;
  if (!num(m)) return "no expected margin";
  if (m === 0) return "even";
  const who = sides ? (m > 0 ? sides.home : sides.away) : (m > 0 ? "home" : "away");
  return `${who} by ${fmt(Math.abs(m), 1)}`;
}

// The field strip: end zones named for the teams, a gridline every 7 points, labels at 2 and 4
// touchdowns either side of a tie, the 80% range as a band with soft edges, and the expected
// margin as a pylon tick with its label. `i` makes the gradient's id unique within the page.
export function fieldStrip(row, i) {
  const sides = sidesOf(row);
  const away = sides?.away ?? "away", home = sides?.home ?? "home";
  const grid = [];
  for (let m = -HALF; m <= HALF; m += TD) {
    grid.push(`<line class="${m === 0 ? "gt-tie" : "gt-grid"}" x1="${at(m)}" x2="${at(m)}" y1="16" y2="46"/>`);
  }
  const labels = [[-4 * TD, "4 TD"], [-2 * TD, "2 TD"], [0, "tie"], [2 * TD, "2 TD"], [4 * TD, "4 TD"]]
    .map(([m, t]) => `<text class="gt-axis" x="${at(m)}" y="58" text-anchor="middle">${t}</text>`).join("");
  const gid = `gt-soft-${Number(i) || 0}`;
  const lo = row?.margin_lo, hi = row?.margin_hi;
  const band = num(lo) && num(hi) && hi >= lo
    ? `<rect class="gt-band" x="${at(lo)}" y="22" width="${(Number(at(hi)) - Number(at(lo))).toFixed(1)}" height="18" rx="4" fill="url(#${gid})"/>`
    : "";
  const mean = row?.margin_mean;
  let tick = "";
  if (num(mean)) {
    const x = marginX(mean);
    const anchor = x < 80 ? "start" : x > W - 80 ? "end" : "middle";
    tick = `<line class="gt-pylon" x1="${at(mean)}" x2="${at(mean)}" y1="14" y2="46"/>`
      + `<rect class="gt-pylon-cap" x="${(x - 2.5).toFixed(1)}" y="12" width="5" height="5" rx="1"/>`
      + `<text class="gt-pylon-label" x="${at(mean)}" y="9" text-anchor="${anchor}">${esc(marginLabel(row))}</text>`;
  }
  const range = band
    ? `; 80% range from ${fmt(lo, 1)} to ${fmt(hi, 1)} (positive favours ${esc(home)})` : "";
  return `<svg class="gt-strip" viewBox="0 0 ${W} 62" role="img"
      aria-label="${esc(marginLabel(row))}${range}">`
    + `<defs><linearGradient id="${gid}" x1="0" x2="1" y1="0" y2="0">`
    + `<stop class="gt-stop" offset="0" stop-opacity="0"/><stop class="gt-stop" offset="0.18" stop-opacity="0.6"/>`
    + `<stop class="gt-stop" offset="0.82" stop-opacity="0.6"/><stop class="gt-stop" offset="1" stop-opacity="0"/>`
    + `</linearGradient></defs>`
    + `<rect class="gt-ez" x="0" y="16" width="${EZ}" height="30"/>`
    + `<rect class="gt-ez" x="${FIELD_R}" y="16" width="${EZ}" height="30"/>`
    + `<rect class="gt-field" x="${FIELD_L}" y="16" width="${FIELD_R - FIELD_L}" height="30"/>`
    + `<text class="gt-ez-name" x="${EZ / 2}" y="35" text-anchor="middle">${esc(away)}</text>`
    + `<text class="gt-ez-name" x="${FIELD_R + EZ / 2}" y="35" text-anchor="middle">${esc(home)}</text>`
    + grid.join("") + band + tick + labels + `</svg>`;
}

// --- the live slot ---------------------------------------------------------------------------

// What the live feed is, once, for the whole slate: `kind` is `none` (an archived week: no slot at
// all), `unavailable` (the feed did not load), `idle` (stale, or it matched no game on this slate)
// or `ok`. `age` is the feed's own age, for the idle chip to name.
export function feedState(live, liveArt, rows, archived) {
  if (archived) return { kind: "none", age: "", byGame: new Map() };
  if (!live) return { kind: "unavailable", age: "", byGame: new Map() };
  const byGame = new Map();
  for (const g of live.rows ?? []) byGame.set(matchupKey(g.away, g.home), g);
  const matched = rows.some(r => byGame.has(matchupKey(sidesOf(r)?.away, sidesOf(r)?.home)));
  const idle = Boolean(liveArt?.stale) || !matched;
  return { kind: idle ? "idle" : "ok", age: ago(live.generated_at), byGame };
}

// The score slot for one game. It never shows a dash that could read as 0-0: a feed that is
// idle names its age in a stale chip, a game the feed has no row for (or has not started) says
// "not started", and a game in progress shows the score in the live state.
export function liveSlot(row, feed) {
  if (feed.kind === "none") return "";
  if (feed.kind === "unavailable") return chipUnconfirmed("scores unavailable");
  if (feed.kind === "idle") return chipStale(feed.age, "scores stale");
  const sides = sidesOf(row);
  const g = sides ? feed.byGame.get(matchupKey(sides.away, sides.home)) : null;
  const scored = g && g.state !== "pre" && g.away_score != null && g.home_score != null;
  if (!scored) return `<span class="gt-nostart">not started</span>`;
  const score = `<span class="gt-score num">${esc(sides.away)} ${esc(g.away_score)} – ${esc(sides.home)} ${esc(g.home_score)}</span>`;
  const note = esc(g.detail ?? g.state);
  return g.state === "in"
    ? `<span class="state-live gt-livebox">${chipLive("live")} ${score} <span class="thin">${note}</span></span>`
    : `${score} <span class="thin">${note}</span>`;
}

// --- the tile ---------------------------------------------------------------------------------

// The provenance popover: what the number was made from, off the row, and the artifact's own
// sentence for the row's price source.
function provenance(row, i, preds) {
  const said = preds?.provenance?.[row.price_source]?.why;
  const items = [["version", row.version], ["fit through week", row.fit_through_week],
    ["priced at", row.priced_at], ["predicted at", row.predicted_at],
    ["price source", row.price_source]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${v == null ? "not published" : esc(v)}</dd>`).join("");
  return `<button type="button" class="gt-info" popovertarget="gt-pop-${i}">Provenance</button>`
    + `<div id="gt-pop-${i}" class="gt-pop" popover><h4>Where this number came from</h4>`
    + `<dl>${items}</dl><p class="legend">${esc(said ?? "No provenance sentence is published for this price source.")}</p></div>`;
}

export function gameTile(row, i, feed, preds) {
  const sides = sidesOf(row);
  const fav = favourite(row);
  const title = sides
    ? `${esc(sides.away)} <span class="gt-at">at</span> ${esc(sides.home)}` : esc(row.game_id ?? "unknown game");
  const win = fav.prob == null ? `<p class="gt-prob thin">no win probability</p>`
    : `<p class="gt-prob"><b class="num">${pct(fav.prob)}</b> `
      + `<span class="gt-wins">${fav.tossup ? "a toss-up" : `${esc(fav.team)} wins`}</span></p>`;
  // Shown only when the model is something other than the betting market's own line: while it is
  // `market_baseline` the two are the same number and a column of differences would be zeros.
  const versus = row.model !== "market_baseline" && num(row.market_prob) && num(row.home_win_prob)
    ? `<p class="legend gt-vs">model ${pct(row.home_win_prob)} · market ${pct(row.market_prob)} (home win)</p>` : "";
  const slot = liveSlot(row, feed);
  const key = sides ? matchupKey(sides.away, sides.home) : "";
  const live = feed.kind === "none" ? ""
    : `<span class="gt-live" data-live-slot data-game="${esc(key)}">${slot}</span>`;
  const locked = row.predicted_at ? chipFrozen(`locked ${ago(row.predicted_at)}`) : chipUnconfirmed("lock time unknown");
  const digest = row.cfg_digest ? `<code class="digest">${esc(row.cfg_digest)}</code>` : "";
  return `<article class="game-tile" data-game="${esc(key)}"><h4 class="gt-title">${title}</h4>${win}${versus}`
    + `${fieldStrip(row, i)}`
    + `<div class="gt-foot"><span class="gt-locked">${locked}</span>${digest}</div>`
    + `<div class="gt-foot">${live}${provenance(row, i, preds)}</div></article>`;
}

// --- the list ---------------------------------------------------------------------------------

export const SORTS = [{ id: "kickoff", label: "Kickoff" }, { id: "lopsided", label: "Most lopsided" }];
export const DEFAULT_SORT = "kickoff";

// A stable sort of the published rows: by kickoff (a row with no kickoff last) or by how far the
// win probability sits from a coin flip. Neither reads anything but the row.
export function sortRows(rows, how) {
  const t = r => { const ms = Date.parse(r.kickoff); return Number.isFinite(ms) ? ms : Infinity; };
  const far = r => (num(r.home_win_prob) ? Math.abs(r.home_win_prob - 0.5) : -1);
  const key = how === "lopsided" ? r => -far(r) : t;
  return rows.map((r, n) => [r, n]).sort((a, b) => {
    const d = key(a[0]) - key(b[0]);
    return d === 0 || Number.isNaN(d) ? a[1] - b[1] : d;
  }).map(x => x[0]);
}

export function sortControls(active) {
  return `<div class="gt-controls" role="group" aria-label="Sort games"><span class="thin">Sort by</span>`
    + SORTS.map(s => `<button type="button" data-sort="${s.id}" aria-pressed="${s.id === active}">${s.label}</button>`).join("")
    + `</div>`;
}

export function gamesBody(rows, feed, preds, how) {
  const sorted = sortRows(rows, how);
  const note = rows.length && rows.every(r => r.model === "market_baseline")
    ? `<p class="legend gt-note">The model is the betting market's line until issue 365, so there is no model-versus-market comparison to show.</p>` : "";
  return sortControls(how) + note
    + `<div class="game-grid">${sorted.map((r, i) => gameTile(r, i, feed, preds)).join("")}</div>`;
}
