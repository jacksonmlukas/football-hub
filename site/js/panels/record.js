// The track record panel's body, and the interval-coverage block under its calibration curve.

import { reliabilitySVG } from "../charts.js";
import { ago, esc, fmt, pct } from "../format.js";
import { chipFrozen, chipStale, chipUnconfirmed, staleOf } from "../state.js";
import { table } from "../table.js";
import { BACKTEST_WINDOW_DAYS } from "./statement.js";

// The two counts as chips (#486): pre-registered and backtest, frozen while the record is current
// and stale (with its age inside the chip) once the manifest marks the record stale or it is older
// than the statement's own window. A count the history could not give is one unconfirmed chip that
// says so, never a zero. `art` is the manifest entry for the record; the tests that render the
// body alone pass none, and the record's own date decides.
export function countChips(tr, art) {
  const pre = tr.n_preregistered ?? null;
  const n = tr.n_scored ?? null;
  if (pre === null) {
    return chipUnconfirmed("pre-registered count unknown");
  }
  const at = Date.parse(tr.generated_at);
  const aged = !Number.isFinite(at) || (Date.now() - at) / 864e5 > BACKTEST_WINDOW_DAYS;
  const stale = staleOf(art) ?? (aged ? { at: tr.generated_at } : null);
  const age = stale ? (stale.at ? ago(stale.at) : "age unknown") : "";
  const chip = label => stale ? chipStale(age, label) : chipFrozen(label);
  const back = typeof n === "number" ? Math.max(0, n - pre) : null;
  return chip(pre + " pre-registered") + (back === null ? "" : " " + chip(back + " backtest"));
}

// A bin's edges off its label ("0.5-0.6"), or null when the label cannot be read.
function edgesOf(b) {
  const m = /^\s*([0-9.]+)\s*-\s*([0-9.]+)\s*$/.exec(String(b?.bin ?? ""));
  return m ? { lo: +m[1], hi: +m[2] } : null;
}

// The one sentence the view opens with (#486), chosen from a season's own bins: the most populated
// bin above 50% (the first of equals), as the bin the model said, the rate the home team won at
// and the games behind it. Returns null when no bin above 50% holds a game. Nothing is typed: the
// bin, the rate and the count are the artifact's.
export function calibrationSentence(season) {
  const bins = Array.isArray(season?.bins) ? season.bins : [];
  let best = null;
  for (const b of bins) {
    const e = edgesOf(b);
    if (!(b.n > 0) || typeof b.actual !== "number" || !Number.isFinite(b.actual)) continue;
    if (!e || e.lo < 0.5) continue;
    if (best === null || b.n > best.b.n) best = { b, e };
  }
  if (best === null) return null;
  const { b, e } = best;
  return `When we said ${Math.round(e.lo * 100)}–${Math.round(e.hi * 100)}%, the home team won ` +
         `${Math.round(b.actual * 100)}% of the time (${b.n} ${b.n === 1 ? "game" : "games"}).`;
}

// The season's log loss beside the coin flip's. The coin flip is read from the same season's
// `baselines` and is left out when the artifact has none.
export function logLossSentence(season, whose) {
  if (typeof season?.log_loss !== "number") return null;
  const coin = season.baselines?.coin_flip?.log_loss;
  return `${whose}'s log loss is ${fmt(season.log_loss, 3)}` +
         (typeof coin === "number" ? `; a coin flip scores ${fmt(coin, 3)}` : "") +
         " (lower is better).";
}

const seasonName = s => s.season == null ? "the predictions with an unrecorded season" : String(s.season);

// One season's curve, captioned with whose it is. Unlabelled, under a table of several seasons, a
// diagram reads as the pool -- which is the reading the split exists to prevent.
function curve(s) {
  const chart = reliabilitySVG(s.bins ?? []);
  return chart ? chart + `<p class="legend">Calibration for ${esc(seasonName(s))}. Each season is
       scored on its own curve.</p>` : "";
}

// The season table: each season's row followed by the rows its own `baselines` give, so a score
// always sits beside what a constant guess scores on the same games. Rows are text; `table` escapes.
function seasonRows(seasons) {
  const out = [];
  for (const s of seasons) {
    const nm = s.season ?? "unrecorded";
    out.push({ season: nm, n: s.n_scored, ll: s.log_loss, br: s.brier, base: false });
    const coin = s.baselines?.coin_flip;
    const home = s.baselines?.home_constant;
    if (coin) out.push({ season: nm + " coin flip", n: coin.n, ll: coin.log_loss, br: coin.brier, base: true });
    if (home) out.push({ season: nm + " home team always (" + pct(home.p) + ")", n: home.n,
                         ll: home.log_loss, br: home.brier, base: true });
  }
  return out;
}

// The track record panel's body.
//
// Split out of `render` because it has two payload shapes to serve and neither can be
// exercised in a browser without publishing an artifact first: `publish.track_record` scores each
// season on its own curve, and an artifact from before that carries no `seasons` key. A panel that
// needed the new key would go blank on the deploy that shipped it.
//
// The order is the view's (#486): the two counts as chips, the record's own note, one sentence
// about calibration, the log loss beside the coin flip's, the newest season's curve (older seasons
// each keep their own, folded), the season table, the player ranges, the survivor price. The
// page-level backtest statement (#354) sits above this body in the same section and is filled by
// `backtestStatement`; the framing "a backtest, not a record" is its, said once, so it is not
// repeated here. `art` is the manifest entry for the record, for the chips' staleness.
export function trackRecordBody(tr, art) {
  // Newest season first, by contract with the writer: `tr.log_loss`, `tr.brier` and
  // `tr.bins` are that season's numbers rather than a pool. Pooling 2025's sixteen scored
  // games with 2026's reports a curve neither season was measured at -- the 2025 model, its
  // fitted parameters and the betting market it was priced against are not the 2026 ones,
  // and a bin holding nine games from each hides that it is holding two answers.
  const seasons = Array.isArray(tr.seasons) ? tr.seasons : [];
  const newest = seasons[0] ?? null;
  // The other half of the record (#289, #293): whether the weekly player interval covers
  // what it claims, and where the survivor price holds by spread. Null when no measurement
  // has been committed, and then the record renders without it rather than not at all.
  const coverage = gameBandNote(tr.game_band) + intervalCoverageBody(tr.interval_coverage);
  // null is "could not tell" (#422) and must not be read as 0: the note then says why.
  const chips = '<p class="rec-chips">' + countChips(tr, art) + "</p>" +
    (tr.note ? `<p class="legend">${esc(tr.note)}</p>` : "");

  if (!newest) {
    // An artifact from before the split. Its numbers are pooled and nothing here can unpool
    // them, so they are shown as they are rather than relabelled with a season they were
    // never scored by -- and no calibration sentence is drawn from a pool.
    return chips + table([
      { label: "metric", get: r => r.k },
      { label: "value", get: r => r.v },
    ], [
      { k: "log loss ↓", v: fmt(tr.log_loss, 4) },
      { k: "Brier ↓", v: fmt(tr.brier, 4) },
    ]) + reliabilitySVG(tr.bins ?? []) + coverage;
  }
  const said = calibrationSentence(newest);
  const lead = said
    ? `<p class="rec-sentence">${esc(said)}</p>`
    : `<p class="rec-sentence">No bin above 50% holds a game yet, so there is no calibration to state.</p>`;
  const ll = logLossSentence(newest, seasonName(newest));
  const older = seasons.slice(1).map(s => {
    const c = curve(s);
    return c ? `<details class="rec-older"><summary>${esc(seasonName(s))}: ${esc(s.n_scored)} scored</summary>`
               + c + "</details>" : "";
  }).join("");
  return chips + lead + (ll ? `<p class="rec-ll">${esc(ll)}</p>` : "") +
    curve(newest) + older +
    table([
      { label: "season", get: r => r.season },
      { label: "scored", get: r => r.n },
      { label: "log loss ↓", get: r => fmt(r.ll, 4) },
      { label: "Brier ↓", get: r => fmt(r.br, 4) },
    ].map(c => ({ ...c, cls: r => r.base ? "thin" : "" })), seasonRows(seasons)) + coverage;
}

// The game margin's band (#507): fixed and normal, not fitted to games and not conformalised,
// so no coverage figure on this record is about it. Null before the publisher carried it.
function gameBandNote(gb) {
  if (!gb || gb.half_width == null) return "";
  return `<p class="legend">Game margin: a fixed normal ${esc(pct(gb.level))} band of
    ±${esc(fmt(gb.half_width, 2))} points (${esc(fmt(gb.z, 4))} × ${esc(fmt(gb.sd, 3))}),
    not conformalised and not scored for coverage here.</p>`;
}

// The player 80% ranges as bullet rows (#486): for the backtest and for each position of the
// current season, the claim as a tick, the tolerance band around it, and the measured coverage as
// the bar. The scale is 50% to 100%, a drawing choice stated here; a value past an end sits on it.
// Every figure is the artifact's. The verdict is a chip: the backtest's is the audit look that was
// taken, and the out-of-sample rows say "no verdict yet" in the unconfirmed state, because the
// binding reads of that season come at the audits after weeks 14 and 18 and no weekly figure is one.
const RANGE_AXIS_LO = 0.5;
const isNum = v => typeof v === "number" && Number.isFinite(v);

function bullet(r) {
  const W = 300, H = 14;
  const sx = v => ((Math.min(Math.max(v, RANGE_AXIS_LO), 1) - RANGE_AXIS_LO) / (1 - RANGE_AXIS_LO) * W);
  const bandL = sx(r.claim - r.band), bandR = sx(r.claim + r.band);
  return `<svg class="rb-plot" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
      aria-label="${esc(r.label)}: measured ${esc(pct(r.measured))}, claimed ${esc(pct(r.claim))}">`
    + `<rect class="rb-band" x="${bandL.toFixed(1)}" y="0" width="${(bandR - bandL).toFixed(1)}" height="${H}"/>`
    + `<rect class="rb-measure" x="0" y="4" width="${sx(r.measured).toFixed(1)}" height="6"/>`
    + `<line class="rb-claim" x1="${sx(r.claim).toFixed(1)}" x2="${sx(r.claim).toFixed(1)}" y1="0" y2="${H}"/></svg>`;
}

export function rangeRows(ic) {
  const rows = [];
  if (isNum(ic?.gate_cov80) && isNum(ic?.gate_claim) && isNum(ic?.band)) {
    rows.push({ label: "Backtest, all positions", measured: ic.gate_cov80, claim: ic.gate_claim,
                band: ic.band, n: ic.gate_n,
                chip: ic.audit
                  ? chipFrozen("audit look " + ic.audit.look + " of " + ic.audit.looks + ": " + ic.audit.marginal)
                  : chipUnconfirmed("no audit yet") });
  }
  const c = ic?.current_season;
  for (const p of Array.isArray(c?.by_position) ? c.by_position : []) {
    if (p.position == null || !isNum(p.cov80)) continue;   // the pooled "all" row is the line below
    rows.push({ label: c.season + " " + p.position, measured: p.cov80, claim: c.claim ?? ic.gate_claim,
                band: c.band ?? ic.band, n: p.n,
                chip: p.verdict ? chipFrozen(String(p.verdict)) : chipUnconfirmed("no verdict yet") });
  }
  const drawable = rows.filter(r => isNum(r.claim) && isNum(r.band));
  if (!drawable.length) return "";
  const items = drawable.map(r =>
    `<li class="rb-row"><span class="rb-name">${esc(r.label)}</span>`
    + `<span class="rb-text">measured <b>${esc(pct(r.measured))}</b> against ${esc(pct(r.claim))} `
    + `±${esc(fmt(r.band * 100, 1))} points`
    + (r.n == null ? "" : `, ${esc(Number(r.n).toLocaleString())} player-weeks`) + `</span>`
    + `<span class="rb-chip">` + r.chip + `</span>`
    + `<span class="rb-bar">` + bullet(r) + `</span></li>`).join("");
  return `<h4 class="rb-head">Player 80% ranges</h4><ul class="rb-list">${items}</ul>`
    + `<p class="legend">Each bar is the share of player-weeks that landed inside the 80% range. The tick `
    + `is the claim and the shaded band around it is the tolerance. The scale runs from 50% to 100%.</p>`;
}

// The weekly interval's coverage and the survivor price by spread bucket, under the
// calibration curve (#289, #293). `ic` is `hub.models.coverage.published_summary` as the
// publisher wrote it into `track_record.json`: the last *committed* measurement, not a
// fresh one, and null when none has been made.
//
// Two claims are made on the page and both are read off the artifact rather than typed
// here. The interval is served under the label 80% and the doc says it covers
// `gate_claim`; the verdict was read against the claim, so the claim is what is printed
// beside it -- `COVERS` beside `77.4%` with only the label on the line would read as the
// label covering. An artifact written before the claim was carried is published with the
// label as its claim, by `published_summary`, so `gate_claim` is always present.
//
// The bucket table keeps every bucket. One holding fewer than `min_bucket` games is shown
// and marked thin, never dropped: four rows where five were expected read as a bucket that
// vanished, and a gap over seven games looks exactly like a gap over seventy without the
// count beside it.
function intervalCoverageBody(ic) {
  if (!ic) return "";
  const weeks = (ic.gate_n ?? 0).toLocaleString();
  // What population the figure is over (#309). The gate grades the whole scored board since it
  // stopped leaving the clipped weeks out, and says how much of the board that is; an artifact
  // from before that is the unclipped weeks and is worded as it always was.
  const whole = ic.gate_subset === "all";
  const pop = whole
    ? weeks + " scored player-weeks" +
      (ic.gate_clipped_share != null ? ", " + pct(ic.gate_clipped_share) + " of them clipped at zero" : "") +
      (ic.n_uncalibrated ? ", " + Number(ic.n_uncalibrated).toLocaleString() + " more not scored for want of a calibration" : "")
    : weeks + " unclipped player-weeks";
  // The weekly figure is a report, never a gate (#310): the binding verdict is taken at audits,
  // three looks per claim-life, and the page says which look the last one was, or that none was.
  const auditNote = ic.audit
    ? " Audit look " + ic.audit.look + " of " + ic.audit.looks + ": " + ic.audit.marginal + "."
    : " The binding verdict is taken at audits; none has been taken yet.";
  // What the figure is about (#507): player-weeks of the QB/RB/WR/TE projection, never games.
  // This sits on the game record, so the sentence names its own unit instead of borrowing the
  // page's. An artifact from before the publisher's `about` field is player-weeks all the same.
  const ranges = rangeRows(ic);
  const weekly = `<p class="legend">Player-week interval (the weekly player projection, not game
    margins), labelled 80%: covers ${esc(pct(ic.gate_cov80))}
    of ${esc(pop)} against the claimed ${esc(pct(ic.gate_claim))}
    (±${esc(pct(ic.band))}) — this week's read ${esc(ic.verdict)}, measured ${esc(ic.generated_at)}.${esc(auditNote)}</p>`;
  // The current season's completed weeks, beside the backtest and never pooled with it (#423):
  // an out-of-sample row that carries no verdict of its own. Looks 2 and 3 of the claim read it,
  // after weeks 14 and 18, and the page says so rather than letting the backtest read as current.
  const c = ic.current_season;
  const current = c ? `<p class="legend">${esc(c.season)} so far, out of sample
    (${esc(c.weeks_complete)} weeks complete, not pooled with the figure above): ` +
    (c.n ? `the 80% interval covers ${esc(pct(c.cov80))} of ${esc(Number(c.n).toLocaleString())}
    scored player-weeks (68% interval ${esc(pct(c.cov68))}, ${esc(pct(c.clipped_share))} clipped at
    zero, ${esc(fmt(c.sigma, 1))} sigma at the binomial standard error)`
      : "no week has been scored yet") +
    `. No verdict is read from this row until the audit looks at weeks ` +
    `${esc(Object.values(c.looks ?? {}).map(l => l.after_week).join(" and "))}.</p>` : "";
  const s = ic.survivor;
  if (!s) return ranges + weekly + current;
  const floor = s.min_bucket ?? 50;
  const buckets = Array.isArray(s.buckets) ? s.buckets : [];
  const signed = v => (v === null || v === undefined) ? "—" : (v >= 0 ? "+" : "") + fmt(v, 3);
  // The verdict is a chip (#486), built apart from the sentence so the sentence stays text.
  const verdict = chipFrozen(String(s.verdict ?? "no verdict"));
  const headline = `<p class="legend">Survivor price, favourites of ${esc(s.favourite_spread)}+:
    priced ${esc(fmt(s.favourite_predicted, 3))}, won ${esc(fmt(s.favourite_actual, 3))}
    over ${esc(s.favourite_n)} games. Verdict: ` + verdict + `. By closing spread, the favourite's
    win rate against its price; a bucket under ${esc(floor)} games is marked, not dropped.</p>`;
  const rows = buckets.length ? table([
    { label: "spread", get: r => r.label ?? r.bin, cls: r => r.thin ? "thin" : "" },
    { label: "games", get: r => (r.n ?? 0) + (r.thin ? " (under " + floor + ")" : ""),
      cls: r => r.thin ? "thin" : "" },
    { label: "priced", get: r => fmt(r.predicted, 3), cls: r => r.thin ? "thin" : "" },
    { label: "won", get: r => fmt(r.actual, 3), cls: r => r.thin ? "thin" : "" },
    { label: "gap", get: r => signed(r.gap), cls: r => r.thin ? "thin" : "" },
  ], buckets) : "";
  return ranges + weekly + current + headline + rows;
}
