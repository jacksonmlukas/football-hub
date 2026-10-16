// The track record panel's body, and the interval-coverage block under its calibration curve.

import { reliabilitySVG } from "../charts.js";
import { esc, fmt, pct } from "../format.js";
import { table } from "../table.js";

// The track record panel's body.
//
// Split out of `render` because it has two payload shapes to serve and neither can be
// exercised in a browser without publishing an artifact first: `publish.track_record`
// scores each season on its own curve, and the committed `track_record.json` will not carry
// a `seasons` key until the next slate runs. A panel that needed the new key to render
// would go blank on the deploy that shipped it.
export function trackRecordBody(tr) {
  // null is "could not tell" (#422) and must not be read as 0: a 0 that means "unchecked" is
  // the defect that labelled 64 pre-registered predictions a backtest.
  const pre = tr.n_preregistered ?? null;
  // Newest season first, by contract with the writer: `tr.log_loss`, `tr.brier` and
  // `tr.bins` are that season's numbers rather than a pool. Pooling 2025's sixteen scored
  // games with 2026's reports a curve neither season was measured at -- the 2025 model, its
  // fitted parameters and the betting market it was priced against are not the 2026 ones,
  // and a bin holding nine games from each hides that it is holding two answers.
  const seasons = Array.isArray(tr.seasons) ? tr.seasons : [];
  const newest = seasons[0] ?? null;
  const chart = reliabilitySVG(tr.bins ?? []);
  // The other half of the record (#289, #293): whether the weekly player interval covers
  // what it claims, and where the survivor price holds by spread. Null when no measurement
  // has been committed, and then the record renders without it rather than not at all.
  const coverage = gameBandNote(tr.game_band) + intervalCoverageBody(tr.interval_coverage);

  // The framing, which decides how everything under it is read and is the whole of what
  // this panel withholds. `n_preregistered` is counted from the commit history (#422): 0 until a
  // prediction's commit predates kickoff, null when the history could not be read; withholding the
  // numbers here as well -- as this did for one revision -- meant showing nobody a
  // calibration at all, while the branch that had them was code no artifact could reach.
  // Refusing to present a backtest *as a record* is the honest claim and this sentence is
  // it: the numbers below are labelled by it rather than hidden behind it.
  //
  // Both branches share this line, and that is a deliberate change to the branch nobody
  // asked about: the pre-registered count used to be a `{ k, v }` row inside the metrics
  // table, and lifting it out is what lets one table serve both framings instead of two
  // tables differing by a row. Recorded here so the next reader does not read it as drift.
  const lede = pre === null
    ? `<p class="empty">${esc(tr.note ?? "")}</p>`
    : pre === 0
    ? `<p class="empty">${esc(tr.note ?? "")}</p>` +
      (tr.n_scored ? `<p class="empty">${esc(tr.n_scored)} predictions scored
         retrospectively. That is a backtest, not a record, and it is not shown as one.</p>`
                   : "")
    : `<p class="legend">${esc(pre)} pre-registered, ${esc(tr.n_scored ?? 0)} scored.</p>` +
      (tr.n_scored > pre ? `<p class="empty">${esc(tr.n_scored - pre)} scored after the
         fact: a backtest, not a record.</p>` : "");

  if (!newest) {
    // An artifact from before the split. Its numbers are pooled and nothing here can unpool
    // them, so they are shown as they are rather than relabelled with a season they were
    // never scored by.
    return lede + table([
      { label: "metric", get: r => r.k },
      { label: "value", get: r => r.v },
    ], [
      { k: "log loss", v: fmt(tr.log_loss, 4) },
      { k: "Brier", v: fmt(tr.brier, 4) },
    ]) + chart + coverage;
  }
  // Whose curve the diagram is. `publish.track_record` groups rows carrying no season under
  // `season: null` and orders that group last, so it is the *newest* group only when every
  // scored row is undated -- the shape `preds_wk18.json` had before the season joined the
  // filename. "Calibration for ." is not a sentence, and a caption with a hole in it is how
  // a reader learns to distrust the numbers beside it.
  const whose = newest.season == null ? "the predictions with an unrecorded season"
                                      : String(newest.season);
  return lede + table([
    { label: "season", get: r => r.season ?? "unrecorded" },
    { label: "scored", get: r => r.n_scored },
    { label: "log loss", get: r => fmt(r.log_loss, 4) },
    { label: "Brier", get: r => fmt(r.brier, 4) },
  ], seasons) + chart +
    // The diagram draws one group's bins. Unlabelled, under a table of several seasons, it
    // reads as the pool -- which is the reading the split exists to prevent.
    (chart ? `<p class="legend">Calibration for ${esc(whose)}. Each season is
       scored on its own curve.</p>` : "") + coverage;
}

// The game margin's band (#507): fixed and normal, not fitted to games and not conformalised,
// so no coverage figure on this record is about it. Null before the publisher carried it.
function gameBandNote(gb) {
  if (!gb || gb.half_width == null) return "";
  return `<p class="legend">Game margin: a fixed normal ${esc(pct(gb.level))} band of
    ±${esc(fmt(gb.half_width, 2))} points (${esc(fmt(gb.z, 4))} × ${esc(fmt(gb.sd, 3))}),
    not conformalised and not scored for coverage here.</p>`;
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
  if (!s) return weekly + current;
  const floor = s.min_bucket ?? 50;
  const buckets = Array.isArray(s.buckets) ? s.buckets : [];
  const signed = v => (v === null || v === undefined) ? "—" : (v >= 0 ? "+" : "") + fmt(v, 3);
  const headline = `<p class="legend">Survivor price, favourites of ${esc(s.favourite_spread)}+:
    priced ${esc(fmt(s.favourite_predicted, 3))}, won ${esc(fmt(s.favourite_actual, 3))}
    over ${esc(s.favourite_n)} games — ${esc(s.verdict)}. By closing spread, the favourite's
    win rate against its price; a bucket under ${esc(floor)} games is marked, not dropped.</p>`;
  const rows = buckets.length ? table([
    { label: "spread", get: r => r.label ?? r.bin, cls: r => r.thin ? "thin" : "" },
    { label: "games", get: r => (r.n ?? 0) + (r.thin ? " (under " + floor + ")" : ""),
      cls: r => r.thin ? "thin" : "" },
    { label: "priced", get: r => fmt(r.predicted, 3), cls: r => r.thin ? "thin" : "" },
    { label: "won", get: r => fmt(r.actual, 3), cls: r => r.thin ? "thin" : "" },
    { label: "gap", get: r => signed(r.gap), cls: r => r.thin ? "thin" : "" },
  ], buckets) : "";
  return weekly + current + headline + rows;
}
