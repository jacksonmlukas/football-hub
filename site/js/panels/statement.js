// The page-level backtest statement (#354).

import { esc } from "../format.js";
import { STATE, failureWarn, staleNote, staleOf } from "../state.js";

// The page-level backtest statement (#354). `n_preregistered = 0` qualifies every model number on
// the page, not only the record panel's, so it is said once, under the header, and every figure in
// it is read off the track record artifact -- none is typed here.
//
// It retires loudly. While nothing is pre-registered it is an `unconfirmed` statement of the
// scored count and zero; the first artifact with `n_preregistered > 0` does not delete it, it
// flips the same slot to a `frozen` statement of both counts, because a line that merely vanishes
// is a line a returning reader has stopped seeing by the week it matters. It carries its as-of
// and the condition that ends it, and it is `stale` -- never a current claim -- when the manifest
// marks the record stale or the record is older than the window. A missing artifact still fills
// the slot, with the standing reason. Returns `{cls, html}`; the page owns the DOM.
// GUARD backtest-statement-is-sourced-and-retires [contracts/test_dashboard_escapes.py::test_the_statement_flips_from_unconfirmed_to_frozen_when_pre_registration_begins]: the statement reads its counts off the artifact and flips state when the pre-registered count rises above zero.
const BACKTEST_WINDOW_DAYS = 9;  // a week of the Wed & Sat regen cadence plus two days' slack (chosen, not fitted; #354)
export function backtestStatement(tr, art) {
  if (!tr) {
    return { cls: STATE.unconfirmed,
             html: `<p>${esc(failureWarn(art, "No track record has been published, so no model number on this page can be called a record."))}</p>` };
  }
  const pre = tr.n_preregistered ?? null;
  const n = tr.n_scored ?? null;
  const scored = n === null ? "An unknown number of" : esc(n);
  const asOf = Number.isFinite(Date.parse(tr.generated_at))
    ? String(tr.generated_at).slice(0, 10) : "an unknown date";
  const lead = pre === null
    ? `<b>${scored} predictions scored; whether any were pre-registered could not be counted &mdash; a backtest until it can be.</b>`
    : pre === 0
    ? `<b>${scored} predictions scored retrospectively, ${esc(pre)} pre-registered &mdash; a backtest, not a record.</b>`
    : `<b>${esc(pre)} pre-registered, ${scored} scored.</b>` +
      (n > pre ? ` The other ${esc(n - pre)} were scored after the fact: a backtest, not a record.`
               : "");
  const ends = pre === null || pre === 0
    ? "This line changes when a prediction's commit predates its kickoff: it then states the pre-registered count."
    : "A prediction counts as pre-registered only once its commit predates its kickoff.";
  const at = Date.parse(tr.generated_at);
  const aged = !Number.isFinite(at) || (Date.now() - at) / 864e5 > BACKTEST_WINDOW_DAYS;
  const stale = staleOf(art) ?? (aged ? {
    reason: Number.isFinite(at) ? `older than its ${BACKTEST_WINDOW_DAYS}-day window`
                                : "the record carries no usable date",
    at: tr.generated_at } : null);
  return {
    cls: stale ? STATE.stale : pre === 0 || pre === null ? STATE.unconfirmed : STATE.frozen,
    html: `<p>${lead}</p><p class="thin">As of ${esc(asOf)}. ${esc(ends)}
           It goes stale ${BACKTEST_WINDOW_DAYS} days after the record was written.</p>`
          + (stale ? staleNote(stale) : ""),
  };
}
// /GUARD
