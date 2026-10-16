// The survivor panel's body, and the notes that qualify it.

import { esc, pct, weekList } from "../format.js";
import { STATE } from "../state.js";
import { table } from "../table.js";

// The survivor panel's body: the remaining plan, and the scope that makes it readable.
//
// Split out of `render` for the reason `trackRecordBody` was -- it builds prose from four
// scope fields and none of it can be exercised in a browser without publishing an artifact
// first.
//
// The scope is *rendered* here, not merely published. `publish.survivor` wrote
// `weeks_played` and nothing read it -- no page code, no test, no CLI -- and
// `weeks_remaining` was read by one test that rendered nothing (issue #57). CONTEXT.md's
// "Remaining plan" is the plan *together with* the scope that makes it readable, so showing
// it is what makes the writer's claim true: a plan whose first row is week 9 has to say what
// became of weeks 1 to 8, and a reader should not have to open the JSON to find out.
// The pool rules nobody has confirmed, said beside the numbers that depend on them. The
// dollar figures are conditional on `co_survivor_rule` in particular -- it decides how a
// shared pot splits -- and publishing them without saying so is the unlabelled claim
// `docs/method.md` exists to stop. The list comes from `hub.config`, so it cannot disagree
// with the settings it describes and it empties itself when a rule is confirmed.
export function unconfirmedNote(surv) {
  const un = Array.isArray(surv?.unconfirmed) ? surv.unconfirmed : [];
  // The digest of the rules this plan was computed under (#160). Rendered beside the caveat
  // rather than hidden in the JSON, because a reader comparing two publishes needs to see
  // whether the rules moved between them -- and until it was rendered, nothing read it.
  const rules = surv?.pool_digest ? ` Rules <code>${esc(surv.pool_digest)}</code>.` : "";
  if (!un.length) return rules ? `<p class="muted">${rules.trim()}</p>` : "";
  return `<p class="${STATE.unconfirmed}">These figures assume pool rules nobody has confirmed:
          <b>${esc(un.join(", "))}</b>. The dollar amounts move if any of them is settled
          differently.${rules}</p>`;
}


// The eliminated state (#380): the week it happened, the team, and whether a buyback is
// still open -- rendered instead of a plan and a survival figure, neither of which an out
// entry has. `surv.eliminated_week`/`surv.eliminated_team` are `null` on a state that names
// our entry as out without ever recording a pick for it -- not a shape a parsed pool
// payload produces, but said honestly rather than guessed at.
export function eliminatedBody(surv) {
  const week = surv.eliminated_week;
  const team = surv.eliminated_team;
  const buybackThrough = surv.buyback_cutoff_week;
  const when = (week === null || week === undefined)
    ? "an unrecorded week"
    : `week ${esc(week)}` + (team ? ` on ${esc(team)}` : "");
  const buyback = surv.buyback_open
    ? `A buyback is still open, through week ${esc(buybackThrough)}.`
    : `The buyback window is closed (it ran through week ${esc(buybackThrough)}).`;
  return `<p class="empty">Eliminated in ${when}. No remaining plan and no survival figure
          while this entry is out. ${buyback}</p>`;
}

// The no-pool-state caveat (#380): a remaining plan is still published when no pool record
// has ever been read, but it is this repository's own -- not verified against what the
// pool says -- and a reader should not mistake it for the pool's record the way the
// artifact this ticket was filed against did.
export function unreadStateNote(surv) {
  if (!surv || surv.status !== "unread-state") return "";
  return `<p class="warn">No pool state has been read for this entry: the remaining plan
          above is this repository's own, not the pool's record.</p>`;
}

export function survivorBody(surv) {
  const rows = Array.isArray(surv?.rows) ? surv.rows : [];
  // Absent and empty are different claims, and both arrive here. `[]` says no week has been
  // played; missing says the artifact never said -- `site/data/survivor.json` as committed
  // carries neither of these keys and will not until the next slate runs, so it is what the
  // deploy shipping this renders. Read as zero, a plan of eighteen weeks would be captioned
  // "0 week(s) covered", which is the caption with a hole in it that the record panel above
  // already learned not to leave.
  const left = surv.weeks_remaining ?? null;
  const played = surv.weeks_played ?? null;
  const spent = surv.spent ?? [];
  // Weeks that are in this plan only because the store was read: the schedule's own moving
  // field carries no spread for them. Reported rather than assumed because it moves --
  // upstream fills that field in as the season approaches, so a week listed here today is
  // not listed here in December.
  const storeOnly = surv.snapshot_only_weeks ?? [];

  const scope = `<p class="legend"><b>Remaining plan, `
    + `${esc(surv.season ?? "an unrecorded season")}</b>: `
    + (left
       ? `${esc(left.length)} week(s) covered &mdash; ${esc(weekList(left))}. `
       : `the rows below are every week it covers, which is all this artifact records. `)
    + (played === null ? ""
       : played.length
         ? `${esc(played.length)} week(s) already played or under way, and absent from it: `
           + `${esc(weekList(played))}. `
         : `No week has been played yet. `)
    + (storeOnly.length
       ? `Priced from a dated snapshot only, and so not on nflverse's own field: `
         + `${esc(weekList(storeOnly))}.`
       : "")
    + `</p>`;

  // Spending a team early costs you that team later, so a mid-season plan is against the
  // teams that are left rather than the whole league.
  const spentNote = spent.length
    ? `<p class="empty">Already spent and unavailable: ${esc(spent.join(", "))}.
         Read from the last published plan, so it is what was planned rather than
         what was entered.</p>`
    : "";

  return scope + spentNote + table([
    { label: "week", get: r => r.week },
    { label: "team", get: r => r.team ?? "" },
    { label: "win", get: r => pct(r.win_prob) },
  ], rows);
}
