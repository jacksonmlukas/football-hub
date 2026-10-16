// The status line (#351): one record per artifact, drawn as one line.

import { ago, esc } from "../format.js";

// The status line's data (#351): one record per artifact, whole -- name, age, stale, reason,
// fetched -- so that a reader-facing expansion later changes only how it is drawn. The
// manifest's artifacts come first; `bigten` and `cfbd` are not in the manifest, so the page
// loads them itself (`sides`, null where the file is absent) and reports their own `reason`.
// An absent side file is a named non-fetch with a standing reason, not an error and not
// absence. Built from data only, so a test can import it.
// GUARD status-records-carry-every-artifact [contracts/test_dashboard_escapes.py::test_the_status_records_name_every_manifest_artifact_with_its_age]: the footer's data has an entry, with age, for every artifact the manifest lists.
export function statusRecords(man, sides) {
  const out = (man?.artifacts ?? []).map(a => ({
    name: String(a?.name ?? "?"),
    age: a?.generated_at ? ago(a.generated_at) : "age unknown",
    stale: !!a?.stale || a?.present === false,
    reason: a?.reason ?? (a?.present === false ? "the artifact is missing" : null),
    fetched: a?.present !== false,
  }));
  for (const name of ["bigten", "cfbd"]) {
    if (out.some(r => r.name === name)) continue;
    const s = sides?.[name] ?? null;
    out.push(s ? {
      name,
      age: s.generated_at ? ago(s.generated_at) : "age unknown",
      stale: !!s.stale || s.fetched === false,
      reason: s.reason ?? null,
      fetched: s.fetched !== false,
    } : {
      name, age: "never", stale: true, fetched: false,
      reason: `no ${name}.json has been published`,
    });
  }
  return out;
}
// /GUARD

// One line, silent when healthy: every record's name and age; for anything stale or unfetched,
// its reason too. Every interpolation is escaped -- the reasons come off feeds.
export function statusLine(records) {
  const items = records.map(r => {
    const bad = r.stale || !r.fetched;
    const why = bad
      ? ` &mdash; ${r.fetched ? "stale" : "not fetched"}: ${esc(r.reason ?? "no reason given")}` : "";
    return `<span class="${bad ? "down" : ""}">${esc(r.name)} ${esc(r.age)}${why}</span>`;
  });
  return `<p class="status">${items.join(" &middot; ")}</p>`;
}
