// Text helpers every panel builder shares: the escaper, number and age formatting.
//
// Pure and DOM-free, so a test imports them directly.

export const cap = s => s ? s[0].toUpperCase() + s.slice(1) + (/[.!?]$/.test(s) ? "" : ".") : s;
export const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(d);
export const pct = v => (v === null || v === undefined) ? "—" : (100 * v).toFixed(1) + "%";
// A list of week numbers as prose. Text, escaped where it lands, exactly like a `table`
// getter -- and worded the way `hub.season.survivor`'s CLI words it, so the two readers of
// one plan read alike.
export const weekList = ws => ws.map(w => `wk ${w}`).join(", ");

export function ago(iso) {
  if (!iso) return "never";
  // The store's timestamps are naive UTC ("2026-10-17 16:58:09"). `new Date` reads an
  // offset-less string as the reader's local time, or rejects the space form outright, so
  // a naive stamp is made explicit UTC first.
  if (typeof iso === "string" && /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(iso)) {
    iso = iso.replace(" ", "T") + "Z";
  }
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (!Number.isFinite(mins)) return "unknown";
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const h = Math.round(mins / 60);
  return h < 48 ? `${h}h ago` : `${Math.round(h / 24)}d ago`;
}

// Everything that reaches innerHTML from a feed goes through this first. Player names,
// team names and status text come from ESPN and nflverse; an apostrophe or an ampersand in
// a name is enough to produce malformed markup, and anything angle-bracketed is injected
// into the page. This site is public, unattended, and refreshed twice a week by a workflow
// with a push token -- nobody is watching it render.
// GUARD esc-escapes-what-reaches-the-page [contracts/test_dashboard_escapes.py]: nothing off a feed reaches innerHTML as markup.
export const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// /GUARD
