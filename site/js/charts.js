// The reliability diagram, as an SVG string.

import { esc, pct } from "./format.js";

// Reliability diagram. Perfect calibration is the diagonal; dot area is the bin count,
// because a bin holding four games should not look like one holding sixty.
//
// Every bin is drawn (#350). An empty bin has no predicted/actual to plot, so it gets a hollow
// ring on the baseline at its own slot, titled "no games": a diagram of the three bins that
// happened to hold games, with no mark for the seven that did not, reads as a diagram of
// three. Where a bin sits comes off its label ("0.2-0.3") so the slot is where the bin is,
// and off its index when the label cannot be read.
export function reliabilitySVG(bins) {
  if (!bins.length) return "";
  const S = 300, pad = 40, span = S - 2 * pad;
  const filled = bins.filter(b => b.n > 0);
  const maxN = Math.max(1, ...filled.map(b => b.n));
  const x = v => pad + v * span, y = v => S - pad - v * span;
  const mid = (b, i) => {
    const m = /^\s*([0-9.]+)\s*-\s*([0-9.]+)\s*$/.exec(String(b.bin ?? ""));
    return m ? (+m[1] + +m[2]) / 2 : (i + 0.5) / bins.length;
  };
  const marks = bins.map((b, i) => {
    if (!(b.n > 0)) {
      return `<circle class="bin-empty" cx="${x(mid(b, i)).toFixed(1)}" cy="${(S - pad).toFixed(1)}" r="3"
               fill="none" stroke="var(--muted)" stroke-width="1.2"><title>${esc(b.bin)}: no games</title></circle>`;
    }
    const r = 4 + 8 * Math.sqrt(b.n / maxN);
    return `<circle class="bin-filled" cx="${x(b.predicted).toFixed(1)}" cy="${y(b.actual).toFixed(1)}" r="${r.toFixed(1)}"
             fill="var(--accent)" fill-opacity="0.65"><title>${esc(b.bin)}: n=${b.n}, predicted ${pct(b.predicted)}, actual ${pct(b.actual)}</title></circle>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t =>
    `<text x="${x(t).toFixed(1)}" y="${S - pad + 13}" text-anchor="middle" font-size="9" fill="var(--muted)">${Math.round(t * 100)}%</text>` +
    `<text x="${pad - 5}" y="${(y(t) + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="var(--muted)">${Math.round(t * 100)}%</text>`
  ).join("");
  return `<svg viewBox="0 0 ${S} ${S}" width="100%" style="max-width:420px;margin-top:0.6rem">
    <line x1="${pad}" y1="${S - pad}" x2="${S - pad}" y2="${pad}"
          stroke="var(--muted)" stroke-dasharray="3 3" stroke-width="1"/>
    <line x1="${pad}" y1="${S - pad}" x2="${S - pad}" y2="${S - pad}" stroke="var(--line)"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${S - pad}" stroke="var(--line)"/>
    ${ticks}${marks}
    <text x="${S / 2}" y="${S - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">predicted</text>
    <text x="9" y="${S / 2}" font-size="10" fill="var(--muted)" transform="rotate(-90 9 ${S / 2})" text-anchor="middle">actual</text>
  </svg>
  <p class="legend">Dot area is the number of games. A hollow ring on the bottom edge is a bin
    that held no games.</p>`;
}
