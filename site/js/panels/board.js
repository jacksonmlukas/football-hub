// The draft board's table (#348 decides whether it shows).
import { fmt } from "../format.js";
import { table } from "../table.js";

export function boardBody(rows) {
  return table([
    { label: "player", get: r => r.player ?? "" },
    { label: "pos", get: r => r.pos ?? "" },
    { label: "VOR", get: r => fmt(r.vor, 1) },
    { label: "edge", get: r => fmt(r.edge, 1), cls: r => r.edge > 0 ? "good" : "" },
    { label: "SoS", get: r => fmt(r.wk15_17_sos, 2) },
  ], rows);
}
