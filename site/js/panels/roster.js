// The roster table: the full roster under the lineup call.
import { fmt } from "../format.js";
import { table } from "../table.js";

export function rosterBody(rows) {
  return table([
    { label: "", get: r => r.starting ? "S" : "", cls: r => r.starting ? "good" : "" },
    { label: "pos", get: r => r.pos ?? "" },
    { label: "team", get: r => r.nfl_team ?? "" },
    { label: "player", get: r => r.player ?? "" },
    { label: "proj", get: r => r.projected ? fmt(r.mu, 1) : "\u2014" },
    { label: "+/-", get: r => r.projected ? fmt(r.sd, 1) : "" },
    // The other two sources for this week (#492): consensus rank, and ESPN's projection
    // for the scoring period. Absent is a dash, never a zero.
    { label: "ecr", get: r => r.ecr == null ? "—" : fmt(r.ecr, 0) },
    { label: "espn wk", get: r => r.espn_week_proj == null ? "—" : fmt(r.espn_week_proj, 1) },
    { label: "best XI", get: r => r.best_start ? "\u2713" : "",
      cls: r => r.best_start && !r.starting ? "good" : "" },
    // A game that has kicked off closes the call (#516); `locked` is on the row only then.
    { label: "status", get: r =>
        r.locked === true ? "locked: game started"
          : r.available === false ? `out ${r.missing_games} gm`
          : ["ACTIVE", "NORMAL", ""].includes(r.injury_status) ? "" : r.injury_status,
      cls: r => r.locked !== true && r.available === false ? "bad" : "" },
  ], rows);
}
