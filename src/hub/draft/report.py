"""The draft-night output, as lines rather than as prints.

Every function here takes what it needs and returns `list[str]`. None of them print, and
none of them read global state. That is the whole point: these blocks used to live inline in
`board.main`, where they were 124 statements that no test could reach, and the ECR-only
crash of 2026-08-25 -- two report sections reading `adp` while guarding on a different flag
-- shipped inside them and was found by running the CLI, not by the suite.

`live.py` already had this shape (`view()` builds, `render()` returns lines) and its report
path is covered. Two adapters wanted the same seam and only one had it.

The formats deliberately stay different between the two. `live` redraws a compact screen on
a timer under a line cap; this is a one-shot verbose report. What they share is that the
*decision about what to say* is testable in both.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from hub.draft import durability

if TYPE_CHECKING:                      # both import from `board`, so runtime would cycle
    from hub.draft.board import BuildReport
    from hub.draft.optimize import ThePick

SOS_GAP = 0.15
SOS_ADP_WINDOW = 8


def degraded(stages: tuple[str, ...]) -> list[str]:
    return [f"  built without: {', '.join(stages)}"] if stages else []


def mistyped(suggestions: dict[str, str | None]) -> list[str]:
    """A misspelt pick leaves that player on the board as available, and the next
    recommendation can hand back someone already drafted. This is the sharp edge of typing
    picks, which is the primary path -- ESPN publishes nothing mid-draft for a mock room."""
    out: list[str] = []
    for name, guess in suggestions.items():
        if guess:
            out.append(f"\n  NOT ON THE BOARD: {name!r} -- did you mean {guess!r}?")
            out.append(f"    until that is fixed, {guess!r} is still shown as available")
        else:
            out.append(f"\n  not on the board: {name!r} (kicker or defence, most likely)")
    return out


def header(board: pl.DataFrame) -> list[str]:
    """Summary only. Never render the frame -- that is the token rule in CLAUDE.md."""
    return [f"\n  {board.height} players | {board['vor'].null_count()} missing xFP"]


def regression(board: pl.DataFrame, n: int = 8) -> list[str]:
    top = board.filter(pl.col("fp_over_expected").is_not_null()).sort("fp_over_expected")
    out = ["\n  Biggest positive regression candidates (underperformed expectation):"]
    for r in top.head(n).iter_rows(named=True):
        out.append(f"    {r['player']:<24} {r['pos'] or '':<4} ECR {r['ecr']:>5.1f}  "
                   f"xFP-FP {-r['fp_over_expected']:>6.1f}")
    return out


def td_luck(board: pl.DataFrame, report: BuildReport) -> list[str]:
    """Players priced on touchdowns their yardage does not support.

    A different cut from xFP-FP, and measurably so -- the two correlate at +0.16 on the live
    board, and they are signed in opposite directions, so a real overlap would show as a
    strong negative. See docs/td-luck.md.

    Both flags, not just `td_luck`: this reaches nflverse and can succeed while ESPN ADP
    fails, and the filter below reads `adp`. That was the ECR-only crash.

    Takes the whole `BuildReport` rather than the two booleans it needs. Destructuring it at
    the call site is what let two consumers pick different flag combinations in the first
    place, and a sixth stage should not change this signature.
    """
    if not (report.td_luck and report.adp):
        return []
    pool = board.filter(pl.col("td_luck").is_not_null()
                        & pl.col("adp").is_not_null() & (pl.col("adp") <= 120))
    if pool.height < 8:
        return []
    out = [f"\n  Touchdown luck, last season's actuals -- {pool.height} drafted players",
           "  Points per game above the touchdowns their yardage supports. Touchdown rate",
           "  has no year-over-year persistence, so this is the part least likely to repeat."]
    for label, frame in (("FADE", pool.sort("td_luck", descending=True).head(5)),
                         ("BUY ", pool.sort("td_luck").head(5))):
        for r in frame.iter_rows(named=True):
            out.append(f"    {label} {r['player']:<24} {r['pos'] or '':<4} "
                       f"ADP {r['adp']:>5.1f}  {r['td_luck']:>+6.2f}/gm")
    out.append("  Strongest for QB, where the room prices last year's touchdowns at 1.02 "
               "against")
    out.append("  volume and their true predictive weight is -0.05. Directional elsewhere.")
    return out


def injuries(board: pl.DataFrame, report: BuildReport) -> list[str]:
    """Players carrying a designation right now, and last season's fragile ones.

    Two different quantities kept visibly apart. Last season's missed games are priced into
    the projection where the market leaves a residual (QB, WR). Today's designation is not
    priced at all -- there is no history of preseason designations against outcomes to fit a
    coefficient on, so inventing one would be worse than showing the drafter the flag.

    Gated on `report.adp` alone: both halves are scoped to "inside ADP 120" and both print
    an ADP, so without that column there is nothing here to show.
    """
    if not report.adp:
        return []
    out: list[str] = []
    pool = board.filter(pl.col("adp").is_not_null() & (pl.col("adp") <= 120))
    if "injury_status" in pool.columns:
        hurt = pool.filter(
            pl.col("injury_status").map_elements(durability.is_flagworthy,
                                                 return_dtype=pl.Boolean)
        ).sort("adp")
        if hurt.height:
            out += [f"\n  Carrying a designation today -- {hurt.height} inside ADP 120",
                    "  Out/Doubtful/IR are priced (-1.63 ppg, fitted on week-1 reports).",
                    "  QUESTIONABLE is not: 12.6% of the August board carries it against",
                    "  2.9% at week 1, so the fitted number is from a much sicker group."]
            for r in hurt.head(8).iter_rows(named=True):
                st = str(r["injury_status"]).upper()
                beta = durability.INJURY_BETA.get(st)
                note = f"{beta:+.2f} ppg" if beta else "not priced"
                out.append(f"    {r['player']:<24} {r['pos'] or '':<4} "
                           f"ADP {r['adp']:>5.1f}  {st:<15} {note}")
    if "missed" in pool.columns:
        frail = pool.filter(pl.col("missed").is_not_null()
                            & (pl.col("missed") >= 4)).sort("missed", descending=True)
        if frail.height:
            out += ["\n  Missed time last season -- priced for QB and WR only",
                    "  Running backs are left alone: the market already discounts them."]
            for r in frail.head(6).iter_rows(named=True):
                pos = r["pos"] or ""
                beta = durability.BETA.get(pos, 0.0)
                note = f"{beta * r['missed']:+.2f} ppg" if beta else "not priced"
                out.append(f"    {r['player']:<24} {pos:<4} ADP {r['adp']:>5.1f}  "
                           f"missed {int(r['missed']):>2}  {note}")
    return out


def sos(board: pl.DataFrame) -> list[str]:
    """Weeks 15-17 strength of schedule, and the swaps it makes actionable."""
    pool = board.filter(pl.col("adp").is_not_null() & pl.col("wk15_17_sos").is_not_null())
    out = [f"\n  Weeks 15-17 strength of schedule -- {pool.height} drafted players",
           "  1.00 = league-average defence for that position; higher is softer.\n"]
    for label, frame in (("SOFTEST", pool.sort("wk15_17_sos", descending=True).head(8)),
                         ("HARDEST", pool.sort("wk15_17_sos").head(8))):
        out.append(f"  {label}:")
        for r in frame.iter_rows(named=True):
            out.append(f"    {r['player']:<24} {r['pos']:<3} {r['team'] or '':<4} "
                       f"ADP {r['adp']:>6.1f}  SoS {r['wk15_17_sos']:.3f}")
        out.append("")
    # The actionable form: players the room prices the same, whose playoff slates differ.
    # Inside a tier this is a free upgrade; across tiers it is not.
    out.append(f"  Same-tier swaps (ADP within {SOS_ADP_WINDOW}, SoS gap > {SOS_GAP}):")
    rows = list(pool.sort("adp").iter_rows(named=True))
    shown = 0
    for i, x in enumerate(rows):
        for y in rows[i + 1:]:
            if y["adp"] - x["adp"] > SOS_ADP_WINDOW:
                break
            if x["pos"] == y["pos"] and abs(x["wk15_17_sos"] - y["wk15_17_sos"]) > SOS_GAP:
                hi, lo = (x, y) if x["wk15_17_sos"] > y["wk15_17_sos"] else (y, x)
                out.append(f"    {hi['player']:<22} ({hi['wk15_17_sos']:.2f}) over "
                           f"{lo['player']:<22} ({lo['wk15_17_sos']:.2f})  "
                           f"{hi['pos']}, ADP {hi['adp']:.0f} vs {lo['adp']:.0f}")
                shown += 1
                break
        if shown >= 8:
            break
    return out


def unmatched(missing: list[str]) -> list[str]:
    if not missing:
        return []
    return [f"\n  {len(missing)} recorded picks matched nobody on the board "
            f"(K/DST are excluded by design): {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")]


def corrections_note(tp: ThePick) -> list[str]:
    """What to say about the bracketed notes, which depends on the route that chose the pick.

    Only the corrected route folds them into the ranking. On the ECR fallback there is no ADP
    for a correction to move a player *relative to*, and "bounded at 20% of ADP" printed
    against a board with no ADP is the kind of sentence that gets believed at 9pm.
    """
    if "corrected" in tp.via:
        return ["    Corrections are where our measurements say the market is wrong. They",
                "    are folded into the ranking, bounded at 20% of ADP, and shown here",
                "    because the size matters to you even where the clamp limits the move."]
    if tp.notes:
        return ["    The bracketed notes are measurements, shown but NOT folded into this",
                "    ranking: corrections move a player relative to ADP, and this board",
                "    has none. Weigh them yourself."]
    return []


def the_pick(tp: ThePick | None) -> list[str]:
    """The market picks, and nothing else does.

    P0b measured championship equity against consensus-following on realised outcomes across
    2022-25 and it lost by 19.66 points per team game, losing in all four seasons. Per the
    rule fixed before that run, equity leaves this output. See docs/adr/0009.
    """
    if tp is None:
        return ["\n  THE PICK unavailable -- the board carries neither ADP nor ECR, which "
                "means it did not build. Serve site/data/draft_board.json instead."]
    return [f"\n  THE PICK -- best available filling a need, by {tp.via}",
            f"    {tp.player}  {tp.pos or ''}  "
            + (f"{tp.rank_label} {tp.rank:.1f}" if tp.rank is not None else "")
            + (f"   [{'; '.join(tp.notes)}]" if tp.notes else ""),
            *corrections_note(tp)]


def also_close(mode: str, rec: pl.DataFrame) -> list[str]:
    """Context, not a recommendation.

    VOR ordering was measured 5.06 points a team-game worse than the market
    (docs/market-value.md), so this shows what else is close -- and how close -- rather than
    something to draft off.
    """
    rule = ("long wait ahead: take who will not survive it"
            if mode == "scarcity" else "short wait ahead: take the highest VOR")
    out = [f"\n  Also close, for context -- not a ranking to draft off ({mode}: {rule})"]
    for r in rec.iter_rows(named=True):
        cw = r.get("cost_of_waiting")
        extra = f"cost_of_waiting {cw:>5.1f}" if cw is not None else ""
        v, sos_v = r["vor"], r.get("wk15_17_sos")
        tag = f"  SoS {sos_v:.2f}" if sos_v is not None else ""
        out.append(f"    {r['player']:<24} {r['pos'] or '':<4} "
                   f"VOR {0.0 if v is None else v:>5.1f}  {extra}{tag}")
    return out


def slots(teams: int, my_slot: int, starters: dict[str, int], drafted: tuple[str, ...],
          picks: list[int], waits: list[int], mode_for: Any) -> list[str]:
    """League shape and your pick schedule. Touches no network and no board."""
    return [f"  teams {teams} | slot {my_slot} | starters "
            + " ".join(f"{k}{v}" for k, v in starters.items()),
            f"  drafted positions: {'/'.join(drafted)}",
            f"  picks: {', '.join(map(str, picks[:8]))} ...",
            f"  waits: {', '.join(map(str, waits[:7]))} ...",
            *(f"    pick {pk:>3}  ->  {mode_for(pk)}" for pk in picks[:6])]
