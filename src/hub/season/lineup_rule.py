"""Which source sets the published lineup this week: the model (`mu`) or weekly consensus (#492).

**The decision.** Issue #492, adopted 2026-10-16: weekly expert consensus rank (FantasyPros ECR)
sets the lineup as a **provisional rule** under ADR-0014 from **week 8 of 2026**, until a
forward-gated model beats it. Week 7 and earlier keep the rule they were set by, `mu`
(`hub.season.roster.lock`'s best lineup by summed `mu`). What reopens it is the forward
record `hub.season.lineup_journal` keeps.

**Point in time.** The boundary is read off the roster's own scoring period (`espn_week`), never
the calendar, so replaying a week-7 roster today sets the lineup it set then. The ECR read is the
journal's: the newest capture taken before that week's deadline (`lineup_journal.ecr_ranks`).

**Nothing in the model changes.** This module chooses which *source* the lineup is set by. `mu`,
`sd` and every arm module pinned in `hub.season.weekly_forward` are read exactly as before; the
ECR lineup is searched by the same `hub.season.lineup.best_by_points` over the same pool `lock`
uses (projected, startable, not on bye, available), scored by rank instead of by `mu`.

**It degrades, it never raises.** The week's strict capture first. If it is missing, the week's
*lineup capture* (`hub.fetch.consensus --lineup`: the newest page dated after the previous week's
last game and before the week's first Sunday, taken after Thursday's games because the archive
publishes on Fridays) -- labelled as such on the artifact and the page, never as a pre-kickoff
capture. If that is missing too, the newest earlier pre-deadline strict capture of the season
(last-good ECR), saying which week that was; if there is none, or ECR ranks too few of the pool to
fill a legal lineup, the model's lineup is served and says why. The reason travels on the
artifact, the journal and the page.

**Wording (ADR-0014).** The ECR rule is labelled `provisional` everywhere it is shown and is never
described as validated or as an edge.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub.config import SEASON_AHEAD
from hub.season.roster import Lock, lock

# The decided boundary (#492): the first week of each season whose lineup ECR sets. A season
# absent here is set by `mu` throughout -- the rule was adopted for 2026 and says nothing else.
ECR_FROM_WEEK: dict[int, int] = {2026: 8}

MU, ECR = "mu", "ecr"
PROVISIONAL = "provisional"
MODEL = "model"

ECR_BASIS = ("weekly expert consensus rank (FantasyPros ECR), from the newest capture taken "
             "before the week's first game day; best legal lineup by rank among projected, "
             "startable, available players not on bye")
ECR_LINEUP_BASIS = ("weekly expert consensus rank (FantasyPros ECR), from the week's lineup "
                    "capture: the newest page dated after the previous week's last game and "
                    "before the week's first Sunday, taken after Thursday's games and so not a "
                    "pre-kickoff capture; best legal lineup by rank among projected, startable, "
                    "available players not on bye")
MU_BASIS = ("the model's weekly mean (mu): the board's blend of ESPN's season average and "
            "last season's xFP per game; best legal lineup by summed mu")
ECR_CAVEAT = ("provisional rule (ADR-0014), not a measured edge: no gate has run on it; it sets "
              "the lineup from week 8 (#492) until a forward-gated model beats it, and every "
              "week is journalled so the forward record can reopen it")


class Decision(NamedTuple):
    """What set this week's lineup, and why, if it is not what the rule asks for."""
    source: str                      # "ecr" or "mu": what actually set `Lock.best_lineup`
    rule: str                        # what the decided rule asks for this week
    week: int | None                 # the roster's scoring period
    ecr_from_week: int | None        # the season's switch week, if it has one
    ecr_week: int | None             # the capture's week when ECR set it (last-good if != week)
    reason: str | None               # why `source` is not the week's own ECR, when it is not
    # Set only when the week's *lineup* capture set it: `{purpose, scrape_date, captured_at,
    # note}`. Absent from `as_dict` otherwise, so every artifact a strict capture, last-good or
    # `mu` set is byte-for-byte what it was before the lineup capture existed.
    ecr_capture: dict[str, str] | None = None

    @property
    def label(self) -> str:
        return PROVISIONAL if self.source == ECR else MODEL

    def as_dict(self) -> dict[str, Any]:
        """The artifact's and the journal's shape."""
        out: dict[str, Any] = {
            "source": self.source, "rule": self.rule, "label": self.label,
            "week": self.week, "ecr_from_week": self.ecr_from_week,
            "ecr_week": self.ecr_week, "reason": self.reason,
            "basis": ((ECR_LINEUP_BASIS if self.ecr_capture else ECR_BASIS)
                      if self.source == ECR else MU_BASIS),
            "caveat": ECR_CAVEAT if self.source == ECR else None}
        if self.ecr_capture is not None:
            out["ecr_capture"] = dict(self.ecr_capture)
        return out


def rule_for(season: int, week: int | None) -> str:
    """What the decided rule asks for: ECR from the season's switch week, `mu` before it.

    A roster with no scoring period cannot be placed either side of the boundary; it keeps the
    rule every week had before the switch, and `decide` says so."""
    start = ECR_FROM_WEEK.get(season)
    if start is None or week is None or week < start:
        return MU
    return ECR


def last_good_ecr(season: int, week: int,
                  base: Path | None = None) -> tuple[dict[str, float], int | None]:
    """`{player_key: ecr}` from the newest pre-deadline capture of an *earlier* week, or `({},
    None)`. Each such capture was taken before its own week began, so before this one."""
    from hub.fetch import consensus
    from hub.names import player_key
    try:
        caps = consensus.read_captures(season, base)
    except Exception:                                # graceful degradation, per CLAUDE.md
        return {}, None
    ok = [c for c in caps if c.week < week
          and not consensus.is_late(c.captured_at, c.first_game_day)]
    if not ok:
        return {}, None
    cap = max(ok, key=lambda c: (c.week, c.captured_at))
    out: dict[str, float] = {}
    for r in cap.rows.iter_rows(named=True):
        if r["ecr"] is not None:
            out.setdefault(player_key(r["player"]), float(r["ecr"]))
    return out, cap.week


def lineup_ecr(season: int, week: int, base: Path | None = None
               ) -> tuple[dict[str, float], dict[str, str] | None]:
    """`{player_key: ecr}` from the week's newest lineup capture, and the label it is shown
    under; `({}, None)` when there is none. Only lineup captures that pass
    `consensus.lineup_problem` are read (taken before the week's first Sunday, scrape in the
    window), so a late one is never served."""
    from hub.fetch import consensus
    from hub.names import player_key
    try:
        caps = [c for c in consensus.read_lineup_captures(season, base) if c.week == week]
    except Exception:                                # graceful degradation, per CLAUDE.md
        return {}, None
    if not caps:
        return {}, None
    cap = caps[-1]                                   # earliest first
    out: dict[str, float] = {}
    for r in cap.rows.iter_rows(named=True):
        if r["ecr"] is not None:
            out.setdefault(player_key(r["player"]), float(r["ecr"]))
    return out, {"purpose": consensus.LINEUP, "scrape_date": cap.scrape_date.isoformat(),
                 "captured_at": cap.captured_at.isoformat(), "note": cap.note()}


def decide(df: pl.DataFrame, season: int = SEASON_AHEAD, week: int | None = None,
           captures: Path | None = None,
           ecr: Mapping[str, float] | None = None,
           ecr_reason: str | None = None) -> tuple[Lock, Decision]:
    """The published lineup and what set it. Never raises on a missing source.

    `week` defaults to the roster's own scoring period. `ecr`/`ecr_reason` may be passed by a
    caller that has already read the week's capture (the publisher, the journal); otherwise it
    is read here from `captures`.
    """
    from hub.season import lineup_journal
    wk = week if week is not None else lineup_journal.roster_week(df)
    start = ECR_FROM_WEEK.get(season)
    want = rule_for(season, wk)

    def model(reason: str | None) -> tuple[Lock, Decision]:
        return lock(df), Decision(MU, want, wk, start, None, reason)

    if want == MU:
        why = (f"roster carries no scoring period, so the week-{start} switch to consensus "
               "rank cannot be placed; the model's lineup (mu) is served"
               if start is not None and wk is None else None)
        return model(why)
    assert wk is not None                            # rule_for returns ECR only for a week
    if ecr is None:
        ecr, ecr_reason = lineup_journal.ecr_ranks(season, wk, captures)
    reasons: list[str] = []
    if ecr:
        got = lock(df, ranks=ecr)
        if got.best_lineup:
            return got, Decision(ECR, want, wk, start, wk, None)
        reasons.append(f"week {wk}'s consensus ranks too few of the pool to fill a legal lineup")
    else:
        reasons.append(ecr_reason or f"no consensus ranks for week {wk}")
    # The week's lineup capture, after its strict one and before last-good. It is this week's
    # page, so it is no departure and carries no reason unless a strict capture existed and
    # could not fill a lineup; its label (`ecr_capture.note`) says it was taken after Thursday.
    late, label = lineup_ecr(season, wk, captures)
    if late:
        got = lock(df, ranks=late)
        if got.best_lineup:
            why = (f"{reasons[0]}; set on the week's lineup capture" if ecr else None)
            return got, Decision(ECR, want, wk, start, wk, why, label)
        reasons.append(f"week {wk}'s lineup capture ranks too few of the pool")
    old, old_week = last_good_ecr(season, wk, captures)
    if old:
        got = lock(df, ranks=old)
        if got.best_lineup:
            return got, Decision(ECR, want, wk, start, old_week,
                                 f"{reasons[0]}; set on week {old_week}'s capture (last-good)")
        reasons.append(f"week {old_week}'s capture (last-good) ranks too few of the pool")
    else:
        reasons.append("no earlier capture to serve as last-good")
    return model("; ".join(reasons) + "; the model's lineup (mu) is served instead")
