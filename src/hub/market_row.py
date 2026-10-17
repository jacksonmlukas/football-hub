"""The betting market's own price, and the kickoff, on every published prediction row (#482).

The Games view sorts by kickoff and sets the model against the betting market, and it can do neither
from a row that carries only `home_win_prob`: the schedule is not on the page and, while every
row's `model` is `market_baseline`, the betting market's price *is* `home_win_prob`, so the page needs
the price as a field of its own to know there is no gap to draw.

**Written at the publish boundary, once per row.** `hub.publish.predictions` calls `attach`
on the rows it is about to write. Nothing here is read by a model (the fit is untouched) and
nothing here is touched by the live refresh: a row that already carries `market_prob` is left
alone, so a carried-forward row keeps what it was published with.

Fields, all always present (a reader tests `null`, not for a missing key):

* `kickoff` -- the scheduled start, ISO 8601 with a UTC offset, from the schedule
  (`schedule.kickoff_expr`). `kickoff_missing` says why when it is null.
* `market_spread`, `market_total` -- the line and total of the capture that priced the row.
* `market_prob` -- the spread-normal home-win probability of `market_spread`, computed with
  `hub.models.market`'s own CDF and `MARGIN_SD`, so on a `market_baseline` row it equals
  `home_win_prob` (a test holds that to 1e-9).
* `market_prob_ml` -- the de-vigged moneyline home probability the capture carried (#495), null
  where the capture carried none. A *different* estimate of the same quantity, not a fallback
  for `market_prob`.
* `market_missing` -- why the betting-market fields are null, or null.
* `priced_at` / `price_source` are the row's own and are not rewritten; this reads the capture
  they name.

**Never 0, never a later number.** A row with no capture (`priced_at` null: priced from the
moving field) has null market fields and says so. A capture stamped after kickoff is in-play
and is never the published market price (#471's guard, applied again here because this reads
the capture by its stamp and not through `lines_as_of`); a kickoff that cannot be found cannot
show the capture precedes it, so the price is withheld too. Missing is `null` with a reason.

**Degrades.** A schedule that cannot be fetched or a store with no lines leaves the fields
null with the reason; publishing the prediction is never made to depend on them.

**Vig is not published.** The snapshot stores the de-vigged moneyline probability and one
side's spread price; the overround removed is not captured, so there is nothing true to put
in a `vig` field. It arrives when the capture carries both sides.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import schedule, store

MARKET_FIELDS = ("kickoff", "kickoff_missing", "market_spread", "market_total", "market_prob",
                 "market_prob_ml", "market_missing")


def _moment(value: Any) -> datetime | None:
    """A naive UTC datetime from what a row holds: a datetime, or the string a prior publish wrote."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        got = value
    else:
        try:
            got = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return got.astimezone(UTC).replace(tzinfo=None) if got.tzinfo else got


def kickoffs(season: int) -> tuple[dict[str, datetime | None], str | None]:
    """`game_id` -> naive-UTC kickoff for the season, and the reason none could be read.

    A schedule without date columns, or one that cannot be fetched, is `({}, reason)`: the
    caller publishes null kickoffs saying why, which is the degradation this repo asks for.
    """
    try:
        from hub.fetch import nflverse
        raw = nflverse.load("schedules", seasons=[season])
        if not {"gameday", "gametime"} <= set(raw.columns):
            return {}, "the schedule carries no kickoff times"
        got = raw.select("game_id", schedule.kickoff_expr())
    except Exception as e:  # unattended network call: say which failure, never raise
        return {}, f"the schedule was unavailable ({type(e).__name__})"
    return dict(zip(got["game_id"].to_list(), got["kickoff"].to_list(), strict=True)), None


def captures(season: int, base: Path | None) -> dict[tuple[str, datetime], dict[str, Any]]:
    """`(game_id, captured_at)` -> that poll's spread, total and moneyline probability."""
    try:
        polls = store.lines(season, base=base)
    except Exception:  # a store that cannot be read prices nothing; the rows say so
        return {}
    keep = [pl.col(c) if c in polls.columns else pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("close_spread", "close_total", "ml_home_prob")]
    polls = polls.select("game_id", "captured_at", *keep).drop_nulls(["game_id", "captured_at"])
    return {(r["game_id"], r["captured_at"]): r for r in polls.to_dicts()}


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def market_fields(row: Mapping[str, Any], kickoff: datetime | None, kickoff_why: str | None,
                  seen: Mapping[tuple[str, datetime], Mapping[str, Any]]) -> dict[str, Any]:
    """The fields for one row; pure, so the controls are tests of this."""
    from hub.models.market import MARGIN_SD, normal_cdf
    out: dict[str, Any] = {
        "kickoff": (kickoff.replace(tzinfo=UTC).isoformat() if kickoff else None),
        "kickoff_missing": (None if kickoff else (kickoff_why or "the game is not on the schedule")),
        "market_spread": None, "market_total": None, "market_prob": None,
        "market_prob_ml": None, "market_missing": None,
    }
    priced = _moment(row.get("priced_at"))
    if priced is None:
        out["market_missing"] = (f"no capture priced this game (price_source "
                                 f"{row.get('price_source') or 'unknown'})")
    elif kickoff is None:
        out["market_missing"] = "kickoff unknown, so the capture cannot be shown to precede it"
    elif priced > kickoff:
        out["market_missing"] = "the only capture is after kickoff (in play), never a market price"
    else:
        cap = seen.get((str(row.get("game_id")), priced))
        spread = _finite(cap.get("close_spread")) if cap is not None else None
        if cap is None or spread is None:
            out["market_missing"] = "the capture named by priced_at is not in the store"
        else:
            out["market_spread"] = spread
            out["market_total"] = _finite(cap.get("close_total"))
            out["market_prob"] = normal_cdf(spread / MARGIN_SD)
            out["market_prob_ml"] = _finite(cap.get("ml_home_prob"))
    return out


def attach(rows: list[dict[str, Any]], season: int,
           base: Path | None = None) -> list[dict[str, Any]]:
    """`rows` with the betting-market fields added to every row that does not already carry them."""
    todo = [r for r in rows if "market_prob" not in r]
    if not todo:
        return rows
    kicks, why = kickoffs(season)
    seen = captures(season, base)
    for r in todo:
        r.update(market_fields(r, kicks.get(str(r.get("game_id"))), why, seen))
    return rows
