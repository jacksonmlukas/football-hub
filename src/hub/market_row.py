"""The betting market's own price, and the kickoff, on every published prediction row (#482).

The Games view sorts by kickoff and sets the model against the betting market, and it can do neither
from a row that carries only `home_win_prob`: the schedule is not on the page and, while every
row's `model` is `market_baseline`, the betting market's price *is* `home_win_prob`, so the page
needs
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

**The vig, since #512.** The snapshot keeps both sides' median prices (`spread_price` and
`spread_price_away`; `ml_home_price` and `ml_away_price`), so the overround -- the implied
probabilities of the two sides summed, less one -- can be published for the betting market a row is
priced from:

* `vig` -- the spread market's overround, `vig_missing` saying why it is null.
* `vig_ml` -- the moneyline's, `vig_ml_missing` likewise.

-110 / -110 is 0.0476 (2 * 110/210 - 1). A side that is absent, or a price that is not American
odds, gives null with a reason; a capture that predates #512 carries neither side's partner
and says so. Nothing is imputed, and a vig is never 0 for want of data. The overround of the
*medians* is published, not a median of per-book overrounds. A row published before #512 gets
the vig read from the capture it names, and nothing else on it is rewritten.
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
                 "market_prob_ml", "market_missing", "vig", "vig_missing", "vig_ml",
                 "vig_ml_missing")
VIG_FIELDS = ("vig", "vig_missing", "vig_ml", "vig_ml_missing")


def _moment(value: Any) -> datetime | None:
    """A naive UTC datetime from a row: a datetime, or the string a prior publish wrote."""
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


_PRICE_COLUMNS = ("spread_price", "spread_price_away", "ml_home_price", "ml_away_price")


def _implied(price: float) -> float:
    """Implied probability of one American price, vig included."""
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def overround(home: Any, away: Any) -> tuple[float | None, str | None]:
    """`(overround, None)` for a two-sided market, else `(None, why)`.

    The overround is the two sides' implied probabilities summed, less one: -110 / -110 is
    2 * (110 / 210) - 1 = 0.0476. A side that is missing, or a price that cannot be American
    odds (inside the hole between -100 and +100), makes the pair unreadable, and an unreadable
    pair is null with the reason, never 0.
    """
    h, a = _finite(home), _finite(away)
    if h is None and a is None:
        return None, "neither side's price was captured"
    if h is None or a is None:
        return None, f"the {'home' if h is None else 'away'} side's price was not captured"
    if abs(h) < 100.0 or abs(a) < 100.0:
        return None, "a price is not American odds"
    return _implied(h) + _implied(a) - 1.0, None


def vig_fields(cap: Mapping[str, Any] | None, why: str | None) -> dict[str, Any]:
    """The four vig fields for one capture; `why` is the reason there is no capture to read."""
    if cap is None:
        reason = why or "no capture to read the prices from"
        return {"vig": None, "vig_missing": reason, "vig_ml": None, "vig_ml_missing": reason}
    out: dict[str, Any] = {}
    for key, home, away in (("vig", "spread_price", "spread_price_away"),
                            ("vig_ml", "ml_home_price", "ml_away_price")):
        out[key], out[f"{key}_missing"] = overround(cap.get(home), cap.get(away))
        if out[key] is None and _finite(cap.get(away)) is None:
            # The partner price is what #512 added: its absence may be a capture that predates it.
            out[f"{key}_missing"] += " (a capture before #512 predates both-sides prices)"
    return out


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


def captures_why(season: int, base: Path | None
                 ) -> tuple[dict[tuple[str, datetime], dict[str, Any]], str | None]:
    """`captures`, and the reason none could be read (a failed store is not an absent capture)."""
    try:
        polls = store.lines(season, base=base)
    except Exception as e:  # a store that cannot be read prices nothing; the rows say so
        return {}, f"the capture store was unreadable ({type(e).__name__})"
    keep = [pl.col(c) if c in polls.columns else pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("close_spread", "close_total", "ml_home_prob", *_PRICE_COLUMNS)]
    polls = polls.select("game_id", "captured_at", *keep).drop_nulls(["game_id", "captured_at"])
    return {(r["game_id"], r["captured_at"]): r for r in polls.to_dicts()}, None


def captures(season: int, base: Path | None) -> dict[tuple[str, datetime], dict[str, Any]]:
    """`(game_id, captured_at)` -> that poll's spread, total and moneyline probability."""
    return captures_why(season, base)[0]


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def market_fields(row: Mapping[str, Any], kickoff: datetime | None, kickoff_why: str | None,
                  seen: Mapping[tuple[str, datetime], Mapping[str, Any]],
                  store_why: str | None = None) -> dict[str, Any]:
    """The fields for one row; pure, so the controls are tests of this."""
    from hub.models.market import MARGIN_SD, normal_cdf
    out: dict[str, Any] = {
        "kickoff": (kickoff.replace(tzinfo=UTC).isoformat() if kickoff else None),
        "kickoff_missing": (None if kickoff else kickoff_why or "the game is not on the schedule"),
        "market_spread": None, "market_total": None, "market_prob": None,
        "market_prob_ml": None, "market_missing": None,
    }
    out.update(vig_fields(None, "the betting-market fields are null: see market_missing"))
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
            out["market_missing"] = (store_why if cap is None and store_why
                                     else "the capture named by priced_at is not in the store")
        else:
            out["market_spread"] = spread
            out["market_total"] = _finite(cap.get("close_total"))
            out["market_prob"] = normal_cdf(spread / MARGIN_SD)
            out["market_prob_ml"] = _finite(cap.get("ml_home_prob"))
            out.update(vig_fields(cap, None))
    return out


def attach(rows: list[dict[str, Any]], season: int,
           base: Path | None = None) -> list[dict[str, Any]]:
    """`rows` with the betting-market fields added to every row that does not already carry them."""
    todo = [r for r in rows if "market_prob" not in r]
    # A row published before #512 carries its market fields and not the vig. The capture it
    # names is immutable, so the vig is read from it and nothing else on the row is touched.
    vig_only = [r for r in rows if "market_prob" in r and "vig" not in r]
    if not todo and not vig_only:
        return rows
    kicks, why = kickoffs(season)
    seen, store_why = captures_why(season, base)
    for r in todo:
        r.update(market_fields(r, kicks.get(str(r.get("game_id"))), why, seen, store_why))
    for r in vig_only:
        priced = _moment(r.get("priced_at"))
        if r.get("market_prob") is None or priced is None:
            r.update(vig_fields(None, r.get("market_missing")))
            continue
        cap = seen.get((str(r.get("game_id")), priced))
        r.update(vig_fields(cap, store_why if cap is None and store_why
                            else "the capture named by priced_at is not in the store"))
    return rows
