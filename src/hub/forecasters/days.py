"""Which day's run forecasts a game, and whether a scheduled trigger should run at all.

`docs/prereg/llm-forecasters.md`, *Forecast procedure*. Both rules are Eastern wall-clock rules
and are computed through `America/New_York`, never a fixed offset: on 1 November 2026, the
Sunday daylight time ends, 17:00Z is 12:00 ET, where the Sunday before it was 13:00.

* **Run day.** A game kicking off at or after 13:00 ET on its date belongs to that date's run;
  anything earlier -- an international morning game, an early Thanksgiving game -- to the run
  the day before.
* **Window.** A run happens in [09:30, 11:30) ET. GitHub schedules in UTC, so the workflow fires
  every 15 minutes 13:30-16:30 UTC, which covers the window under either offset, and each
  trigger asks `should_run` -- inside the window, and today's run not finished -- and exits if
  not. The extra triggers are the backup for a delayed or dropped schedule.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SAME_DAY_FROM = time(13, 0)
WINDOW_OPEN = time(9, 30)
WINDOW_CLOSE = time(11, 30)

# The workflow's schedule, as (hour, minute) UTC: every 15 minutes from 13:30 to 16:30.
# `.github/workflows/forecast.yml` is held to this by a test.
CRON_TRIGGERS_UTC: tuple[tuple[int, int], ...] = tuple(
    (h, m) for h in range(13, 17) for m in (0, 15, 30, 45) if (13, 30) <= (h, m) <= (16, 30))


def eastern(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError(f"{t!r} is naive; a kickoff or trigger time must carry its zone")
    return t.astimezone(ET)


def run_day(kickoff: datetime) -> date:
    """The date of the run that forecasts a game kicking off at `kickoff`."""
    k = eastern(kickoff)
    return k.date() if k.time() >= SAME_DAY_FROM else k.date() - timedelta(days=1)


def due(kickoffs: Mapping[str, datetime], day: date) -> list[str]:
    """The game ids whose run day is `day`, in kickoff order."""
    return sorted((g for g, k in kickoffs.items() if run_day(k) == day),
                  key=lambda g: (kickoffs[g], g))


def in_window(now: datetime) -> bool:
    return WINDOW_OPEN <= eastern(now).time() < WINDOW_CLOSE


def should_run(now: datetime, *, done_days: Collection[date]) -> tuple[bool, date]:
    """(run?, today's run day in ET). False outside the window or once today's run is done."""
    today = eastern(now).date()
    return in_window(now) and today not in done_days, today
