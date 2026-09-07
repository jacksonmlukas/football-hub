#!/usr/bin/env bash
# Whether this run is still inside the window it was *scheduled* for.
#
# The watchdog measures the age of the published `live.json` against a threshold. That
# measurement only means anything while the `live` loop is running, which is only inside a
# game window: outside one nothing refreshes the overlay and its age grows without bound,
# which is the correct state rather than a failure. `watchdog.yml`'s crons are placed inside
# those windows and `tests/contracts/test_game_windows.py` holds them there -- but a cron is
# a request for a *start*, not a guarantee of one. GitHub queues scheduled workflows and
# drops them under load; `live.yml` records those starts being delivered 97-126 minutes
# apart, median 106.
#
# So a run scheduled inside a window routinely executes after it. On 2026-09-07 the run
# behind cron `*/10 0-5  * * 1` -- Monday 00:00-05:59 UTC -- executed at 09:21Z, three and a
# half hours after its window closed, measured a heartbeat nobody expected to move, and filed
# issue #193. That incident could not clear either: the next Monday-adjacent cron is Tuesday
# and the season's first game was the Wednesday, so no healthy check could occur before it.
#
# The missing context was never the threshold. It was **whether the check is still inside the
# window it was scheduled for**, which the run can answer from two things it already has: the
# cron GitHub hands it in `github.event.schedule`, and the clock.
#
# The cron string is the window definition and stays the only one -- the same rule `live.yml`
# reads its league off the triggering cron for. Nothing here restates which hours a window
# covers.
#
#   window_state.sh <cron> <season-opens-iso-date> [now-epoch-seconds]
#
# Prints one of:
#
#   preseason <iso-date>   before the season's first game there is no window at all, so there
#                          is nothing to be late for and nothing to measure
#   inside <seconds>       inside the window, which closes in <seconds>; measure
#   late <seconds>         the window closed <seconds> ago; report, do not measure
#   early <seconds>        the window opens in <seconds>; report, do not measure
#   unscheduled            no cron behind this run -- a dispatch; measure, an operator asked
#   unparseable            the cron cannot be read; measure, and say so
#
# **Both fall-throughs point at monitoring.** An unreadable cron, an absent season date and a
# season date already past all mean "measure", never "stay quiet". A monitor that silences
# itself on a case it does not understand is the failure this repo has already paid for
# twice: the check that could not reach its healthy branch, and the label whose absence
# exited the whole job. Suppression here is only ever the narrow, positively-established
# case.
set -uo pipefail

cron=${1-}
season_opens=${2-}
now=${3:-$(date -u +%s)}

case "$now" in ''|*[!0-9]*) echo "unparseable"; exit 0 ;; esac

# GNU `date` reads an epoch with `-d @N`; BSD `date` with `-r N`. The watchdog runs on
# ubuntu-latest and the contract tests run wherever the developer is, so this takes both.
_at() { date -u -r "$1" +"$2" 2>/dev/null || date -u -d "@$1" +"$2"; }

# --- before the season's first game there is no window ------------------------
#
# Checked first and ahead of the cron, because it is the stronger statement: the crons fire
# every week of the year, so in August a run can be perfectly inside its window and still be
# watching an overlay no game will move. "Inside the window" is not the same claim as "there
# is something to watch".
#
# An absent or unreadable date, or one already in the past, suppresses nothing -- so this
# rots toward monitoring rather than toward silence when nobody updates it next August.
today=$(_at "$now" %Y-%m-%d)
case "$season_opens" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9])
    if [[ "$today" < "$season_opens" ]]; then
      echo "preseason $season_opens"
      exit 0
    fi ;;
  "") : ;;
  *)  echo "window_state: '$season_opens' is not an ISO date; assuming the season is under way" >&2 ;;
esac

# --- the window this run was scheduled for ------------------------------------

[ -n "$cron" ] || { echo "unscheduled"; exit 0; }

read -r _minute hourfield dom mon dowfield _rest <<<"$cron"
if [ -z "${dowfield:-}" ] || [ -n "${_rest:-}" ] || [ "$dom" != "*" ] || [ "$mon" != "*" ]; then
  # A window cron names days of the week only -- the same shape
  # `tests/contracts/test_game_windows.py` asserts. Anything else is not one of ours, and
  # guessing at it would be inventing a window.
  echo "unparseable"
  exit 0
fi

# Expand `16-23`, `0-5`, `3`, `*`, or a comma-separated mix. A step (`*/2`) is refused rather
# than widened away: reading `*/2` as "every hour" would report a run as inside a window it
# was never scheduled in, which is the direction that files false incidents.
_expand() {  # _expand <field> <count-that-*-means>
  local field=$1 span=$2 part lo hi i out=""
  local -a parts
  IFS=',' read -ra parts <<<"$field"
  for part in "${parts[@]}"; do
    case "$part" in *"/"*) return 1 ;; esac
    if [ "$part" = "*" ]; then
      for ((i = 0; i < span; i++)); do out+="$i "; done
      continue
    fi
    case "$part" in
      *-*) lo=${part%%-*}; hi=${part##*-} ;;
      *)   lo=$part; hi=$part ;;
    esac
    case "$lo" in ''|*[!0-9]*) return 1 ;; esac
    case "$hi" in ''|*[!0-9]*) return 1 ;; esac
    [ "$lo" -le "$hi" ] || return 1
    for ((i = lo; i <= hi; i++)); do out+="$i "; done
  done
  [ -n "$out" ] || return 1
  printf '%s' "$out"
}

hours=$(_expand "$hourfield" 24) || { echo "unparseable"; exit 0; }
dows=$(_expand "$dowfield" 7)    || { echo "unparseable"; exit 0; }
# cron writes Sunday as either 0 or 7.
dows=" ${dows// 7 / 0 } "
hours=" $hours "

_in_window() {  # _in_window <epoch>
  local d h
  d=$(_at "$1" %w)
  h=$((10#$(_at "$1" %H)))
  [ "$d" -eq 7 ] && d=0
  case "$dows" in *" $d "*) : ;; *) return 1 ;; esac
  case "$hours" in *" $h "*) return 0 ;; *) return 1 ;; esac
}

hour_start=$((now - now % 3600))

if _in_window "$now"; then
  # How much of the window is left, so a log line can say it. Walked forward an hour at a
  # time rather than read off the field's maximum, because a window that wraps midnight is
  # two crons and this one only knows its own.
  probe=$((hour_start + 3600))
  for _ in $(seq 1 168); do
    _in_window "$probe" || break
    probe=$((probe + 3600))
  done
  echo "inside $((probe - now))"
  exit 0
fi

# Outside it: which side, and by how much. Both directions are walked so a run is never
# reported as six days late when it is in fact a few minutes early.
since_close=""
probe=$hour_start
for _ in $(seq 1 168); do
  probe=$((probe - 3600))
  if _in_window "$probe"; then since_close=$((now - (probe + 3600))); break; fi
done

until_open=""
probe=$((hour_start + 3600))
for _ in $(seq 1 168); do
  if _in_window "$probe"; then until_open=$((probe - now)); break; fi
  probe=$((probe + 3600))
done

if [ -n "$since_close" ] && { [ -z "$until_open" ] || [ "$since_close" -le "$until_open" ]; }; then
  echo "late $since_close"
elif [ -n "$until_open" ]; then
  echo "early $until_open"
else
  # A cron that matches no hour of any week. Not reachable from a cron this file accepted,
  # and reported as unreadable rather than as a window, because the one thing it must not do
  # is claim a window exists.
  echo "unparseable"
fi
