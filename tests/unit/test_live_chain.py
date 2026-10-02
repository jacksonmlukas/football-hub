"""#391, the half that extends a running loop: at its cap the `live` loop continues while ESPN
reports a game in progress, and only on game *state*.

Fixtures are `live_state` rows; nothing here calls ESPN.
"""
import re
from pathlib import Path

import pytest

from hub import live_chain


def row(state, gid="1"):
    return {"id": gid, "state": state, "home": "PHI", "away": "DAL"}


def test_a_game_in_progress_continues_the_loop():
    """Planted: the loop's cap lands mid-game, as #390's late-started loop's did."""
    assert live_chain.decide([row("pre", "1"), row("in", "2"), row("post", "3")]) == (
        live_chain.CONTINUE)


def test_planted_no_game_in_progress_ends_the_loop():
    assert live_chain.decide([row("pre"), row("post", "2")]) == live_chain.QUIET
    assert live_chain.decide([]) == live_chain.QUIET


def test_the_monday_loop_that_ended_before_the_game_had_nothing_to_extend():
    """The honest limit of this half, replayed. The `live` loop of 2026-09-28 started 18:18Z
    and ended ~20:18Z, with PHI @ CHI (00:15Z on the 29th) still `pre`. State says no game is
    in progress, so no loop is chained -- and that is why the window was only caught by the
    watchdog half (`hub.watchdog_gap`), which has the replay that does fire."""
    assert live_chain.decide([row("pre", "phi-chi")]) == live_chain.QUIET


def test_the_decision_reads_state_and_nothing_about_the_calendar():
    """No hour, weekday or kickoff time reaches it: the crons stay the only window."""
    source = Path(live_chain.__file__).read_text()
    code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    for forbidden in ("datetime", "weekday", "strftime", "hour", "timedelta", "time."):
        assert forbidden not in code, f"{forbidden!r} is a window shape in a module that has none"


def test_the_cli_exits_zero_only_when_a_game_is_live(monkeypatch, capsys):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [row("in")])
    assert live_chain.main(["--league", "nfl"]) == 0
    assert "continue: 1 nfl game(s) in progress" in capsys.readouterr().out
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [row("post")])
    assert live_chain.main(["--league", "cfb"]) == 1
    assert "quiet" in capsys.readouterr().out


def test_the_cli_asks_for_the_league_the_loop_is_refreshing(monkeypatch):
    from hub.fetch import espn
    asked = []
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": asked.append(league) or [])
    live_chain.main(["--league", "cfb"])
    assert asked == ["cfb"]


def test_an_espn_that_cannot_be_asked_does_not_chain_a_loop(monkeypatch, capsys):
    """Unknown is not continue. A chain kept alive by an outage has nothing to end it."""
    from hub.fetch import espn

    def down(league="nfl"):
        raise RuntimeError("ESPN unreachable")
    monkeypatch.setattr(espn, "live_state", down)
    assert live_chain.main([]) == live_chain.UNKNOWN != live_chain.CONTINUE
    assert "unknown" in capsys.readouterr().out


def test_an_unknown_league_is_refused():
    with pytest.raises(SystemExit):
        live_chain.main(["--league", "mls"])
