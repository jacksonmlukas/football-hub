"""Both writers of the live overlay produce the same shape.

Two processes write `site/data/live.json` and they wrote different documents. The site writer
produced the artifact envelope every other panel uses -- `generated_at`, `n`, `rows`. The
poller produced `{ts, games, detail}`. The page reads `rows` and `generated_at`, so the poller
-- the thing a Sunday exists to run -- would have overwritten the file with a document the
page cannot read, and every game would have fallen to "not on the board" under a caption
reading "Live scores unavailable".

**The dashboard worked because nothing was polling.** That is also why the watchdog was reading
`ts`: it was written against the poller's document while the site was written against the
publisher's, and neither knew the other existed.

Nothing asserted that two producers of one file agreed, so nothing could notice. This is that
assertion, and it is a contract test rather than a unit test because the defect is not inside
either producer -- each is self-consistent -- but between them.
"""
import json

import pytest

from hub import jsonio, publish

# What `site/index.html` reads off the live artifact. Top-level first, then per row: the page
# keys games by `away`/`home`, prints the two scores, and falls back from `detail` to `state`
# for the status cell.
PAGE_NEEDS = ("generated_at", "rows")
PAGE_NEEDS_PER_ROW = ("away", "home", "away_score", "home_score", "detail", "state")

_ROW = {"id": "1", "state": "in", "detail": "Q3 4:12",
        "home": "PHI", "home_score": "21", "away": "DAL", "away_score": "17",
        "possession": "1", "down_distance": "2nd & 7"}


def _from_the_site_writer(tmp_path, monkeypatch):
    monkeypatch.setattr("hub.fetch.espn.live_state", lambda league="nfl": [_ROW])
    got = publish.live(out=tmp_path)
    assert got is not None
    return json.loads((tmp_path / "live.json").read_text())


def _from_the_poller(tmp_path, monkeypatch):
    from hub.fetch import espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [_ROW])
    monkeypatch.setattr(espn, "summary", lambda gid, league="nfl": {
        "winprobability": [{"homeWinPercentage": 0.71}]})
    out = tmp_path / "live.json"
    espn.poll_once(out=out, league="nfl", watch=["1"])
    return json.loads(out.read_text())


@pytest.fixture(params=["site writer", "poller"])
def written(request, tmp_path, monkeypatch):
    build = {"site writer": _from_the_site_writer, "poller": _from_the_poller}[request.param]
    return build(tmp_path, monkeypatch)


def test_the_page_can_read_whatever_wrote_it(written):
    missing = [k for k in PAGE_NEEDS if k not in written]
    assert not missing, f"the page reads {missing} and this producer does not write them"
    assert written["rows"], "a producer with games must write rows, not an empty overlay"
    for row in written["rows"]:
        gone = [k for k in PAGE_NEEDS_PER_ROW if k not in row]
        assert not gone, f"the page reads {gone} off every live row"


def test_the_watchdog_can_read_whatever_wrote_it(written):
    """The heartbeat is `generated_at` for both. It used to be `ts`, which only one wrote."""
    from datetime import datetime
    assert datetime.fromisoformat(written["generated_at"])


def test_both_producers_agree_on_the_envelope(tmp_path, monkeypatch):
    """Not merely 'both carry what the page needs' -- the same keys, so a reader written
    against one is written against the other."""
    site = _from_the_site_writer(tmp_path / "a", monkeypatch)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    poller = _from_the_poller(tmp_path / "b", monkeypatch)
    assert set(site) == set(poller)


def test_the_pollers_win_probabilities_survive_the_shared_shape(tmp_path, monkeypatch):
    """The reason this is not simply deleting one writer. The poller's tier-2 fan-out is the
    only source of per-game win probability, and matching shapes must not cost it."""
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    got = _from_the_poller(tmp_path / "b", monkeypatch)
    assert got["detail"]["1"]["home_win_prob"] == 0.71


def test_the_envelope_is_shared_without_closing_a_cycle():
    """Sharing a shape must not cost a cycle, which is the reason `jsonio` exists at all --
    see its module docstring.

    `publish -> fetch.espn` is the existing direction and is fine: the site writer fetches a
    scoreboard. The reverse would close the loop, and it is the edge a caller reaches for when
    the envelope lives in `publish` and the poller needs it. It does not live there."""
    import ast
    import pathlib
    espn = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "fetch" / "espn.py"
    imports = {n.module for n in ast.walk(ast.parse(espn.read_text()))
               if isinstance(n, ast.ImportFrom) and n.module}
    assert "hub.publish" not in imports, "the poller reaches back into the site writer"
    assert hasattr(jsonio, "artifact"), "the envelope has no shared home"
