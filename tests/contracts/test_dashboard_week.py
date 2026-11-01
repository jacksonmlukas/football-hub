"""The This week view leads with the lineup call, its swaps and their odds (#484).

Each property is a function of a module run in node or of the page's real `render()` against a
stub DOM (`dashboard_render.mjs`), on fixtures built here:

* the swap cards state a forced swap's reason and an ordinary swap's odds as a sentence, and below
  0.6 the sentence reads as a coin flip;
* the heading states the decision, and the best lineup's dots are the published quantiles;
* an artifact without `swaps` or `quantiles` draws the pre-rebuild body, not an empty card;
* the withheld note stays in the unconfirmed state, and the pipeline health replaces the footer.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
JS = SITE / "js"
RENDER = Path(__file__).resolve().parent / "dashboard_render.mjs"
HOSTILE = "<script>alert(1)</script>"


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _mod(rel: str) -> str:
    return json.dumps((JS / rel).as_uri())


def _run(node: str, body: str) -> str:
    got = subprocess.run([node, "--input-type=module", "-e", body],
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def _render(node: str, ov: dict) -> dict:
    got = subprocess.run([node, str(RENDER)], input=json.dumps({"site": str(SITE), "ov": ov}),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def _row(player, pos, team, mu=10.0, best=False, **kw):
    return {"player": player, "pos": pos, "nfl_team": team, "mu": mu, "sd": 5.0,
            "projected": True, "starting": False, "best_start": best, "available": True,
            "quantiles": [float(i) for i in range(1, 21)], "quantiles_reason": None, **kw}


ROSTER = {
    "name": "roster", "shape": "rows", "n": 5, "week": 6,
    "generated_at": "2026-10-17T16:58:11+00:00",
    "set_total": 80.0, "optimal_total": 85.0, "gain": 5.0,
    "withheld": ["Hurt Back"], "withheld_reasons": {"Hurt Back": "on injured reserve"},
    "sit": ["Tuten", "Hurt Back"], "start": ["Chase", "Fill In"],
    "swaps": [
        {"out": "Tuten", "in": "Chase", "gain": 4.0, "forced": False, "p_in_beats_out": 0.8},
        {"out": "Hurt Back", "in": "Fill In", "gain": -1.0, "forced": True,
         "p_in_beats_out": None},
    ],
    "odds_reason": None,
    "unprojected": ["Kicker K"], "unprojected_reasons": {"Kicker K": "K: outside the board's scope"},
    "rows": [_row("Tuten", "RB", "JAX"), _row("Chase", "WR", "CIN", 14.0, True),
             _row("Fill In", "RB", "DAL", 9.0, True), _row("Bench Guy", "TE", "KC", 5.0),
             _row("Hurt Back", "RB", "MIA", available=False)],
}


def _cards(node: str, ros: dict) -> str:
    return _run(node, f"import {{ swapCards }} from {_mod('panels/lineup.js')};\n"
                      f"const r = {json.dumps(ros)};\n"
                      "console.log(swapCards(r, new Map(r.rows.map(x => [x.player, x]))));")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_a_forced_and_an_ordinary_swap_each_say_their_sentence(node):
    out = _text(_cards(node, ROSTER))
    assert "Chase outscores Tuten in 8 of 10 weeks like this one." in out
    assert "Hurt Back is withheld: on injured reserve. Not a choice, so no odds." in out
    assert out.count("outscores") == 1, "the forced swap carries no odds sentence"
    html = _cards(node, ROSTER)
    assert html.count('class="swap-card"') == 2
    assert html.index("<s>") < html.index("&rarr;"), "the outgoing player is struck through"
    assert 'data-state="frozen"' in html and "+4.0 pts" in html
    assert 'data-state="unconfirmed"' in html


def test_below_point_six_the_sentence_reads_as_a_coin_flip(node):
    for p, said in ((0.55, "6 of 10"), (0.2, "2 of 10")):
        ros = json.loads(json.dumps(ROSTER))
        ros["swaps"][0]["p_in_beats_out"] = p
        out = _text(_cards(node, ros))
        assert f"in {said} weeks like this one: close to a coin flip, not a recommendation." in out
    ros = json.loads(json.dumps(ROSTER))
    ros["swaps"][0]["p_in_beats_out"] = 0.6
    assert "coin flip" not in _cards(node, ros), "0.6 is the first recommendation"


def test_a_swap_with_no_odds_says_why_in_the_unconfirmed_state(node):
    ros = json.loads(json.dumps(ROSTER))
    ros["swaps"][0]["p_in_beats_out"] = None
    ros["odds_reason"] = "odds unavailable: no correlation"
    html = _cards(node, ros)
    assert "odds unavailable: no correlation" in html and 'data-state="unconfirmed"' in html


def test_a_hostile_name_in_a_card_is_escaped(node):
    ros = json.loads(json.dumps(ROSTER))
    ros["swaps"][0]["in"] = HOSTILE
    ros["rows"].append(_row(HOSTILE, "WR", HOSTILE))
    assert "<script>" not in _cards(node, ros)


def _heading(node: str, ros: dict) -> str:
    return _run(node, f"import {{ weekHeading }} from {_mod('panels/week.js')};\n"
                      f"console.log(weekHeading({json.dumps(ros)}, null));")


def test_the_heading_states_the_decision(node):
    assert _heading(node, ROSTER) == "Week 6: make two changes"
    one = dict(ROSTER, swaps=ROSTER["swaps"][:1])
    assert _heading(node, one) == "Week 6: make one change"
    assert _heading(node, dict(ROSTER, gain=0.0, swaps=[])) == \
        "Week 6: your lineup is already the best one"
    assert _heading(node, dict(ROSTER, gain=None)) == "Week 6: no lineup could be filled"
    gap = dict(ROSTER, incomplete={"empty_slots": ["TE"]})
    assert _heading(node, gap) == "Week 6: fill the empty slot"


def test_the_best_lineup_plots_the_published_quantiles_and_tints_incoming(node):
    html = _run(node, f"import {{ bestLineupBody }} from {_mod('panels/week.js')};\n"
                      f"console.log(bestLineupBody({json.dumps(ROSTER)}));")
    best = html[:html.index("<details")]
    assert best.count('class="bl-dot"') == 40, "20 dots for each of the two best-lineup players"
    assert 'cx="10.0"' in html and 'cx="200.0"' in html, "the quantiles 1 and 20 on a 0-40 axis"
    assert 'x1="140.0"' in html, "the projection line sits at mu = 14 on the same axis"
    assert best.count("bl-incoming") == 2
    bench = html[html.index("<details"):html.index("</details>")]
    assert "<details class" in bench and " open" not in bench[:20]
    assert "Bench Guy" in bench and "Chase" not in bench
    assert "Kicker K" in html and "outside the board" in html


def test_a_player_without_quantiles_gives_the_reason_instead_of_a_plot(node):
    ros = json.loads(json.dumps(ROSTER))
    ros["rows"][1]["quantiles"] = None
    ros["rows"][1]["quantiles_reason"] = "bye: no game this week"
    html = _run(node, f"import {{ bestLineupBody }} from {_mod('panels/week.js')};\n"
                      f"console.log(bestLineupBody({json.dumps(ros)}));")
    assert "bye: no game this week" in html


def _man(**roster_fields):
    return {"season": 2026, "week": 6, "generated_at": "2026-10-17T16:58:00+00:00", "artifacts": [
        {"name": "roster", "present": True, "stale": False, "reason": None,
         "generated_at": "2026-10-17T16:58:11+00:00", **roster_fields},
        {"name": "survivor", "present": True, "stale": True, "reason": "CFBD quota exhausted",
         "generated_at": "2026-10-10T16:58:11+00:00"}]}


def test_the_rebuilt_view_renders_heading_cards_best_lineup_and_the_withheld_note(node):
    got = _render(node, {"manifest": _man(), "roster": ROSTER})
    assert got["head"] == {"text": "Week 6: make two changes", "hidden": False}
    body = got["panels"]["p-lineup"]["body"]
    assert "swap-card" in body and 'class="swaps"' not in body
    assert '<p class="state-unconfirmed">Withheld as unavailable:' in body
    assert got["panels"]["p-best"]["hidden"] is False


@pytest.mark.parametrize("strip", ["swaps", "quantiles"])
def test_an_artifact_without_swaps_or_quantiles_renders_the_pre_rebuild_body(node, strip):
    ros = json.loads(json.dumps(ROSTER))
    if strip == "swaps":
        del ros["swaps"]
    else:
        for r in ros["rows"]:
            del r["quantiles"]
    got = _render(node, {"manifest": _man(), "roster": ros})
    body = got["panels"]["p-lineup"]["body"]
    assert body.startswith('<div class="lineup-gain">+5.0') and '<ul class="swaps">' in body
    assert "swap-card" not in body
    assert got["head"]["hidden"] is True and got["panels"]["p-best"]["hidden"] is True


def test_the_pipeline_health_chips_every_artifact_with_its_age(node):
    body = _render(node, {"manifest": _man(), "roster": ROSTER})["panels"]["p-health"]["body"]
    for name in ("roster", "survivor", "bigten", "cfbd"):
        assert f"<b>{name}</b>" in body
    assert body.count('data-state="stale"') == 1 and "CFBD quota exhausted" in body
    assert 'data-state="frozen"' in body
    assert body.count('data-state="unconfirmed"') == 2, "bigten and cfbd were never fetched"
    assert 'class="chip-age"' in body


def test_the_footer_status_line_gives_way_to_the_health_on_this_view():
    css = (SITE / "index.html").read_text()
    assert re.search(r"main:has\(#v-week:not\(\[hidden\]\)\)\s*>\s*footer\s*\{\s*display:\s*none", css)
