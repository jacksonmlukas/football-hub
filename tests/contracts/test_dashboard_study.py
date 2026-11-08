"""The Study view: the LLM-vs-market leaderboard, and its designed empty state (#487).

Each property is a function of `panels/study.js` run in node, or of the page's real `render()`
against a stub DOM (`dashboard_render.mjs`), on a leaderboard built to `hub.forecasters.leaderboard`'s
schema (`tests/fixtures/site/leaderboard/leaderboard.json`, kept outside `fixtures/site/data/` so
the screenshot fixture set still has no leaderboard and shows the empty state):

* with no artifact (or an unreadable one) the view is the empty state -- "not started", the question,
  the twelve contestants as the prereg groups them, the three steps -- never an error, never blank;
* every estimate and interval drawn is a field of the artifact, and the page computes no score;
* the interim label renders unless the artifact itself says `interim: false`;
* each metric header carries its better direction;
* a contestant under the artifact's minimum game count is listed, unranked and marked preliminary;
* the blind / line-aware toggle switches the condition; baselines are greyed;
* nothing off the feed reaches the markup unescaped.
"""
# ruff: noqa: RUF001  (the minus sign and the arrows are the things under test)
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from html import unescape
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The excision harness in test_dashboard_escapes.py hands a child run a mutated copy of the page
# through this variable, so a guard marked in `site/js/panels/study.js` is proved by deleting it.
SITE = Path(os.environ.get("HUB_DASHBOARD_PAGE") or ROOT / "site" / "index.html").parent
JS = SITE / "js"
FIXTURE = ROOT / "tests" / "fixtures" / "site" / "leaderboard" / "leaderboard.json"
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


def _lb(**kw) -> dict:
    lb = json.loads(FIXTURE.read_text())
    lb.update(kw)
    return lb


def _body(node: str, lb: dict | None) -> str:
    return _run(node, f"import {{ studyBody }} from {_mod('panels/study.js')};\n"
                      f"console.log(studyBody({json.dumps(lb)}));")


def _render(node: str, ov: dict) -> dict:
    got = subprocess.run([node, str(RENDER)], input=json.dumps({"site": str(SITE), "ov": ov}),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def _table(html: str, cond: str) -> str:
    return html.split(f'data-cond-body="{cond}"', 1)[1].split("</table>", 1)[0]


def _rows(table: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in
            re.finditer(r'<tr class="[^"]*" data-contestant="([^"]*)">(.*?)</tr>', table, flags=re.S)}


def _sgn(v: float) -> str:
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.3f}"


# --- the empty state ------------------------------------------------------------

def test_with_no_leaderboard_the_view_is_the_empty_state_not_an_error_or_a_blank(node):
    for ov in ({"leaderboard": None}, {}):
        body = _render(node, ov)["panels"]["p-study"]["body"]
        out = _text(body)
        assert body.strip(), "the panel was left blank"
        assert 'data-state="unconfirmed"' in body and "not started" in out
        assert "Can a language model forecast a college football game better than the betting market?" in out
        assert "12 contestants, one benchmark" in out
        for who in ("Four LLMs, blind", "Four LLMs, line-aware", "LLM + retrieval", "Hub model",
                    "Home constant", "Coin flip", "Market (benchmark)"):
            assert who in out, who
        for issue in ("#448", "#451", "#453"):
            assert issue in out, issue
        assert "Three steps to the first score" in out
        assert "error" not in out.lower() and "undefined" not in out and "NaN" not in out
        assert "<table" not in body and "Interim" not in out


def test_an_artifact_with_no_contestants_is_also_the_empty_state(node):
    for lb in (_lb(contestants=[]), {"interim": True}, {"contestants": "x"}):
        body = _body(node, lb)
        assert "not started" in _text(body) and "<table" not in body


def test_the_empty_states_contestants_are_the_prereg_twelve(node):
    out = _run(node, f"import {{ CONTESTANTS }} from {_mod('panels/study.js')};\n"
                     "console.log(CONTESTANTS.reduce((a, c) => a + c.n, 0));")
    assert out == "12", "the prereg (restated 2026-10-22, #511) has twelve forecasters"
    prereg = (ROOT / "docs" / "prereg" / "llm-forecasters.md").read_text()
    assert "There are **twelve** forecasters" in prereg


def test_the_fixture_site_data_has_no_leaderboard_so_the_screenshot_shows_the_empty_state():
    assert not (ROOT / "tests" / "fixtures" / "site" / "data" / "leaderboard.json").exists()


# --- the artifact is the only source of a number --------------------------------

def test_every_estimate_and_interval_drawn_is_read_from_the_artifact(node):
    lb = _lb()
    html = _body(node, lb)
    rows = {**_rows(_table(html, "blind")), **_rows(_table(html, "line_aware"))}
    assert set(rows) == {r["contestant"] for r in lb["contestants"]}
    for r in lb["contestants"]:
        text = _text(rows[r["contestant"]])
        for key in ("log_loss_diff", "brier_diff"):
            iv = r[key]
            assert f"{_sgn(iv['estimate'])} [{_sgn(iv['lo'])}, {_sgn(iv['hi'])}]" in text, (r["contestant"], key)
        assert f" {r['scorable']} {r['missing']} " in f" {text} ", r["contestant"]


def test_a_changed_artifact_changes_the_page_and_a_changed_interval_moves_the_whisker(node):
    lb = _lb()
    row = next(r for r in lb["contestants"] if r["contestant"] == "llm1:blind")
    row["log_loss_diff"].update(estimate=0.123, lo=-0.456, hi=0.789)
    html = _body(node, lb)
    llm1 = _text(_rows(_table(html, "blind"))["llm1:blind"])
    assert "+0.123 [−0.456, +0.789]" in llm1
    bar = re.search(r'data-contestant="llm1:blind".*?class="lb-whisker" x1="([\d.]+)" x2="([\d.]+)"', html, re.S)
    # The axis is shared and symmetric about zero: half-width 1.1 * the widest bound drawn.
    m = 0.789 * 1.1
    want = lambda v: f"{(v + m) / (2 * m) * 160:.1f}"       # noqa: E731
    assert bar is not None and bar.groups() == (want(-0.456), want(0.789))


def test_the_page_computes_no_score():
    src = (JS / "panels" / "study.js").read_text()
    code = re.sub(r"//[^\n]*", "", src)
    assert not re.search(r"Math\.(log|sqrt|pow|exp)|\*\*|reduce\([^)]*\+", code.replace(
        "CONTESTANTS.reduce((a, c) => a + c.n, 0)", "")), "a score or an interval is computed in the page"
    assert not re.search(r"fetch\(|https?://", code)


def test_the_betting_market_is_its_own_row_at_zero_and_baselines_are_greyed(node):
    html = _body(node, _lb())
    for cond in ("blind", "line_aware"):
        t = _table(html, cond)
        assert t.count("lb-market-row") == 1 and "Market (benchmark)" in t
        rows = _rows(t)
        assert {"hub", "home_constant", "coin_flip"} <= set(rows), "no-condition rows appear in both"
        greyed = re.findall(r'<tr class="lb-row lb-base" data-contestant="([^"]*)"', t)
        assert sorted(greyed) == ["coin_flip", "home_constant"], "the hub model is exploratory, not a baseline"


# --- the interim label ----------------------------------------------------------

def test_the_interim_label_cannot_be_dropped_while_the_artifact_says_interim(node):
    for lb in (_lb(), _lb(label="something else"), {k: v for k, v in _lb().items() if k != "label"}):
        html = _body(node, lb)
        assert 'class="lb-interim"' in html and "Interim: descriptive, not a test" in _text(html)
    # The artifact's own word is the only thing that clears it, and so is a missing flag not a clear.
    no_flag = {k: v for k, v in _lb().items() if k != "interim"}
    assert "Interim: descriptive, not a test" in _text(_body(node, no_flag))
    assert "Interim:" not in _text(_body(node, _lb(interim=False)))


def test_the_rendered_page_carries_the_interim_label_and_the_through_week(node):
    body = _render(node, {"leaderboard": _lb()})["panels"]["p-study"]["body"]
    assert "Interim: descriptive, not a test · through week 11" in _text(body)


# --- headers and ranking --------------------------------------------------------

def test_each_metric_header_carries_its_better_direction(node):
    html = _body(node, _lb())
    heads = re.findall(r"<th[^>]*>(.*?)</th>", _table(html, "blind"), flags=re.S)
    metric = [h for h in heads if re.search(r"log loss|Brier|Missing", h)]
    assert len(metric) == 3
    for h in metric:
        assert "↓" in h and "lower is better" in _text(h), h


def test_rows_are_ranked_by_the_artifacts_estimate_and_a_preliminary_row_is_listed_unranked(node):
    lb = _lb(min_games=20)
    html = _body(node, lb)
    blind = _rows(_table(html, "blind"))
    order = [(c, re.findall(r'<td class="num">([^<]*)</td>', r)[0]) for c, r in blind.items()]
    est = {r["contestant"]: r["log_loss_diff"]["estimate"] for r in lb["contestants"]}
    ranked = [c for c, rank in order if rank != "—"]
    assert ranked == sorted(ranked, key=est.__getitem__)
    assert [rank for _, rank in order if rank != "—"] == [str(i + 1) for i in range(len(ranked))]
    # The retrieval arm has 12 games in its interval, under 20: preliminary, in the line-aware table.
    la = _rows(_table(html, "line_aware"))
    r = la["llm4+retrieval:line_aware"]
    assert "preliminary" in _text(r) and '<td class="num">—</td>' in r.split("</td>")[0] + "</td>"
    assert list(la)[-1] == "llm4+retrieval:line_aware" or "preliminary" in _text(list(la.values())[-1])
    assert "preliminary" not in _text(blind["llm1:blind"])


def test_without_a_minimum_game_count_in_the_artifact_nothing_is_marked_preliminary(node):
    assert "preliminary" not in _body(node, _lb())


# --- the toggle -----------------------------------------------------------------

def test_the_condition_toggle_splits_blind_from_line_aware(node):
    html = _body(node, _lb())
    blind, la = _rows(_table(html, "blind")), _rows(_table(html, "line_aware"))
    assert {c for c in blind if c.endswith(":line_aware")} == set()
    assert {c for c in la if c.endswith(":blind")} == set()
    assert sum(c.endswith(":blind") for c in blind) == 4
    assert sum(c.endswith(":line_aware") for c in la) == 5, "four LLMs and the retrieval arm"
    assert re.search(r'data-cond-body="line_aware" hidden', html)
    assert 'data-cond-body="blind" hidden' not in html


def test_choosing_a_condition_shows_its_table_and_presses_its_button(node):
    out = _run(node, f"import {{ chooseCondition }} from {_mod('panels/study.js')};\n"
                     """const mk = a => ({ a: { ...a }, hidden: a.hidden ?? false,
  getAttribute(k) { return this.a[k]; }, setAttribute(k, v) { this.a[k] = v; } });
const bs = [mk({ "data-cond": "blind" }), mk({ "data-cond": "line_aware" })];
const ds = [mk({ "data-cond-body": "blind" }), mk({ "data-cond-body": "line_aware", hidden: true })];
const root = { querySelectorAll: s => s === "[data-cond]" ? bs : ds };
bs[1].closest = () => root;
chooseCondition(bs[1]);
console.log(JSON.stringify({ p: bs.map(b => b.a["aria-pressed"]), h: ds.map(d => d.hidden) }));""")
    assert json.loads(out) == {"p": ["false", "true"], "h": [True, False]}


# --- state, escaping, the router ------------------------------------------------

def test_the_rendered_leaderboard_goes_stale_through_the_manifest_entry(node):
    man = json.loads((SITE / "data" / "manifest.json").read_text())
    # replace, not append: the live manifest has carried its own leaderboard entry since the
    # week 9 slate refresh (59df145), and the first entry by name is the one read
    man["artifacts"] = [a for a in man["artifacts"] if a["name"] != "leaderboard"]
    man["artifacts"].append({"name": "leaderboard", "stale": True, "reason": "scoring failed",
                             "generated_at": "2026-10-01T00:00:00+00:00"})
    got = _render(node, {"leaderboard": _lb(), "manifest": man})["panels"]["p-study"]
    assert "state-stale" in got["cls"] and "Stale: scoring failed" in got["body"]
    assert "lb-table" in got["body"], "a stale board keeps its numbers"


def test_a_hostile_value_in_any_field_is_escaped(node):
    lb = _lb()
    lb["contestants"][0]["contestant"] = HOSTILE
    lb["contestants"][1]["scorable"] = HOSTILE
    lb["contestants"][2]["missing"] = HOSTILE
    lb["through_week"] = HOSTILE
    html = _body(node, lb)
    assert "<script>" not in html
    lb["contestants"][3]["contestant"] = HOSTILE + ":blind"
    assert "<script>" not in _body(node, lb)


def test_study_has_its_own_panel_and_the_router_no_longer_holds_its_placeholder():
    page = (SITE / "index.html").read_text()
    study = page.split('id="v-study"')[1].split('id="v-ledger"')[0]
    assert 'id="p-study"' in study and "view-body" not in study
    assert "study:" not in (JS / "router.js").read_text().split("const RENDERERS")[1].split("};")[0]


def test_study_adds_no_motion_and_no_colour_literal():
    src = (JS / "panels" / "study.js").read_text()
    assert not re.search(r"animation|transition|setInterval|requestAnimationFrame", src)
    css = re.findall(r"/\* Study \(#487\).*?(?=/\* The view change)", (SITE / "index.html").read_text(), flags=re.S)
    assert css and not re.search(r"animation|transition|@import|url\(", css[0])
    rules = re.sub(r"/\*.*?\*/", "", css[0], flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgb|hsl", rules), "a colour that is not a token"


def test_the_leaderboard_fixture_is_the_publishers_schema():
    lb = copy.deepcopy(_lb())
    assert set(lb) >= {"interim", "label", "season", "first_scored_week", "through_week", "scored_weeks",
                       "h2_implementation", "contestants", "missing_games", "market_snapshot_ids",
                       "packet_refused"}
    for r in lb["contestants"]:
        assert set(r) == {"contestant", "log_loss_diff", "brier_diff", "abs_gap", "missing", "scorable",
                          "missing_games"}
        assert set(r["log_loss_diff"]) == {"estimate", "lo", "hi", "weeks", "games"}
    src = (ROOT / "src" / "hub" / "forecasters" / "leaderboard.py").read_text()
    for k in ("log_loss_diff", "brier_diff", "abs_gap", "scorable", "missing_games", "interim"):
        assert f'"{k}"' in src
