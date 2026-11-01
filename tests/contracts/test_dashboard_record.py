"""The Record view leads with one calibration sentence, then the curve with its intervals (#486).

Each property is a function of modules run in node (`panels/record.js`, `charts.js`) on the week-6
fixture's own `track_record.json` or on copies of it edited here, plus the page's real `render()`
against a stub DOM (`dashboard_render.mjs`):

* the sentence's bin, rate and count are read from the newest season's bins, nothing is typed;
* each season keeps its own curve, and nothing pools 2025 with 2026;
* a bin with no games is a hollow ring on the baseline; a bin with `lo`/`hi` has a bar from `lo` to
  `hi`, and an artifact without them draws the curve without bars;
* the season table carries the coin-flip and home-constant rows from `baselines` and none without;
* the counts are frozen or stale chips, and an unknown count is one unconfirmed chip;
* the player ranges are bullet rows, the out-of-sample ones marked "no verdict yet";
* nothing off the feed reaches the markup unescaped.
"""
# ruff: noqa: RUF001  (the sentence's en dash is the thing under test)
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from html import unescape
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
JS = SITE / "js"
FIXTURE = ROOT / "tests" / "fixtures" / "site" / "data" / "track_record.json"
RENDER = Path(__file__).resolve().parent / "dashboard_render.mjs"
HOSTILE = "<script>alert(1)</script>"
S, PAD = 300, 40          # charts.js: the viewBox and its padding; the baseline sits at S - PAD


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


def _tr(**kw) -> dict:
    """The week-6 fixture's record, made fresh so its chips are frozen unless a test says not."""
    tr = json.loads(FIXTURE.read_text())
    tr["generated_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    tr.update(kw)
    return tr


def _body(node: str, tr: dict, art: dict | None = None) -> str:
    return _run(node, f"import {{ trackRecordBody }} from {_mod('panels/record.js')};\n"
                      f"console.log(trackRecordBody({json.dumps(tr)}, {json.dumps(art)}));")


def _sentence(node: str, season: dict) -> str | None:
    out = _run(node, f"import {{ calibrationSentence }} from {_mod('panels/record.js')};\n"
                     f"console.log(JSON.stringify(calibrationSentence({json.dumps(season)})));")
    return json.loads(out)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def _bin(label: str, n: int, actual: float | None, lo=None, hi=None) -> dict:
    mid = sum(float(x) for x in label.split("-")) / 2
    return {"bin": label, "n": n, "predicted": mid if n else None, "actual": actual if n else None,
            "gap": None, "lo": lo, "hi": hi}


def _render(node: str, ov: dict) -> dict:
    got = subprocess.run([node, str(RENDER)], input=json.dumps({"site": str(SITE), "ov": ov}),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


# --- the sentence --------------------------------------------------------------

def test_the_sentence_is_the_most_populated_bin_above_fifty_percent_on_the_week_six_fixture(node):
    season = json.loads(FIXTURE.read_text())["seasons"][0]
    assert season["season"] == 2026
    above = [b for b in season["bins"] if b["n"] and float(b["bin"].split("-")[0]) >= 0.5]
    top = max(above, key=lambda b: b["n"])
    want = (f"When we said {round(float(top['bin'].split('-')[0]) * 100)}–"
            f"{round(float(top['bin'].split('-')[1]) * 100)}%, the home team won "
            f"{round(top['actual'] * 100)}% of the time ({top['n']} games).")
    assert _sentence(node, season) == want
    # The same, as the numbers a reader will see: pinned so a change to the choice is deliberate.
    assert want == "When we said 50–60%, the home team won 56% of the time (18 games)."


def test_the_sentence_follows_the_artifact_not_a_constant(node):
    season = {"bins": [_bin("0.5-0.6", 4, 0.5), _bin("0.6-0.7", 9, 2 / 3), _bin("0.7-0.8", 5, 0.8)]}
    assert _sentence(node, season) == \
        "When we said 60–70%, the home team won 67% of the time (9 games)."
    season["bins"][2]["n"] = 12
    assert _sentence(node, season) == \
        "When we said 70–80%, the home team won 80% of the time (12 games)."


def test_a_bin_below_fifty_percent_is_never_the_sentence_and_a_tie_takes_the_lower_bin(node):
    season = {"bins": [_bin("0.3-0.4", 40, 0.25), _bin("0.5-0.6", 7, 0.5), _bin("0.6-0.7", 7, 0.7)]}
    assert _sentence(node, season) == \
        "When we said 50–60%, the home team won 50% of the time (7 games)."
    assert _sentence(node, {"bins": [_bin("0.3-0.4", 40, 0.25)]}) is None
    assert _sentence(node, {"bins": [_bin("0.5-0.6", 1, 1.0)]}) == \
        "When we said 50–60%, the home team won 100% of the time (1 game)."


def test_a_season_with_no_bin_above_fifty_percent_says_so_instead_of_inventing_one(node):
    tr = _tr()
    tr["seasons"][0]["bins"] = [_bin("0.3-0.4", 3, 0.3)]
    out = _text(_body(node, tr))
    assert "When we said" not in out
    assert "No bin above 50% holds a game yet" in out


def test_the_sentence_leads_and_the_log_loss_sits_beside_the_coin_flips(node):
    html = _body(node, _tr())
    out = _text(html)
    assert out.index("pre-registered") < out.index("When we said") < out.index("log loss is")
    assert html.index("When we said") < html.index('class="bin-filled"'), "the sentence comes before the curve"
    tr = json.loads(FIXTURE.read_text())
    s = tr["seasons"][0]
    assert (f"2026's log loss is {s['log_loss']:.3f}; a coin flip scores "
            f"{s['baselines']['coin_flip']['log_loss']:.3f} (lower is better).") in out


def test_without_baselines_the_log_loss_stands_alone(node):
    tr = _tr()
    del tr["seasons"][0]["baselines"]
    out = _text(_body(node, tr))
    assert "2026's log loss is" in out and "coin flip scores" not in out


# --- the curve -----------------------------------------------------------------

def test_each_season_keeps_its_own_curve_and_nothing_pools(node):
    html = _body(node, _tr())
    newest, older = html.split("<details", 1)
    assert newest.count('class="bin-filled"') == 7 and older.count('class="bin-filled"') == 6
    assert "n=18," in newest and "n=18," not in older, "2026's bin leaked into 2025's curve"
    assert "n=6," in older and "n=6," not in newest
    assert "n=20," not in html and "n=24," not in html, "a bin pooled the seasons' games"
    assert re.search(r"Calibration for 2026\.", newest) and re.search(r"Calibration for 2025\.", older)


def test_a_bin_with_no_games_is_a_hollow_ring_on_the_baseline(node):
    html = _body(node, _tr())
    newest = html.split("<details", 1)[0]
    rings = re.findall(r'<circle class="bin-empty" cx="([\d.]+)" cy="([\d.]+)"', newest)
    assert [cy for _, cy in rings] == [f"{S - PAD:.1f}"] * 3, "three empty 2026 bins, all on the baseline"
    # In their own slots: 0.0-0.1, 0.1-0.2 and 0.9-1.0 on a 220-wide axis starting at 40.
    assert [cx for cx, _ in rings] == ["51.0", "73.0", "249.0"]
    assert 'fill="none"' in newest.split('class="bin-empty"', 1)[1].split("</circle>", 1)[0]


def test_each_dot_has_a_bar_from_lo_to_hi_and_its_count_beside_it(node):
    tr = _tr()
    html = _body(node, tr)
    newest = html.split("<details", 1)[0]
    bins = [b for b in tr["seasons"][0]["bins"] if b["n"]]
    bars = re.findall(r'<line class="bin-bar" x1="([\d.]+)" x2="([\d.]+)" y1="([\d.]+)" y2="([\d.]+)"',
                      newest)
    assert len(bars) == len(bins)
    y = lambda v: f"{S - PAD - v * (S - 2 * PAD):.1f}"      # noqa: E731 (charts.js's scale)
    x = lambda v: f"{PAD + v * (S - 2 * PAD):.1f}"          # noqa: E731
    for (x1, x2, y1, y2), b in zip(bars, bins, strict=True):
        assert x1 == x2 == x(b["predicted"])
        assert (y1, y2) == (y(b["hi"]), y(b["lo"])), f"bin {b['bin']}: the bar is not lo to hi"
    counts = re.findall(r'<text class="bin-n"[^>]*>(\d+)</text>', newest)
    assert counts == [str(b["n"]) for b in bins]


def test_the_axes_are_titled_in_words(node):
    html = _body(node, _tr())
    assert "we said the home team wins…" in html and "…and it did" in html
    assert ">predicted<" not in html and ">actual<" not in html


def test_an_artifact_without_lo_and_hi_draws_the_curve_without_bars(node):
    tr = _tr()
    for s in tr["seasons"]:
        for b in s["bins"]:
            b.pop("lo", None)
            b.pop("hi", None)
    html = _body(node, tr)
    assert "bin-bar" not in html and ", range " not in html
    assert html.count('class="bin-filled"') == 13 and 'class="bin-n"' in html


def test_one_bin_missing_its_interval_is_drawn_without_a_bar_and_the_rest_keep_theirs(node):
    tr = _tr()
    tr["seasons"][0]["bins"][5]["lo"] = None
    newest = _body(node, tr).split("<details", 1)[0]
    assert newest.count('class="bin-bar"') == 6 and newest.count('class="bin-filled"') == 7


# --- the table -----------------------------------------------------------------

def test_the_season_table_has_the_baseline_rows_and_an_arrow_in_each_metric_header(node):
    tr = _tr()
    html = _body(node, tr)
    table = html[html.index("<table"):html.index("</table>")]
    assert "<th>log loss ↓</th>" in table and "<th>Brier ↓</th>" in table
    rows = [re.findall(r"<td[^>]*>(.*?)</td>", r) for r in re.findall(r"<tr>(.*?)</tr>", table)[1:]]
    labels = [r[0] for r in rows]
    s0, s1 = tr["seasons"]
    assert labels[0] == "2026" and labels[1] == "2026 coin flip" and labels[2].startswith("2026 home team always (")
    assert labels[3:] == ["2025", "2025 coin flip", labels[5]] and labels[5].startswith("2025 home team always")
    assert rows[1][2:] == [f"{s0['baselines']['coin_flip']['log_loss']:.4f}",
                           f"{s0['baselines']['coin_flip']['brier']:.4f}"]
    h = s0["baselines"]["home_constant"]
    assert rows[2][1] == str(h["n"]) and rows[2][2] == f"{h['log_loss']:.4f}"
    assert labels[2] == f"2026 home team always ({100 * h['p']:.1f}%)"
    assert rows[3][1] == str(s1["n_scored"])


def test_an_artifact_without_baselines_draws_the_table_without_baseline_rows(node):
    tr = _tr()
    for s in tr["seasons"]:
        del s["baselines"]
    html = _body(node, tr)
    assert "coin flip" not in html and "home team always" not in html
    table = html[html.index("<table"):html.index("</table>")]
    assert table.count("<tr>") == 3, "a header and the two seasons"
    tr = _tr()
    tr["seasons"][0]["baselines"]["home_constant"] = None
    html = _body(node, tr)
    assert "2026 coin flip" in html and "2026 home team always" not in html


# --- the chips -----------------------------------------------------------------

def test_the_counts_are_frozen_chips_while_the_record_is_current(node):
    tr = _tr(n_preregistered=107, n_scored=123)
    chips = _body(node, tr).split("</p>", 1)[0]
    assert chips.count('data-state="frozen"') == 2 and 'data-state="stale"' not in chips
    assert "107 pre-registered" in chips and "16 backtest" in chips


def test_the_counts_go_stale_with_their_age_when_the_record_is_old_or_marked_stale(node):
    old = _tr(generated_at=(datetime.now(UTC) - timedelta(days=12)).isoformat())
    chips = _body(node, old).split("</p>", 1)[0]
    assert chips.count('data-state="stale"') == 2 and "12d ago" in chips
    assert 'data-state="frozen"' not in chips
    marked = _body(node, _tr(), {"name": "track_record", "stale": True, "reason": "scoring failed",
                                 "generated_at": (datetime.now(UTC) - timedelta(days=3)).isoformat()})
    chips = marked.split("</p>", 1)[0]
    assert chips.count('data-state="stale"') == 2 and "3d ago" in chips


def test_an_unknown_count_is_one_unconfirmed_chip_not_a_zero(node):
    chips = _body(node, _tr(n_preregistered=None)).split("</p>", 1)[0]
    assert 'data-state="unconfirmed"' in chips and "count unknown" in chips
    assert "0 pre-registered" not in chips and "backtest" not in chips


# --- the player ranges and the survivor price ----------------------------------

def test_the_player_ranges_are_bullet_rows_for_the_backtest_and_each_position(node):
    tr = _tr()
    html = _body(node, tr)
    rows = re.findall(r'<li class="rb-row">(.*?)</li>', html, flags=re.S)
    ic = tr["interval_coverage"]
    names = [re.findall(r'rb-name">([^<]*)<', r)[0] for r in rows]
    assert names == ["Backtest, all positions", "2026 QB", "2026 RB", "2026 WR", "2026 TE"]
    for r in rows:
        assert all(c in r for c in ('class="rb-band"', 'class="rb-measure"', 'class="rb-claim"'))
    text = _text(rows[0])
    assert f"measured {ic['gate_cov80'] * 100:.1f}% against 80.0% ±2.0 points" in text
    assert "audit look 1 of 3: COVERS" in text and 'data-state="frozen"' in rows[0]
    qb = ic["current_season"]["by_position"][0]
    assert f"measured {qb['cov80'] * 100:.1f}%" in _text(rows[1])


def test_out_of_sample_rows_say_no_verdict_yet_in_the_unconfirmed_state(node):
    rows = re.findall(r'<li class="rb-row">(.*?)</li>', _body(node, _tr()), flags=re.S)
    for r in rows[1:]:
        assert 'data-state="unconfirmed"' in r and "no verdict yet" in _text(r), r
    assert 'data-state="unconfirmed"' not in rows[0]
    tr = _tr()
    del tr["interval_coverage"]["audit"]
    first = re.findall(r'<li class="rb-row">(.*?)</li>', _body(node, tr), flags=re.S)[0]
    assert 'data-state="unconfirmed"' in first and "no audit yet" in first


def test_a_record_without_coverage_has_no_range_rows(node):
    html = _body(node, _tr(interval_coverage=None))
    assert "rb-row" not in html and "Player 80% ranges" not in html and "<svg" in html


def test_the_survivor_verdict_is_a_chip(node):
    tr = _tr()
    html = _body(node, tr)
    verdict = tr["interval_coverage"]["survivor"]["verdict"]
    assert re.search(r'Verdict: <span class="chip chip-frozen" data-state="frozen">.*?'
                     rf'<span class="chip-label">{verdict}</span>', html, flags=re.S)
    assert f"— {verdict}." not in html, "the verdict is no longer a bare word in the sentence"


# --- escaping, motion, third parties -------------------------------------------

def test_a_hostile_value_in_any_new_field_is_escaped(node):
    tr = _tr()
    ic = tr["interval_coverage"]
    ic["current_season"]["by_position"][0]["position"] = HOSTILE
    ic["current_season"]["by_position"][1]["verdict"] = HOSTILE
    ic["audit"]["marginal"] = HOSTILE
    ic["survivor"]["verdict"] = HOSTILE
    s = tr["seasons"][0]
    s["season"] = HOSTILE
    s["baselines"]["home_constant"]["p"] = HOSTILE
    tr["seasons"][1]["season"] = HOSTILE
    tr["note"] = HOSTILE
    tr["seasons"][0]["bins"][5]["bin"] = HOSTILE
    html = _body(node, tr)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_the_record_adds_no_motion_and_no_third_party_request():
    for rel in ("panels/record.js", "charts.js"):
        src = (JS / rel).read_text()
        assert not re.search(r"animation|transition|setInterval|requestAnimationFrame|fetch\(", src), rel
        assert not re.search(r"https?://", src), f"{rel} names a host"
    css = re.findall(r"/\* Record \(#486\).*?(?=/\* The view change)", (SITE / "index.html").read_text(),
                     flags=re.S)
    assert css and not re.search(r"animation|transition|@import|url\(", css[0])
    rules = re.sub(r"/\*.*?\*/", "", css[0], flags=re.S)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgb|hsl", rules), "a colour that is not a token"


# --- the page ------------------------------------------------------------------

def test_the_rendered_record_panel_carries_the_sentence_and_the_statement_sits_with_it(node):
    got = _render(node, {"track_record": _tr()})["panels"]
    body = got["p-record"]["body"]
    assert "When we said 50–60%, the home team won 56% of the time (18 games)." in _text(body)
    assert "rec-chips" in body and "bin-bar" in body
    assert "pre-registered" in got["st-backtest"]["body"], "the page-level statement is still filled"
    page = (SITE / "index.html").read_text()
    section = page[page.index('<section id="p-record"'):page.index("</section>", page.index('id="p-record"'))]
    assert 'id="st-backtest"' in section, "the statement moved under the record's heading"


def test_the_rendered_record_goes_stale_through_the_manifest_entry(node):
    man = json.loads((SITE / "data" / "manifest.json").read_text())
    for a in man["artifacts"]:
        if a["name"] == "track_record":
            a.update(stale=True, reason="scoring failed")
    body = _render(node, {"track_record": _tr(), "manifest": man})["panels"]["p-record"]["body"]
    assert body.split('class="rec-chips">', 1)[1].split("</p>", 1)[0].count('data-state="stale"') == 2


def test_the_committed_artifact_renders_a_sentence(node):
    """The published `track_record.json`, as the page would fetch it, produces a sentence whose
    figures are its own (whatever week it is): the property, not a pinned number."""
    committed = json.loads((SITE / "data" / "track_record.json").read_text())
    body = _body(node, dict(committed, generated_at=datetime.now(UTC).isoformat()))
    seasons = committed["seasons"]
    above = [b for b in seasons[0]["bins"] if b["n"] and float(b["bin"].split("-")[0]) >= 0.5]
    if above:
        top = max(above, key=lambda b: b["n"])
        assert f"({top['n']} games)." in _text(body)
