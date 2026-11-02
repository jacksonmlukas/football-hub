"""The Games view draws one tile per game, with its margin range on a field (#485).

Each property is a function of modules run in node (`games.js`, `slate.js`) on fixtures built here,
against a stub document and a stub `fetch`; nothing is lifted out of a file by pattern:

* a tile's band spans `margin_lo` to `margin_hi` on the strip's scale, its tick sits at the
  expected margin, and its label names the favourite;
* the live slot never shows a dash that could read as 0-0, and an archived week has no slot;
* the poll replaces the score slots and nothing else, and cannot replace a frozen number;
* sorting reorders the same tiles; the model-versus-market comparison is absent while the model
  is the market's line; every feed value is escaped; the grid reflows 3 / 2 / 1.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "site" / "js"
HOSTILE = "<script>alert(1)</script>"
NOW = "2026-10-18T06:00:00Z"


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _mod(rel: str) -> str:
    return json.dumps((JS / rel).as_uri())


def _run(node: str, body: str):
    got = subprocess.run([node, "--input-type=module", "-e", body],
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout.strip())


def _row(game_id="2026_06_ARI_LA", p=0.8367, mean=12.5, lo=-3.83, hi=28.83, **kw):
    return {"game_id": game_id, "home_win_prob": p, "margin_mean": mean, "margin_lo": lo,
            "margin_hi": hi, "model": "market_baseline", "version": "market-e999ae6b-live",
            "price_source": "live", "priced_at": "2026-10-17 16:58:05.2", "fit_through_week": 5,
            "predicted_at": "2026-10-17 16:58:09", "cfg_digest": "839ceda9",
            "kickoff": "2026-10-18T20:05:00+00:00", "market_prob": p, **kw}


PREDS = {"provenance": {"live": {"why": "the capture is dated and immutable."}},
         "season": 2026, "week": 6, "n": 3,
         "rows": [_row(),
                  _row("2026_06_BUF_NE", 0.30, -4.5, -20.0, 11.0, kickoff="2026-10-18T17:00:00+00:00"),
                  _row("2026_06_DEN_KC", 0.51, 0.5, -15.0, 16.0, kickoff="2026-10-17T00:30:00+00:00")]}


def _scale(v: float) -> float:
    """The strip's scale, written out here and not imported: 28..272 across -35..35 points."""
    return 28 + (min(max(v, -35), 35) + 35) / 70 * 244


PRELUDE = (f"import * as G from {_mod('panels/games.js')};\n"
           f"Date.now = () => Date.parse({json.dumps(NOW)});\n")


def _tile(node: str, row: dict, feed: dict | None = None, preds: dict | None = None) -> str:
    feed = feed or {"kind": "none"}
    return _run(node, PRELUDE + f"console.log(JSON.stringify(G.gameTile({json.dumps(row)}, 0, "
                      f"{{...{json.dumps(feed)}, byGame: new Map(Object.entries({json.dumps(feed.get('by', {}))}))}}, "
                      f"{json.dumps(preds or PREDS)})));")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_the_band_spans_the_range_on_the_scale_and_the_label_names_the_favourite(node):
    html = _tile(node, _row())
    band = re.search(r'class="gt-band" x="([\d.]+)" y="\d+" width="([\d.]+)"', html)
    assert band, "no band drawn"
    x, w = float(band.group(1)), float(band.group(2))
    assert x == pytest.approx(_scale(-3.83), abs=0.06), "the band starts at margin_lo"
    assert x + w == pytest.approx(_scale(28.83), abs=0.06), "the band ends at margin_hi"
    tick = re.search(r'class="gt-pylon" x1="([\d.]+)"', html)
    assert tick and float(tick.group(1)) == pytest.approx(_scale(12.5), abs=0.06)
    text = _text(html)
    assert "ARI at LA" in text and "83.7% LA wins" in text
    assert "LA by 12.5" in text, "the pylon tick is labelled with the expected margin"


def test_the_away_favourite_is_named_and_its_probability_is_the_complement(node):
    text = _text(_tile(node, _row("2026_06_BUF_NE", 0.30, -4.5, -20.0, 11.0)))
    assert "70.0% BUF wins" in text and "BUF by 4.5" in text


def test_the_strip_has_a_gridline_every_seven_points_and_labels_at_two_and_four_touchdowns(node):
    html = _tile(node, _row())
    grid = [float(x) for x in re.findall(r'class="gt-(?:grid|tie)" x1="([\d.]+)"', html)]
    assert len(grid) == 11
    steps = [round((b - a) / (244 / 70), 3) for a, b in pairwise(grid)]
    assert steps == [7.0] * 10, "a gridline every 7 points"
    axis = re.findall(r'class="gt-axis"[^>]*>([^<]*)<', html)
    assert axis == ["4 TD", "2 TD", "tie", "2 TD", "4 TD"]
    names = re.findall(r'class="gt-ez-name"[^>]*>([^<]*)<', html)
    assert names == ["ARI", "LA"], "the end zones are labelled with each team"


def test_a_range_past_the_strip_sits_on_its_end(node):
    html = _tile(node, _row(lo=-60.0, hi=80.0, mean=70.0))
    band = re.search(r'class="gt-band" x="([\d.]+)" y="\d+" width="([\d.]+)"', html)
    assert band, "no band drawn"
    assert float(band.group(1)) == pytest.approx(28.0)
    assert float(band.group(1)) + float(band.group(2)) == pytest.approx(272.0, abs=0.06)


def test_the_footer_has_the_frozen_chip_the_digest_and_the_provenance_popover(node):
    html = _tile(node, _row())
    assert 'data-state="frozen"' in html and "locked 13h ago" in _text(html)
    assert '<code class="digest">839ceda9</code>' in html
    pop = html[html.index("popover>"):]
    for want in ("market-e999ae6b-live", "fit through week", "2026-10-17 16:58:05.2",
                 "2026-10-17 16:58:09", "price source", "the capture is dated and immutable."):
        assert want in pop, want
    assert 'popovertarget="gt-pop-0"' in html and 'id="gt-pop-0"' in html


def test_a_missing_provenance_sentence_says_so(node):
    assert "No provenance sentence" in _tile(node, _row(price_source="close"))


# --- the live slot -------------------------------------------------------------------------------

def _slot(node: str, kind: str, by: dict | None = None, row: dict | None = None) -> str:
    by = by or {}
    return _run(node, PRELUDE + f"const feed = {{ kind: {json.dumps(kind)}, age: '3h ago', "
                      f"byGame: new Map(Object.entries({json.dumps(by)})) }};\n"
                      f"console.log(JSON.stringify(G.liveSlot({json.dumps(row or _row())}, feed)));")


def test_an_archived_week_has_no_slot_at_all(node):
    html = _tile(node, _row(), {"kind": "none"})
    assert "data-live-slot" not in html and "scores" not in html and "not started" not in html


def test_an_idle_feed_is_a_stale_chip_naming_its_age_and_never_a_dash(node):
    got = _slot(node, "idle")
    assert 'data-state="stale"' in got and "3h ago" in got
    assert not re.search(r"[—\u2013-]\s*[—\u2013-]|0\s*[\u2013-]\s*0", _text(got))


def test_an_unavailable_feed_is_unconfirmed(node):
    assert 'data-state="unconfirmed"' in _slot(node, "unavailable")


def test_a_game_the_feed_does_not_have_or_has_not_started_says_not_started(node):
    assert "not started" in _slot(node, "ok", {})
    pre = {"ARI_LA": {"away": "ARI", "home": "LA", "state": "pre", "away_score": "0",
                      "home_score": "0", "detail": "Sun 4:05 PM"}}
    got = _slot(node, "ok", pre)
    assert "not started" in got and "0" not in _text(got), "a pre-game 0-0 is not shown"
    nulls = {"ARI_LA": {"away": "ARI", "home": "LA", "state": "in", "away_score": None,
                        "home_score": None}}
    assert "not started" in _slot(node, "ok", nulls)


def test_a_game_in_progress_shows_the_score_in_the_live_state(node):
    inn = {"ARI_LA": {"away": "ARI", "home": "LA", "state": "in", "away_score": "7",
                      "home_score": "3", "detail": "Q2 4:10"}}
    got = _slot(node, "ok", inn)
    assert 'class="state-live gt-livebox"' in got and 'data-state="live"' in got
    assert "ARI 7 \u2013 LA 3" in _text(got) and "Q2 4:10" in _text(got)


def test_espn_spelling_is_joined_to_nflverse(node):
    inn = {"LA_WAS": {"away": "LAR", "home": "WSH", "state": "in", "away_score": "1",
                      "home_score": "2", "detail": "Q1"}}
    got = _slot(node, "ok", inn, _row("2026_06_LA_WAS"))
    assert "LA 1 \u2013 WAS 2" in _text(got)


# --- sorting, and the market note --------------------------------------------------------------------

def _order(node: str, how: str) -> list[str]:
    return _run(node, PRELUDE + f"console.log(JSON.stringify(G.sortRows({json.dumps(PREDS['rows'])}, "
                      f"{json.dumps(how)}).map(r => r.game_id)));")


def test_sort_by_kickoff_and_by_most_lopsided(node):
    assert _order(node, "kickoff") == ["2026_06_DEN_KC", "2026_06_BUF_NE", "2026_06_ARI_LA"]
    assert _order(node, "lopsided") == ["2026_06_ARI_LA", "2026_06_BUF_NE", "2026_06_DEN_KC"]


def test_a_row_with_no_kickoff_sorts_last_and_ties_keep_the_published_order(node):
    rows = [_row("2026_06_A_B", kickoff=None), _row("2026_06_C_D"), _row("2026_06_E_F")]
    got = _run(node, PRELUDE + f"console.log(JSON.stringify(G.sortRows({json.dumps(rows)}, 'kickoff')"
                      ".map(r => r.game_id)));")
    assert got == ["2026_06_C_D", "2026_06_E_F", "2026_06_A_B"]


def _body(node: str, rows: list[dict]) -> str:
    return _run(node, PRELUDE + f"console.log(JSON.stringify(G.gamesBody({json.dumps(rows)}, "
                      "{ kind: 'none', byGame: new Map() }, {}, 'kickoff')));")


def test_no_model_versus_market_column_while_the_model_is_the_markets_line(node):
    body = _body(node, PREDS["rows"])
    assert "until issue 365" in body and "model-versus-market" in body
    assert "gt-vs" not in body
    other = [_row(model="elo", home_win_prob=0.7, market_prob=0.65)]
    got = _body(node, other)
    assert "until issue 365" not in got and "model 70.0% · market 65.0%" in _text(got)


def test_one_tile_per_prediction_row_with_no_cap(node):
    rows = [_row(f"2026_06_T{n:02d}_U{n:02d}") for n in range(20)]
    assert _body(node, rows).count('class="game-tile"') == 20


# --- escaping ----------------------------------------------------------------------------------------

def test_every_feed_value_in_a_tile_is_escaped(node):
    row = _row(f"2026_06_{HOSTILE}_LA", version=HOSTILE, cfg_digest=HOSTILE, priced_at=HOSTILE,
               price_source=HOSTILE)
    html = _tile(node, row, preds={"provenance": {HOSTILE: {"why": HOSTILE}}})
    assert "<script>" not in html
    bad = _tile(node, _row("garbage<b>"), preds=PREDS)
    assert "<b>" not in bad, "an id that names no sides is escaped whole"
    inn = {"ARI_LA": {"away": "ARI", "home": "LA", "state": "in", "away_score": HOSTILE,
                      "home_score": "1", "detail": HOSTILE}}
    assert "<script>" not in _slot(node, "ok", inn)


# --- the poll, the archive and the sort, through the slate -------------------------------------------

MAN = {"season": 2026, "week": 6, "generated_at": "2026-10-17T16:58:00+00:00",
       "weeks": [{"name": "preds_2026_wk06", "season": 2026, "week": 6},
                 {"name": "preds_2026_wk05", "season": 2026, "week": 5}],
       "artifacts": [{"name": "preds_2026_wk06", "stale": False, "generated_at": "2026-10-17T16:58:13+00:00"},
                     {"name": "live", "stale": False, "generated_at": "2026-10-18T05:59:00+00:00"}]}


def _slate_script(steps: str) -> str:
    """A stub document and fetch, `renderSlate` run once, then `steps`. Fetches are logged;
    the second fetch of a predictions artifact would return different numbers if it were made."""
    return (f"import * as S from {_mod('panels/slate.js')};\n"
            f"Date.now = () => Date.parse({json.dumps(NOW)});\n"
            "const els = {}; const log = [];\n"
            "const el = () => ({ innerHTML: '', textContent: '', hidden: false, className: '', _c: {},"
            " querySelector(s) { return this._c[s] ??= el(); } });\n"
            "let slots = [];\n"
            "globalThis.document = { getElementById: id => els[id] ??= el(),"
            " querySelectorAll: () => slots };\n"
            f"const MAN = {json.dumps(MAN)}; const PREDS = {json.dumps(PREDS)};\n"
            "let livePayload = { generated_at: '2026-10-18T05:59:00+00:00', rows: [] };\n"
            "globalThis.fetch = async url => {\n"
            "  const n = url.replace(/^data\\//, '').replace(/\\.json.*$/, ''); log.push(n);\n"
            "  const seen = log.filter(x => x === n).length;\n"
            "  if (n.startsWith('preds_')) return { ok: true, json: async () => seen > 1 ? "
            "{ ...PREDS, rows: PREDS.rows.map(r => ({ ...r, margin_mean: 99 })) } : PREDS };\n"
            "  if (n === 'live') return { ok: true, json: async () => livePayload };\n"
            "  return { ok: false };\n};\n"
            "const body = () => els['p-slate']._c['.body'].innerHTML;\n"
            "const slotsOf = html => [...html.matchAll(/data-live-slot data-game=\"([^\"]*)\"/g)]"
            ".map(m => { const o = { key: m[1], innerHTML: '', getAttribute: k => "
            "k === 'data-game' ? m[1] : null }; return o; });\n"
            "await S.renderSlate(MAN);\n" + steps)


def test_the_poll_replaces_the_score_slots_and_cannot_replace_a_frozen_number(node):
    """The existing frozen-cache test, ported to tiles: the predictions are fetched once for the
    page, a poll fetches only the live feed, rewrites only the slots, and the body (every frozen
    number, the order, the popovers) is byte-identical afterwards."""
    got = _run(node, _slate_script(
        "const before = body(); slots = slotsOf(before);\n"
        "livePayload = { generated_at: '2026-10-18T06:00:00+00:00', rows: [{ away: 'ARI', home: 'LA', "
        "state: 'in', away_score: '7', home_score: '3', detail: 'Q2' }] };\n"
        "await S.refreshLive(MAN); await S.refreshLive(MAN);\n"
        "console.log(JSON.stringify({ same: body() === before, n: slots.length, "
        "filled: slots.filter(s => s.innerHTML.includes('ARI 7')).map(s => s.key), "
        "idle: slots.filter(s => s.innerHTML.includes('not started')).length, "
        "preds: log.filter(x => x.startsWith('preds_')).length, "
        "live: log.filter(x => x === 'live').length, "
        "frozenMargin: before.includes('99.0')}));"))
    assert got["same"] is True, "a poll rewrote something other than a score slot"
    assert got["n"] == 3 and got["filled"] == ["ARI_LA"] and got["idle"] == 2
    assert got["preds"] == 1, "the predictions are fetched once per page"
    assert got["live"] == 3 and got["frozenMargin"] is False


def test_an_archived_week_draws_no_slot_and_the_poll_leaves_it_alone(node):
    got = _run(node, _slate_script(
        "await S.chooseWeek('preds_2026_wk05');\n"
        "const after = body(); const mark = log.length; slots = slotsOf(after);\n"
        "await S.refreshLive(MAN);\n"
        "console.log(JSON.stringify({ slots: slots.length, hasSlot: after.includes('data-live-slot'), "
        "polled: log.slice(mark), archived: S.isArchived(), "
        "live: log.slice(0, log.length - 0).filter(x => x === 'live').length }));"))
    assert got["slots"] == 0 and got["hasSlot"] is False
    assert got["polled"] == [], "an archived week is not polled"
    assert got["archived"] is True and got["live"] == 1, "live was fetched for the live week only"


def test_choosing_back_to_the_manifests_week_restores_the_slots(node):
    got = _run(node, _slate_script(
        "await S.chooseWeek('preds_2026_wk05'); await S.chooseWeek('preds_2026_wk06');\n"
        "console.log(JSON.stringify({ slots: slotsOf(body()).length, archived: S.isArchived() }));"))
    assert got == {"slots": 3, "archived": False}


def test_choosing_a_sort_reorders_the_tiles_and_keeps_the_numbers(node):
    got = _run(node, _slate_script(
        "const order = h => [...h.matchAll(/<article class=\"game-tile\" data-game=\"([A-Z]+_[A-Z]+)\"/g)].map(m => m[1]);\n"
        "const first = order(body());\n"
        "await S.chooseSort('lopsided'); const second = order(body());\n"
        "console.log(JSON.stringify({ first, second, pressed: body().match(/data-sort=\"lopsided\" "
        "aria-pressed=\"(\\w+)\"/)[1], preds: log.filter(x => x.startsWith('preds_')).length }));"))
    assert got["first"] == ["DEN_KC", "BUF_NE", "ARI_LA"], "kickoff order is the default"
    assert got["first"] != got["second"] and got["second"][0] == "ARI_LA"
    assert got["pressed"] == "true" and got["preds"] == 1


# --- the page's shell --------------------------------------------------------------------------------

STYLE = (ROOT / "site" / "index.html").read_text()


def test_the_grid_reflows_from_three_columns_to_one():
    assert re.search(r"\.game-grid\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*;", STYLE)
    two = re.search(r"@media \(min-width: (\d+)px\)\s*\{\s*\.game-grid\s*\{[^}]*repeat\(2,", STYLE)
    three = re.search(r"@media \(min-width: (\d+)px\)\s*\{\s*\.game-grid\s*\{[^}]*repeat\(3,", STYLE)
    assert two and three, "the grid no longer steps through two and three columns"
    assert int(two.group(1)) < int(three.group(1)) <= 1280, "three columns by 1280px"
    assert int(two.group(1)) > 390, "one column at 390px"


def test_the_tiles_style_names_no_colour_literal():
    css = STYLE[STYLE.index("/* Games (#485)"):STYLE.index("/* The view change is a same-document")]
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)     # the comment cites the issue as #485
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", css)
    assert "animation" not in css and "transition" not in css, "nothing here moves"


def test_no_third_party_request_in_the_games_modules():
    text = (JS / "panels" / "games.js").read_text()
    assert not re.search(r"https?://|import\s*\(|fetch\(", text)
