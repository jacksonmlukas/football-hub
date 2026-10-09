"""The page escapes what it renders.

`site/index.html` builds its tables by concatenating field values into HTML strings and
assigning them to `innerHTML`. Player names, team names and status text arrive from ESPN and
nflverse and went in unescaped: an apostrophe or an ampersand in a name is enough to produce
malformed markup, and anything angle-bracketed in a third-party feed is injected into the
page. The site is public, unattended, and refreshed twice a week by a workflow with a push
token -- nobody is watching it render.

The exposure is small today, which is exactly why it was worth fixing while it was cheap.

Two tests, and the second is the one that lasts. Running `esc` and `table` for real proves
they escape; the static guard proves nobody added a thirteenth interpolation and forgot.
`table` escaping by default is what makes the guard's job small: a column cannot forget.

The sections below are here for the same reason and by the same technique: lift the real
function out of the page, run it in node, and assert on what it produces. They cover the
two things the page can only get wrong silently -- which artifact it asks for (a name it
rebuilt rather than read, disagreeing with the manifest and reporting nothing) and which
season a published week and a calibration curve belong to.

The last section turns the same excision habit `tests/contracts/test_guards_are_load_bearing.py`
applies to `src/` on the page itself: each protection is declared where it lives, deleted, and
required to turn these tests red (#61).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from guardlib import (
    CHILD_FLAGS,
    Guard,
    declared,
    excise,
    marker,
    outcome,
    selector_args,
    selector_file,
)

ROOT = Path(__file__).resolve().parents[2]

# Every test below reads the page through this one name, so the excision harness at the foot
# of the file can hand a child run a mutated copy and have all of them apply to it. An
# environment variable is the thing that went wrong in the Python harness -- pytest's own
# `pythonpath` setting shadowed it and every mutant silently ran against the real source --
# so nothing here assumes the override took: the pair of tests under "the harness is
# looking at the mutant" prove it on every run.
PAGE = Path(os.environ.get("HUB_DASHBOARD_PAGE") or ROOT / "site" / "index.html")

# Names carrying every character that matters, in the shapes a feed actually produces.
HOSTILE = "<script>alert(1)</script>"
REAL = "Ja'Marr Chase & D/ST <TEN>"


def _lift(pattern: str) -> str:
    """Pull one function out of the page so node can run the real thing.

    Lifted rather than reimplemented. A copy of the escaper in this file would pass while
    the page's own copy was wrong, which is the failure mode of testing a transcription.
    """
    got = re.search(pattern, PAGE.read_text())
    assert got, f"{pattern!r} no longer matches the page; the test is looking at nothing"
    return got.group(0)


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _run(node: str, body: str) -> str:
    got = subprocess.run([node, "-e", body], capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def test_a_name_carrying_markup_renders_as_text(node):
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    out = _run(node, f"{esc}\nconsole.log(esc({json.dumps(HOSTILE)}));")
    assert "<script>" not in out
    assert out == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_an_ordinary_name_survives_intact(node):
    """A gate that mangles `Ja'Marr Chase` is one that gets turned off. The entities have to
    be entities, and they have to be the right ones."""
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    out = _run(node, f"{esc}\nconsole.log(esc({json.dumps(REAL)}));")
    assert out == "Ja&#39;Marr Chase &amp; D/ST &lt;TEN&gt;"
    assert "Chase" in out and "D/ST" in out


def test_the_table_escapes_every_cell_without_being_asked(node):
    """The structural half. Escaping at each call site is discipline; escaping in `table` is
    a property, and it is why a new column cannot reintroduce this."""
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    tbl = _lift(r"function table\(cols, rows\) \{[\s\S]*?\n\}")
    body = (f"{esc}\n{tbl}\n"
            f"console.log(table([{{label: 'player', get: r => r.player}}], "
            f"[{{player: {json.dumps(HOSTILE)}}}]));")
    out = _run(node, body)
    assert "<script>" not in out, "a raw script tag reached the page's markup"
    assert "&lt;script&gt;" in out


def test_a_missing_value_renders_as_nothing_not_as_undefined(node):
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    assert _run(node, f"{esc}\nconsole.log('[' + esc(null) + esc(undefined) + ']');") == "[]"


# --- nobody added a thirteenth interpolation and forgot ----------------------

# Values that arrive from a third party and end up inside markup: fields read off a row,
# and the locals the prose around the tables builds from them. `r.live.*` is ESPN's
# scoreboard; the rest are names off nflverse and the ESPN league.
FROM_A_FEED = ("player", "pos", "team", "teams", "injury_status", "nfl_team",
               "detail", "state", "away_score", "home_score", "withheld",
               "held", "spent", "start", "sit")


def test_every_feed_value_in_the_markup_goes_through_esc():
    """Read off the page rather than remembered. Each `${...}` that names a third-party
    value has to escape it, and the ones that legitimately build markup -- `html`, a nested
    template, a call to `table` -- name no such value and are not caught here.

    Verified against the page as it stood before the fix: this reports the three hand-built
    slate cells and the roster and survivor prose. The `table` getters are not `${...}` at
    all, which is why `table` escaping on its own carries them."""
    text = PAGE.read_text()
    raw = []
    for expr in re.findall(r"\$\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text):
        names = re.findall(r"[\w.]+", expr)
        names = [n for part in names for n in part.split(".")]
        if not any(n in FROM_A_FEED for n in names):
            continue
        if "esc(" in expr or "table(" in expr or "fmt(" in expr or "pct(" in expr:
            continue
        raw.append(expr.strip())
    assert not raw, (
        f"these interpolate a feed value straight into markup: {raw}. Wrap each in `esc`, "
        f"or move the column into `table`, which escapes on its own.")


def test_the_scan_would_notice_an_unescaped_value():
    """The guard's own premise. A regex that matched no interpolations at all would make the
    assertion above vacuously true, which is how a guard like this dies quietly."""
    text = PAGE.read_text()
    exprs = re.findall(r"\$\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    assert len(exprs) > 20, f"only {len(exprs)} interpolations found; the page uses many more"
    assert any("esc(" in e for e in exprs), "no escaped interpolation found at all"


def test_the_getters_return_text_rather_than_markup():
    """`table` escapes unconditionally, so a getter returning `&mdash;` would render the
    entity literally. The em dash is the character now, and nothing may go back."""
    text = PAGE.read_text()
    getters = re.findall(r"get: r => ([^\n]*)", text)
    entities = [g for g in getters if "&" in g and ";" in g]
    assert not entities, f"these getters return HTML entities: {entities}"


# --- the weekly artifact is named by its writer, not spelled again here ------

DATA = PAGE.parent / "data"

# `publish.artifacts` puts exactly one `preds_`-prefixed entry in a manifest, so there is
# never a set to choose from -- which is why `predsArt` takes no week to disambiguate with.
NEW_NAMING = {"season": 2026, "week": 1,
              "artifacts": [{"name": "track_record", "stale": False},
                            {"name": "preds_2026_wk01", "stale": True, "reason": "no odds"}]}
# The same manifest as it was written before `preds_name` grew a season. The page must ask
# for this one too: the file beside it is called whatever this entry says it is called.
OLD_NAMING = {"season": 2026, "week": 1,
              "artifacts": [{"name": "preds_wk01", "stale": False}]}


def _preds_art(node: str, man):
    """`predsArt` for real, on a manifest, returning the entry it settles on."""
    fn = _lift(r"function predsArt\(man\) \{[\s\S]*?\n\}")
    return json.loads(_run(node, f"{fn}\nconsole.log(JSON.stringify("
                                 f"predsArt({json.dumps(man)}) ?? null));"))


def test_the_slate_asks_for_the_name_the_writer_wrote(node):
    """`publish.preds_name` owns the spelling and `store.week_key` owns the zero-pad inside
    it. A page that rebuilds the name keeps a second copy of both in a language that can
    import neither, so the test is that the *manifest's* spelling wins over anything this
    page could reconstruct: a week-1 2026 manifest spelled the old way is asked for the old
    way, and no arithmetic here gets a vote."""
    assert _preds_art(node, OLD_NAMING)["name"] == "preds_wk01"
    assert _preds_art(node, NEW_NAMING)["name"] == "preds_2026_wk01"


def test_the_stale_reason_comes_back_with_the_name(node):
    """Why the name is read rather than reconstructed. `findArt` matched nothing while the
    manifest said `preds_wk01` and the committed file said `preds_2026_wk01`, so the slate
    silently stopped reporting stale and its reason. Name and reason are one lookup now."""
    got = _preds_art(node, NEW_NAMING)
    assert got["stale"] is True and got["reason"] == "no odds"


def test_a_manifest_naming_no_slate_names_nothing_rather_than_guessing(node):
    """The page does not invent a name it could not read.

    An earlier fix here rebuilt `preds_${man?.season ?? ""}_wk01` as a fallback, which is
    the copy this whole change exists to delete -- and unreachable besides, since
    `publish.artifacts` emits a `preds_` entry unconditionally. A guessed name also fetches
    nothing and reports the 404 as a week that was never published, which is a lie about
    the record. No entry means no slate, and the panel says exactly that."""
    assert _preds_art(node, {"season": 2026, "week": 1, "artifacts": []}) is None
    assert _preds_art(node, {"artifacts": []}) is None
    assert _preds_art(node, None) is None


def test_the_page_never_spells_an_artifact_name_itself():
    """The guard that keeps finding 1 fixed. Any interpolation building a `preds_...` name
    is a second copy of `preds_name` and `store.week_key`, wherever it hides."""
    built = re.findall(r"`preds_[^`]*\$\{[^`]*`", PAGE.read_text())
    assert not built, (f"these rebuild the artifact name the writer owns: {built}. Read it "
                       f"off `manifest.artifacts` instead.")


# --- the draft board retires itself when a newer slate exists (#348) ---------

BOARD_AT = "2026-09-04T17:22:50+00:00"
JULY = {"season": 2027, "week": 1, "artifacts": [
    {"name": "draft_board", "generated_at": "2027-07-10T12:00:00+00:00"}]}
WEEK_2 = {"season": 2026, "week": 2, "artifacts": [
    {"name": "preds_2026_wk02", "generated_at": "2026-09-16T17:00:00+00:00"},
    {"name": "draft_board", "generated_at": BOARD_AT}]}


def _board_shows(node: str, man) -> bool:
    """`boardShows` for real -- with `findArt`, which it reads the board's entry through."""
    fns = "\n".join([_lift(r"function findArt\(man, name\) \{[\s\S]*?\n\}"),
                     _lift(r"function boardShows\(man\) \{[\s\S]*?\n\}")])
    return json.loads(_run(node, f"{fns}\nconsole.log(JSON.stringify("
                                 f"boardShows({json.dumps(man)})));"))


def test_the_board_retires_when_a_slate_is_newer_than_it(node):
    assert _board_shows(node, WEEK_2) is False


def test_the_board_shows_when_it_is_newer_than_every_slate(node):
    man = {"artifacts": [
        {"name": "preds_2026_wk02", "generated_at": "2026-09-16T17:00:00+00:00"},
        {"name": "draft_board", "generated_at": "2027-07-10T12:00:00+00:00"}]}
    assert _board_shows(node, man) is True


def test_the_board_shows_when_no_slate_was_ever_published(node):
    assert _board_shows(node, JULY) is True


def test_one_newer_slate_among_older_ones_retires_the_board(node):
    man = {"artifacts": [
        {"name": "preds_2025_wk18", "generated_at": "2026-01-10T00:00:00+00:00"},
        {"name": "draft_board", "generated_at": BOARD_AT},
        {"name": "preds_2026_wk02", "generated_at": "2026-09-16T17:00:00+00:00"}]}
    assert _board_shows(node, man) is False


def test_a_board_with_no_usable_stamp_renders_nothing_rather_than_a_guess(node):
    for entry in ({"name": "draft_board"},
                  {"name": "draft_board", "generated_at": None},
                  {"name": "draft_board", "generated_at": "yesterday-ish"}):
        assert _board_shows(node, {"artifacts": [entry]}) is False, entry
    assert _board_shows(node, {"artifacts": [WEEK_2["artifacts"][0]]}) is False
    assert _board_shows(node, None) is False


def test_a_slate_stamp_that_does_not_parse_cannot_retire_the_board(node):
    man = {"artifacts": [{"name": "draft_board", "generated_at": BOARD_AT},
                         {"name": "preds_2026_wk02", "generated_at": "garbage"},
                         {"name": "preds_2026_wk03"}]}
    assert _board_shows(node, man) is True


def test_the_board_panel_is_gated_on_the_comparison():
    """The rule is only worth its tests if the render asks it. The panel is hidden off the
    answer and the board is not even fetched for a retired one."""
    text = PAGE.read_text()
    assert re.search(r"const shows = boardShows\(man\);", text), "render no longer asks"
    assert "boardEl.hidden = !shows;" in text, "the answer no longer hides the panel"
    assert re.search(r"shows \? await load\(\"draft_board\"\)", text)


def test_the_retired_board_is_hidden_not_deleted():
    """Retirement is a render decision: the section stays in the markup, and the rule never
    touches the file or its manifest entry."""
    text = PAGE.read_text()
    assert '<section id="p-board">' in text
    assert "section[hidden] { display: none; }" in text
    assert (DATA / "draft_board.json").exists()
    man = json.loads((DATA / "manifest.json").read_text())
    assert any(a["name"] == "draft_board" for a in man["artifacts"])


# --- both published weeks are reachable, and each says which season (#23) ----

def _other_weeks(node: str, man, shown):
    fn = "\n".join([_lift(r"function predsArt\(man\) \{[\s\S]*?\n\}"),
                    _lift(r"const ARCHIVE = \[[\s\S]*?\];"),
                    _lift(r"function otherWeeks\(man, shown\) \{[\s\S]*?\n\}")])
    return json.loads(_run(node, f"{fn}\nconsole.log(JSON.stringify(otherWeeks("
                                 f"{json.dumps(man)}, {json.dumps(shown)})));"))


def test_the_other_published_week_can_be_reached_from_the_one_on_screen(node):
    """#23 wants both published weeks rendered, not only the current one. The manifest
    describes the week it was written for and nothing else, so an older artifact is not
    discoverable from the page at all -- `preds_2025_wk18.json` is committed, is what the
    track record is scored from, and no reader could reach it. Both directions, so the
    archive is somewhere you can come back from."""
    from_current = _other_weeks(node, NEW_NAMING, "preds_2026_wk01")
    assert [w["name"] for w in from_current] == ["preds_2025_wk18"]
    from_archive = _other_weeks(node, NEW_NAMING, "preds_2025_wk18")
    assert [w["name"] for w in from_archive] == ["preds_2026_wk01"]
    assert all(w["label"].strip() for w in from_current + from_archive), "an unlabelled link"


def test_every_week_the_page_offers_is_a_file_that_is_committed(node):
    """The archive is a hand-copied list of names the writer wrote, which is the one thing
    that can rot: a renamed or dropped artifact becomes a link to a 404. Pinned to the
    directory rather than to a memory of it."""
    listed = _other_weeks(node, {"artifacts": []}, None)
    assert listed, "the archive is empty; `preds_2025_wk18.json` is committed and reachable"
    for w in listed:
        assert (DATA / f"{w['name']}.json").exists(), f"{w['name']} is offered but not there"


def test_the_slate_legend_names_its_season_and_escapes_it():
    """`site/data/` holds 2025 week 18 and 2026 week 01. "week 18" alone does not say which
    of them is on screen, and the subtitle's season is the manifest's, not the artifact's."""
    legend = re.search(r"const head = `[\s\S]*?`;", PAGE.read_text())
    assert legend, "the slate legend is no longer built as `const head = ...`"
    named = [e for e in re.findall(r"\$\{([^{}]*)\}", legend.group(0))
             if "season" in e.lower() or "stamp" in e.lower()]
    assert named, "the slate legend interpolates no season; a reader cannot tell which one"
    assert all("esc(" in e for e in named), f"unescaped season in the legend: {named}"


def test_the_slate_season_comes_off_the_artifact_and_only_the_artifact():
    """The manifest names the week it was written for -- the one week that can disagree with
    the rows on screen. Falling back to it is the skew the panel exists to make visible, so
    an artifact that does not say which season it is says that instead of borrowing one."""
    text = PAGE.read_text()
    shown = re.findall(r"const shown(?:Season|Week) = [^\n]*", text)
    assert len(shown) == 2, f"expected a season and a week off the artifact, got {shown}"
    assert not [s for s in shown if "man" in s], f"these read the manifest: {shown}"


def test_one_artifact_is_fetched_once_however_many_weeks_are_read(node):
    """The freeze, and it is structural rather than a convention: a model number that moves
    while games are on is worthless as a pre-registered claim. Keyed by the artifact's name
    so that reading an archived week cannot disturb the current one, and so the key always
    names the artifact actually served."""
    fn = "\n".join([_lift(r"const frozenPreds = new Map\(\);"),
                    _lift(r"async function frozen\(key\) \{[\s\S]*?\n\}")])
    body = (f"const asked = [];\nasync function load(n) {{ asked.push(n); return {{}}; }}\n"
            f"{fn}\n(async () => {{ await frozen('wk01'); await frozen('wk18');\n"
            f"await frozen('wk01'); await frozen('wk18');\n"
            f"console.log(JSON.stringify(asked)); }})();")
    assert json.loads(_run(node, body)) == ["wk01", "wk18"]


# --- the track record is per season, not one pooled curve (#23) --------------

BINS_2025 = [{"bin": "0.5-0.6", "n": 9, "predicted": 0.55, "actual": 0.44},
             {"bin": "0.6-0.7", "n": 7, "predicted": 0.64, "actual": 0.71}]
BINS_2026 = [{"bin": "0.5-0.6", "n": 3, "predicted": 0.55, "actual": 0.67}]
# The artifact as `site/data/track_record.json` holds it today: one pooled curve, no
# `seasons` key, and the next slate is the first run that will write one.
LEGACY_RECORD = {"n_scored": 16, "n_preregistered": 0, "is_backtest": True,
                 "note": "No pre-registered predictions yet.",
                 "log_loss": 0.5562546, "brier": 0.1940208, "bins": BINS_2025}
# The agreed payload: newest season first, and the top-level numbers are that season's.
SEASONS = [{"season": 2026, "n_scored": 3, "log_loss": 0.4111, "brier": 0.1322,
            "bins": BINS_2026},
           {"season": 2025, "n_scored": 16, "log_loss": 0.5562546, "brier": 0.1940208,
            "bins": BINS_2025}]
PER_SEASON = dict(LEGACY_RECORD, n_scored=19, seasons=SEASONS,
                  log_loss=0.4111, brier=0.1322, bins=BINS_2026)


def _record_body(node: str, tr: dict) -> str:
    lifted = "\n".join([
        _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);"),
        _lift(r"const fmt = \(v, d = 2\) =>[^\n]*"),
        _lift(r"const pct = v =>[^\n]*"),
        _lift(r"function table\(cols, rows\) \{[\s\S]*?\n\}"),
        _lift(r"function reliabilitySVG\(bins\) \{[\s\S]*?\n\}"),
        _lift(r"function trackRecordBody\(tr\) \{[\s\S]*?\n\}"),
        _lift(r"function intervalCoverageBody\(ic\) \{[\s\S]*?\n\}"),
    ])
    return _run(node, f"{lifted}\nconsole.log(trackRecordBody({json.dumps(tr)}));")


# What `hub.models.coverage.published_summary` writes into the record (#289, #293): the
# weekly interval's claim beside its verdict, and the survivor price by spread bucket with
# one bucket under the floor.
COVERAGE = {
    "centre": "prior", "lookahead": False, "n": 16061, "gate_subset": "unclipped",
    "gate_n": 10536, "gate_cov80": 0.7737, "gate_claim": 0.77, "band": 0.02,
    "verdict": "COVERS", "generated_at": "2026-09-13T00:00:00+00:00",
    "survivor": {
        "verdict": "HOLDS", "favourite_spread": 7.0, "favourite_predicted": 0.775,
        "favourite_actual": 0.797, "favourite_gap": 0.023, "favourite_sigma": 1.1,
        "favourite_n": 395, "n_games": 1420, "margin_sd": 12.741, "min_bucket": 50,
        "buckets": [
            {"bin": "0.0-3.0", "label": "<3", "n": 344, "predicted": 0.560, "actual": 0.535,
             "gap": -0.025, "thin": False},
            {"bin": "3.0-7.0", "label": "3-7", "n": 548, "predicted": 0.616, "actual": 0.630,
             "gap": 0.013, "thin": False},
            {"bin": "7.0-10.0", "label": "7-10", "n": 316, "predicted": 0.708,
             "actual": 0.728, "gap": 0.020, "thin": False},
            {"bin": "10.0-14.0", "label": "10-14", "n": 160, "predicted": 0.802,
             "actual": 0.850, "gap": 0.048, "thin": False},
            {"bin": "14.0-30.0", "label": "14+", "n": 12, "predicted": 0.879,
             "actual": 0.942, "gap": 0.063, "thin": True},
        ],
    },
}


def test_the_record_shows_the_interval_claim_beside_its_verdict(node):
    """#289: `COVERS` is printed with the 77% it was read against and the 80% the interval
    is labelled, so the verdict cannot be read as the label covering."""
    out = _record_body(node, dict(PER_SEASON, interval_coverage=COVERAGE))
    assert "77.4%" in out and "77.0%" in out and "COVERS" in out
    assert "labelled 80%" in out and "10,536" in out


def test_the_record_says_a_whole_board_figure_is_over_the_whole_board(node):
    """#309: once the gate grades every scored week, the page says so, gives the clipped share
    and the rows it could not score, and does not call the figure 'unclipped'. An artifact from
    before (`COVERAGE` above) keeps its old wording, which the test above holds."""
    whole = dict(COVERAGE, gate_subset="all", gate_n=15000, gate_cov80=0.8012,
                 gate_clipped_share=0.31, n_uncalibrated=1061, verdict="OVER-COVERS")
    out = _record_body(node, dict(PER_SEASON, interval_coverage=whole))
    assert "15,000 scored player-weeks" in out and "31.0% of them clipped at zero" in out
    assert "1,061 more not scored" in out and "unclipped" not in out
    assert "80.1%" in out and "OVER-COVERS" in out and "77.0%" in out


def test_the_record_calls_the_weekly_figure_a_read_and_names_the_audit(node):
    """#310: the weekly figure is a report, never a gate. The page says the binding verdict is
    taken at audits and, once one was taken, which look it was."""
    out = _record_body(node, dict(PER_SEASON, interval_coverage=COVERAGE))
    assert "this week's read" in out and "none has been taken yet" in out
    audited = dict(COVERAGE, audit={"look": 2, "looks": 3, "marginal": "COVERS"})
    out = _record_body(node, dict(PER_SEASON, interval_coverage=audited))
    assert "Audit look 2 of 3: COVERS" in out


def test_the_record_shows_every_survivor_bucket_and_marks_the_thin_one(node):
    """#293: five rows, the count beside each rate, and the twelve-game bucket present and
    marked rather than gone."""
    out = _record_body(node, dict(PER_SEASON, interval_coverage=COVERAGE))
    for label in ("&lt;3", "3-7", "7-10", "10-14", "14+"):
        assert f">{label}</td>" in out, f"bucket {label} is not on the page"
    assert "12 (under 50)" in out, "the thin bucket is shown and marked, not dropped"
    assert "344" in out and "HOLDS" in out and "+0.063" in out


CURRENT_ROW = {"label": "out-of-sample", "season": 2026, "weeks_complete": 4, "n": 0,
               "looks": {"2": {"after_week": 14, "reached": False},
                         "3": {"after_week": 18, "reached": False}}}


def test_the_record_shows_the_current_season_row_apart_from_the_backtest(node):
    """#423: the 2026 weeks are a separate, labelled, out-of-sample line that says it carries no
    verdict until looks 2 and 3, and says so when nothing is scored yet. Not folded into the
    backtest's figure."""
    out = _record_body(node, dict(PER_SEASON, interval_coverage=dict(
        COVERAGE, current_season=CURRENT_ROW)))
    assert "2026 so far, out of sample" in out and "no week has been scored yet" in out
    assert "not pooled" in out and "weeks 14 and 18" in out
    scored = dict(CURRENT_ROW, n=1234, cov80=0.8123, cov68=0.6712, clipped_share=0.31, sigma=0.9)
    out = _record_body(node, dict(PER_SEASON, interval_coverage=dict(
        COVERAGE, current_season=scored)))
    assert "81.2%" in out and "1,234" in out and "0.9 sigma" in out


def test_a_hostile_current_row_cannot_reach_the_markup(node):
    out = _record_body(node, dict(PER_SEASON, interval_coverage=dict(
        COVERAGE, current_season=dict(CURRENT_ROW, season=HOSTILE))))
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_a_record_with_no_coverage_measurement_still_renders(node):
    """The committed `track_record.json` carries `interval_coverage: null` until the slate
    next runs; the record must render its calibration without it."""
    out = _record_body(node, dict(PER_SEASON, interval_coverage=None))
    assert "<svg" in out and "Weekly interval" not in out and "spread" not in out


def test_the_coverage_block_escapes_its_interpolations():
    """The same rule `test_the_record_panel_escapes_its_own_numbers_too` holds the record
    panel to: every hand-built `${...}` in the coverage block goes through `esc`."""
    body = _lift(r"function intervalCoverageBody\(ic\) \{[\s\S]*?\n\}")
    raw = [e for e in re.findall(r"\$\{([^{}]*)\}", body) if "esc(" not in e]
    assert not raw, f"unescaped interpolations in the coverage block: {raw}"


def test_a_hostile_bucket_label_cannot_reach_the_markup(node):
    hostile = dict(COVERAGE, survivor=dict(
        COVERAGE["survivor"],
        buckets=[dict(COVERAGE["survivor"]["buckets"][0], label=HOSTILE)]))
    out = _record_body(node, dict(PER_SEASON, interval_coverage=hostile))
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_the_numbers_are_shown_in_the_branch_every_artifact_actually_lands_in(node):
    """`n_preregistered` stays 0 until a prediction's commit predates kickoff, so this is
    the branch the committed artifact and every near-future one renders through. Metrics
    withheld here are metrics no reader ever sees, and #23 asks for the calibration to be
    visible per season -- which is a different claim from presenting a backtest as a record.
    The sentence that refuses the second one stays exactly where it was."""
    out = _record_body(node, PER_SEASON)
    assert "backtest, not a record" in out, "the framing that makes showing these honest"
    assert "0.5563" in out and "0.4111" in out, "per-season metrics are still withheld"
    assert "2025" in out and "2026" in out
    assert "<svg" in out, "no calibration curve at all"
    # The top-level bins are the newest season's, so the diagram is one season's curve and
    # must say which -- unlabelled, beside a table of seasons, it reads as the pool.
    assert re.search(r"Calibration for [^<]*2026", out), "the diagram names no season"


def test_a_pre_registered_record_reads_the_same_way(node):
    """Same numbers, different framing: once predictions are pre-registered the panel stops
    calling itself a backtest and says how many were."""
    out = _record_body(node, dict(PER_SEASON, n_preregistered=19))
    assert "backtest" not in out
    assert "19 pre-registered" in out
    assert "0.5563" in out and "0.4111" in out and "<svg" in out


def test_a_partly_pre_registered_record_says_how_many_are_which(node):
    """#422: 64 pre-registered and 16 backfilled must not share one framing."""
    out = _record_body(node, dict(PER_SEASON, n_scored=80, n_preregistered=64))
    assert "64 pre-registered, 80 scored" in out
    assert "16 scored after the" in out and "a backtest, not a record" in out


def test_an_unknown_pre_registration_count_is_not_rendered_as_zero(node):
    """`n_preregistered: null` is "could not tell". The page used to coalesce it to 0 and
    say "backtest, not a record", the same defect as a producer that writes 0."""
    note = "Pre-registration could not be checked this run."
    out = _record_body(node, dict(PER_SEASON, n_preregistered=None, note=note))
    assert note in out
    assert "backtest" not in out and "0 pre-registered" not in out


def test_an_artifact_written_before_the_split_shows_its_pooled_numbers(node):
    """`site/data/track_record.json` will not carry `seasons` until the next slate runs, and
    a panel that needs the new key would go blank on the deploy that shipped it. Nothing
    here can unpool a pooled curve, so it is shown as it is rather than relabelled."""
    out = _record_body(node, LEGACY_RECORD)
    assert "backtest, not a record" in out
    assert "0.5563" in out and "<svg" in out
    assert "<th>season</th>" not in out, "invented a per-season split the artifact lacks"


def test_a_hostile_season_label_cannot_reach_the_markup(node):
    """The seasons come off our own artifact, but the panel is where a new table was added,
    and `table` escaping by default is only a property while everything goes through it."""
    out = _record_body(node, dict(PER_SEASON, seasons=[dict(SEASONS[0], season=HOSTILE)]))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_the_record_panel_escapes_its_own_numbers_too():
    """`FROM_A_FEED` is silent on `n_preregistered` and `n_scored` because they are ours,
    which is exactly how the rule erodes: a count is a number until someone makes it a
    string. Every hand-built interpolation in this panel goes through `esc`, no exemptions
    to remember."""
    body = _lift(r"function trackRecordBody\(tr\) \{[\s\S]*?\n\}")
    raw = [e for e in re.findall(r"\$\{([^{}]*)\}", body) if "esc(" not in e]
    assert not raw, f"unescaped interpolations in the record panel: {raw}"


def test_an_archived_week_is_not_joined_to_todays_scoreboard():
    """Found by rendering it: the live map is keyed by the matchup alone -- `AWAY_HOME`, no
    season -- and three of 2026 week 1's fixtures repeat 2025 week 18's. Reaching the older
    week therefore put tonight's score beside a prediction made for a different season's
    game, under a column captioned "live", which is precisely the confusion the frozen/live
    split exists to prevent. An archived week gets no overlay rather than a plausible wrong
    one, and the panel says why.

    The join is guarded on `archived` rather than on `viewing` directly since #57: the two
    say the same thing, and the test below is why there is only one of them."""
    text = PAGE.read_text()
    joined = re.search(r"const live = [^\n]*", text)
    assert joined, "the slate no longer loads the live overlay where the test expects"
    assert "archived" in joined.group(0), (
        f"{joined.group(0)!r} joins scores to whatever week is on screen; guard it on the "
        f"week the manifest names.")


# --- the slate asks what it is showing once (#57) ----------------------------

def _slate_render() -> str:
    """The slate render, both halves, in the order the panel runs them."""
    return "\n".join([_lift(r"async function renderSlate\(man\) \{[\s\S]*?\n\}"),
                      _lift(r"async function slateBody\([^)]*\) \{[\s\S]*?\n\}")])


def test_the_slate_answers_the_archived_week_question_in_one_place():
    """"Is this an archived week" was re-branched on `viewing` four times inside one render
    -- the manifest entry it reports staleness from, the live join, the age of the scores,
    and the warning. Four readings of one question is three chances for one of them to be
    the odd one out, and the one that mattered was the live join: an archived week joined to
    tonight's scoreboard shows a real score beside a prediction for a different season's
    game. Answered once now, and everything downstream reads the answer."""
    src = _slate_render()
    answered = re.search(r"const archived = ([^\n]*)", src)
    assert answered, "the slate no longer names what it is showing; #57 collapsed it to one"
    assert "viewing" in answered.group(1), (
        f"`archived` is not derived from the week a reader clicked: {answered.group(1)!r}")
    after = src[answered.end():]
    assert "viewing" not in after, (
        f"the slate re-asks which week is on screen after answering it: "
        f"{[ln.strip() for ln in after.splitlines() if 'viewing' in ln]}")


def test_the_other_published_weeks_are_offered_at_the_slates_one_exit():
    """The links were concatenated onto the html at each of the two exits of the render.
    Two exits is two chances to forget, and the one that would be forgotten is the empty
    case -- a week that could not be published is exactly when a reader wants the week that
    was. One exit, so the links cannot be missing from one of them."""
    text = PAGE.read_text()
    exits = text.count('panel("p-slate"')
    assert exits == 1, (
        f"the slate renders through {exits} exits; the published-week links have to be "
        f"appended at each of them, and one of them will be the one that forgets.")


def test_a_group_that_records_no_season_is_captioned_truthfully(node):
    """`publish.track_record` puts rows carrying no season into a group of their own with
    `season: null`, ordered last -- so it is the *newest* group only when every scored row
    is undated, which is the shape `preds_wk18.json` had before the season joined the
    filename. "Calibration for ." is not a sentence, and a caption with a hole in it is how
    a reader learns to distrust the numbers beside it."""
    undated = [dict(SEASONS[0], season=None)]
    out = _record_body(node, dict(PER_SEASON, seasons=undated))
    assert "Calibration for ." not in out and "Calibration for <" not in out
    assert "unrecorded" in out, "neither the caption nor the row says the season is missing"
    # And the group still appears in the table rather than as an empty cell.
    assert "<td class=\"num \"></td>" not in out


# --- the remaining plan says what it is scoped to (#57) ----------------------

# The survivor artifact as `publish.survivor` writes it, mid-season: week 1 is behind the
# plan, KC was spent in it, and week 3 reached the grid only because the store was read.
SURVIVOR = {"name": "survivor", "source": "hub.season.survivor", "season": 2026, "n": 2,
            "rows": [{"week": 2, "team": "SF", "win_prob": 0.70},
                     {"week": 3, "team": "SEA", "win_prob": 0.60}],
            "survival": 0.42, "unpriced_weeks": [], "weeks_remaining": [2, 3],
            "weeks_played": [1], "spent": ["KC"], "snapshot_only_weeks": [3]}


def _survivor_body(node: str, surv: dict) -> str:
    lifted = "\n".join([
        _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);"),
        _lift(r"const pct = v =>[^\n]*"),
        _lift(r"const weekList = ws =>[^\n]*"),
        _lift(r"function table\(cols, rows\) \{[\s\S]*?\n\}"),
        _lift(r"function survivorBody\(surv\) \{[\s\S]*?\n\}"),
    ])
    return _run(node, f"{lifted}\nconsole.log(survivorBody({json.dumps(surv)}));")


def test_the_remaining_plan_says_which_weeks_it_is_over_and_which_are_behind_it(node):
    """`weeks_played` was published and read by nothing at all -- no page code, no test, no
    CLI -- and `weeks_remaining` by one test that rendered nothing (#57). A field nobody
    reads is what this repo deletes on sight, and the alternative it chose here is that the
    panel shows it: a plan whose first row is week 9 has to say why, and a reader should not
    have to open the JSON to find the eight weeks that are behind it."""
    out = _survivor_body(node, SURVIVOR)
    assert "wk 2" in out and "wk 3" in out, "the panel does not say which weeks it covers"
    assert "wk 1" in out and "played" in out, (
        f"nothing says week 1 is behind this plan, so its first row is unexplained: {out}")


def test_a_plan_with_nothing_behind_it_says_so_rather_than_leaving_a_hole(node):
    """Preseason is the state the panel spent all summer in. "0 week(s) already played" and
    a dangling colon are how a reader learns the scope line is boilerplate."""
    out = _survivor_body(node, dict(SURVIVOR, weeks_played=[], spent=[]))
    assert ": ." not in out and "wk ." not in out
    assert "played" in out or "behind" in out, f"the scope line says nothing at all: {out}"


def test_an_artifact_written_before_the_scope_existed_still_renders(node):
    """`site/data/survivor.json` as committed carries `season`, `survival`,
    `unpriced_weeks` and `snapshot_only_weeks` and none of the rest, and it will not carry
    them until the next slate runs -- so it is exactly what the deploy that ships this
    renders. Absent and empty are different claims: `[]` says no week has been played,
    missing says the artifact never said. Reading both as zero would caption a plan of
    eighteen weeks "0 week(s) covered", which is the caption with a hole in it the record
    panel already learned not to leave."""
    legacy = {k: v for k, v in SURVIVOR.items()
              if k not in ("weeks_remaining", "weeks_played", "spent")}
    out = _survivor_body(node, legacy)
    assert "0 week(s)" not in out, f"an absent scope read as a scope of nothing: {out}"
    assert "&mdash; ." not in out and ": .</p>" not in out, f"a hole in the caption: {out}"
    assert "already played" not in out, "claimed a scope the artifact does not record"
    assert "SF" in out and "SEA" in out, "the plan itself stopped rendering"


def test_the_weeks_that_are_in_the_remaining_plan_only_because_the_store_was_read_are_named(node):
    """`snapshot_only_weeks` is the third scope field and had no reader either. It is the
    one that decides whether reading the store bought anything, and it moves -- a week in
    the list today is not in it in December -- so it is reported rather than assumed."""
    out = _survivor_body(node, SURVIVOR)
    assert "wk 3" in out and "snapshot" in out.lower(), (
        f"the panel does not say which weeks nflverse's own field cannot price: {out}")
    quiet = _survivor_body(node, dict(SURVIVOR, snapshot_only_weeks=[]))
    assert "snapshot" not in quiet.lower(), "a caveat about no weeks at all"


def test_a_hostile_team_name_cannot_reach_the_survivor_markup(node):
    """The spent teams and the picked ones are nflverse's, and the scope prose around them
    is hand-built rather than built by `table`."""
    out = _survivor_body(node, dict(SURVIVOR, spent=[HOSTILE],
                                    rows=[{"week": 2, "team": HOSTILE, "win_prob": 0.7}]))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_the_survivor_panel_escapes_its_own_numbers_too():
    """The same rule the record panel is held to, for the same reason: `FROM_A_FEED` is
    silent on week numbers because they are ours, and that is exactly how the rule erodes.
    Every hand-built interpolation in this panel goes through `esc`, no exemptions."""
    body = _lift(r"function survivorBody\(surv\) \{[\s\S]*?\n\}")
    raw = [e for e in re.findall(r"\$\{([^{}]*)\}", body)
           if "esc(" not in e and "pct(" not in e]
    assert not raw, f"unescaped interpolations in the survivor panel: {raw}"


# --- the page speaks one confidence grammar (#347) ---------------------------
#
# Every number is frozen, live, stale or unconfirmed and the page says which by how it draws
# the number. Each state below is planted -- an artifact older than its window, a pool rule
# nobody confirmed -- and the rendered label asserted, because a test that renders only the
# healthy case passes on a page that has no grammar at all. The stale test is the one a
# guard on the page points at, and `test_a_panel_with_the_wrong_state_turns_the_grammar_red`
# plants the mislabelled panel and requires red.

STATE_CLASSES = {"frozen": "state-frozen", "live": "state-live", "stale": "state-stale",
                 "unconfirmed": "state-unconfirmed"}


def _panel_parts(node: str, opts_js: str) -> dict:
    """`panelParts(...)` as the page defines it, run in node. `opts_js` is a JS expression, so
    a test can plant an age relative to now rather than a date that will go out of date."""
    lifted = "\n".join([
        _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);"),
        _lift(r"function ago\(iso\) \{[\s\S]*?\n\}"),
        _lift(r"const STATE = \{[\s\S]*?\};"),
        _lift(r"function staleNote\(stale\) \{[\s\S]*?\n\}"),
        _lift(r"function panelParts\([\s\S]*?\n\}"),
        _lift(r"const staleOf = [^\n]*"),
    ])
    return json.loads(_run(node, f"{lifted}\nconsole.log(JSON.stringify({opts_js}));"))


PLANTED_STALE = ("(() => { const art = {stale: true, reason: 'ESPN unreachable', "
                 "generated_at: new Date(Date.now() - 3 * 864e5).toISOString()}; "
                 "return panelParts({age: '12 rostered', html: '<td>Q. Example 18.4</td>', "
                 "stale: staleOf(art)}); })()")


def test_a_stale_artifact_renders_in_the_stale_state_with_its_age(node):
    """Planted: an artifact three days old that the manifest marks stale. The warning is in
    the stale state, carries the age, the panel is stale, and the value is still there."""
    got = _panel_parts(node, PLANTED_STALE)
    assert got["cls"] == STATE_CLASSES["stale"]
    assert f'class="warn {STATE_CLASSES["stale"]}"' in got["body"]
    assert "Stale: ESPN unreachable" in got["body"]
    assert "artifact 3d ago" in got["body"], got["body"]
    assert "<td>Q. Example 18.4</td>" in got["body"], "a stale panel must not blank its value"


def test_a_stale_artifact_that_cannot_say_its_age_says_so(node):
    """The age stamp is required. An artifact with no `generated_at` is stale with an
    unknown age, and the warning says that rather than omitting the stamp."""
    got = _panel_parts(node, "panelParts({html: 'v', stale: staleOf("
                             "{stale: true, reason: 'r', generated_at: null})})")
    assert "artifact age unknown" in got["body"]
    assert got["cls"] == STATE_CLASSES["stale"]


def test_a_current_artifact_is_frozen_and_carries_no_stale_label(node):
    """The control for the two above: the same panel with a current manifest entry is the
    default state, and nothing on it says stale."""
    got = _panel_parts(node, "panelParts({html: 'v', stale: staleOf("
                             "{stale: false, reason: null, generated_at: 'x'})})")
    assert got["cls"] == STATE_CLASSES["frozen"]
    assert got["body"] == "v"


def test_a_panel_with_no_data_renders_its_reason_and_does_not_throw(node):
    got = _panel_parts(node, "panelParts({warn: 'No roster yet.'})")
    assert got["body"] == '<p class="warn">No roster yet.</p>'
    assert got["cls"] == STATE_CLASSES["frozen"]


def test_an_unconfirmed_pool_rule_renders_in_the_unconfirmed_state(node):
    """Planted: a survivor artifact naming a rule nobody has confirmed. The note renders in
    the unconfirmed state and names the rule; with nothing unconfirmed, no state is drawn."""
    lifted = "\n".join([
        _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);"),
        _lift(r"const STATE = \{[\s\S]*?\};"),
        _lift(r"function unconfirmedNote\(surv\) \{[\s\S]*?\n\}"),
    ])
    out = _run(node, f"{lifted}\nconsole.log(unconfirmedNote("
                     f"{json.dumps({'unconfirmed': ['co_survivor_rule']})}));")
    assert f'class="{STATE_CLASSES["unconfirmed"]}"' in out
    assert "co_survivor_rule" in out
    clean = _run(node, f"{lifted}\nconsole.log(unconfirmedNote({{unconfirmed: []}}));")
    assert STATE_CLASSES["unconfirmed"] not in clean


def test_the_roster_withheld_note_renders_in_the_unconfirmed_state():
    """The note is built inside `render`, which needs a DOM to run, so this reads the page:
    the sentence that names the withheld players is the one opening in the state."""
    got = re.search(r'`<p class="([^"]*)">Withheld as unavailable:', PAGE.read_text())
    assert got, "the withheld note is gone, or no longer an unlabelled paragraph"
    assert got.group(1) == "${STATE.unconfirmed}"


def test_every_stale_warning_goes_through_the_one_stale_note():
    """No panel spells its own `Stale:` warning. A hand-written one would render without the
    age, which is the omission the grammar exists to stop."""
    text = PAGE.read_text()
    assert len(re.findall(r"`[^`]*Stale:", text)) == 1
    sites = re.findall(r"stale: staleOf\(", text)
    assert len(sites) == 5, f"expected the five panel exits that can be stale, got {len(sites)}"


def _style() -> str:
    text = PAGE.read_text()
    return text[text.index("<style>"):text.index("</style>")]


def test_the_four_states_are_a_named_token_set_in_both_colour_schemes():
    style = _style()
    for cls in STATE_CLASSES.values():
        assert re.search(rf"\.{cls}\b[^{{]*\{{", style), f"no rule for .{cls}"
    split = style.index("prefers-color-scheme: dark")
    light, dark = style[:split], style[split:style.index("* { box-sizing")]
    tokens = set(re.findall(r"--st-[\w-]+(?=:)", light))
    assert tokens, "no --st- tokens declared"
    missing = {t for t in tokens if t not in dark}
    assert not missing, f"tokens with no dark-scheme value: {sorted(missing)}"


def _rule(selector: str) -> str:
    got = re.search(rf"(?m)^\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", _style())
    assert got, f"no rule for {selector!r}"
    return got.group(1)


def test_the_four_states_are_drawn_differently():
    """Distinguishable by treatment and not by hue alone: solid rule, tinted field, dotted
    rule on dimmed ink, dashed hatch."""
    assert "solid" in _rule(".state-frozen")
    assert "var(--st-live-tint)" in _rule(".state-live, .grp-live")
    assert "dotted" in _rule(".state-stale")
    unconfirmed = _rule(".state-unconfirmed")
    assert "dashed" in unconfirmed and "repeating-linear-gradient" in unconfirmed


def test_the_slates_live_group_is_an_alias_of_the_live_state_not_a_second_rule():
    defs = re.findall(r"(?m)^\s*([^\n{}]*\.grp-live[^\n{}]*?)\s*\{", _style())
    assert defs == [".state-live, .grp-live"], (
        f"`.grp-live` must be defined once, sharing `.state-live`'s rule; found {defs}")


def _token(block: str, name: str) -> str:
    got = re.search(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{3,6}})\b", block)
    assert got, f"no {name} colour in that scheme"
    return got.group(1)


def _linear(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _full(hex_: str) -> str:
    h = hex_.lstrip("#")
    return "#" + ("".join(c * 2 for c in h) if len(h) == 3 else h)


def _luminance(hex_: str) -> float:
    h = _full(hex_).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def _blend(fg: str, bg: str, alpha: float) -> str:
    f, b = _full(fg).lstrip("#"), _full(bg).lstrip("#")
    return "#" + "".join(f"{round(alpha * int(f[i:i + 2], 16) + (1 - alpha) * int(b[i:i + 2], 16)):02x}"
                         for i in (0, 2, 4))


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _stale_body_contrast(style: str) -> dict[str, float]:
    """WCAG contrast of stale body text against the card, per scheme, from the tokens.

    Any `opacity` a `.state-stale` rule applies to body content is blended in: dimming the
    ink and then fading the element stacks, which is how this was measured at 2.8:1."""
    split = style.index("prefers-color-scheme: dark")
    schemes = {"light": style[:split], "dark": style[split:style.index("* { box-sizing")]}
    alpha = 1.0
    for sel, body in re.findall(r"(?m)^\s*([^\n{}]*\.state-stale[^\n{}]*)\{([^}]*)\}", style):
        if ".warn" in sel and ":not" not in sel:
            continue
        for op in re.findall(r"(?<![-\w])opacity:\s*([\d.]+)", body):
            alpha *= float(op)
    out = {}
    for name, block in schemes.items():
        ink = re.search(r"--st-stale-ink:\s*(#[0-9a-fA-F]{3,6})\b", block)
        card = re.search(r"--card:\s*(#[0-9a-fA-F]{3,6})\b", block)
        assert ink, f"no --st-stale-ink in the {name} scheme"
        card = card or re.search(r"--card:\s*(#[0-9a-fA-F]{3,6})\b", style)
        assert card
        out[name] = _contrast(_blend(ink.group(1), card.group(1), alpha), card.group(1))
    return out


def test_stale_body_text_meets_wcag_aa_in_both_schemes_and_stays_distinct_from_frozen():
    """Stale text is body text: at least 4.5:1, with the opacity stacked in. And it is not
    simply the frozen ink: it must be measurably dimmer than frozen, or the state is lost."""
    style = _style()
    got = _stale_body_contrast(style)
    assert all(v >= 4.5 for v in got.values()), f"stale body contrast under 4.5:1: {got}"
    split = style.index("prefers-color-scheme: dark")
    for name, block in {"light": style[:split], "dark": style[split:]}.items():
        fg = _token(block, "--fg")
        card = _token(block, "--card")
        assert _contrast(fg, card) > got[name] + 3, f"stale is not dimmer than frozen in {name}"


def test_a_too_faint_stale_value_turns_the_contrast_test_red():
    """The planted failure: the original values, and a stacked opacity, must both fail."""
    style = _style()
    faint = style.replace("--st-stale-ink: #5b6170", "--st-stale-ink: #9ca3af")
    assert faint != style
    assert min(_stale_body_contrast(faint).values()) < 4.5
    stacked = style.replace(".state-stale { color", ".state-stale > :not(.warn) { opacity: 0.5; }\n  .state-stale { color")
    assert stacked != style
    assert min(_stale_body_contrast(stacked).values()) < 4.5


def test_reduced_motion_still_stops_the_live_pulse():
    style = _style()
    assert "animation: pulse" in style
    media = re.search(r"@media \(prefers-reduced-motion: reduce\) \{([^}]*)\}", style)
    assert media and "animation: none" in media.group(1)
    assert ".tag.live::before" in media.group(1)


# --- the page's own guards, proved by deleting them (#61) --------------------
#
# `tests/contracts/test_guards_are_load_bearing.py` does this for the guards in
# `src/`: declare a guard where it lives, delete the block, run the target tests against the
# mutant, require red. It reads Python, so the page -- the public, unattended surface that
# renders third-party feed values -- sat outside it. Three protections live here and none had
# the property: `esc`, `table` escaping every cell and header without being asked, and the
# static interpolation scan above, which is the thing that would catch a new unescaped
# `${...}` and whose own removal was caught by nothing.
#
# The grammar, the record type, the excision and the reading of a child run are all
# `tests/guardlib.py` since issue #65, shared with the Python harness and the shell scan's.
# The bracket after a guard's name resolves by that module's one rule -- pytest selectors
# relative to `tests/` -- which is why the page's markers name this module rather than
# carrying nothing and leaving the harness to assume it.
#
# The two traps carried over from the Python harness rather than rediscovered:
#
#   * A mutant that no longer parses is a *harness* error, never a pass. Excising a block can
#     leave source that will not run, everything then fails, and that looks exactly like a
#     load-bearing guard. `node --check` is the JavaScript analogue of `ast.parse`.
#   * The harness must be running against the mutant. The first Python version passed the
#     mutated tree through an environment variable that pytest's own `pythonpath` shadowed,
#     so every mutant ran against the real source and all four guards reported green. Here
#     the override is proved on every run rather than assumed -- see the pair of tests under
#     "the harness is looking at the mutant".

# `// GUARD <name> [<selectors>]: <why>` ... `// /GUARD`, inside the page's script.
PAGE_GUARD = marker("GUARD", "//")
TESTS = ROOT / "tests"

# Set on the child run, which exists to exercise the page's tests against a mutated copy.
# Without it the child would re-enter this section and mutate the mutant, forever.
IN_CHILD = "HUB_DASHBOARD_MUTANT" in os.environ
parent_only = pytest.mark.skipif(
    IN_CHILD, reason="the child run exercises the page's tests, not the harness's")


def _declared(text: str) -> list[Guard]:
    return declared(text, PAGE_GUARD, PAGE)


def _without(text: str, guard: Guard) -> str:
    """The page with that guard's block removed, markers and all."""
    return excise(text, guard)


def _script(text: str) -> str:
    """The page's script, for `node --check`.

    One block today. If a second ever appears this stops rather than checking the first and
    calling the rest parsed -- a syntax check that silently covers half the page is the same
    kind of quietly-dead guard this file is about.
    """
    blocks = re.findall(r"<script>([\s\S]*?)</script>", text)
    assert len(blocks) == 1, (
        f"expected one <script> block, found {len(blocks)}; check each of them rather than "
        f"only the first")
    return blocks[0]


def _parses(node: str, js: str) -> subprocess.CompletedProcess:
    """`node --check` on the page's script -- the analogue of `ast.parse` on a Python mutant.

    A file rather than stdin because `--check` takes a path, and `.js` rather than `.mjs`
    because the page's script is a classic script and nothing in it is a module."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mutant.js"
        path.write_text(js)
        return subprocess.run([node, "--check", str(path)],
                              capture_output=True, text=True, timeout=30)


def _mutant_site(tmp_path: Path, text: str) -> Path:
    """A whole copy of `site/`, with the mutated page in it.

    The directory and not the file alone: `DATA` above is `PAGE.parent / "data"`, so a page
    written somewhere without its artifacts beside it would fail the archive tests for a
    reason that has nothing to do with the guard -- every mutant red, every guard reported
    load-bearing, which is the trap this file is built around.
    """
    site = tmp_path / "site"
    if not site.exists():
        shutil.copytree(ROOT / "site", site)
    (site / "index.html").write_text(text)
    return site / "index.html"


def _run_module_against(page: Path, selectors: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run the page's tests against the page at `page`.

    `selectors` are a guard's own, relative to `tests/`; with none given the whole module runs,
    which is what the pair of tests below that prove the child is reading the mutant want.
    """
    args = selector_args(selectors, TESTS) if selectors else [str(Path(__file__).resolve())]
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--tb=no", *CHILD_FLAGS],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "HUB_DASHBOARD_PAGE": str(page), "HUB_DASHBOARD_MUTANT": "1"},
    )


def _failed(got: subprocess.CompletedProcess) -> set[str]:
    """The test names the child reported as failures -- not errors, which prove nothing."""
    return set(outcome(got).failed)


# --- the harness's own premise -----------------------------------------------
#
# `parent_only` throughout: these read the markers, and an excision has just deleted a pair
# of them. Left running in the child they would fail on every mutant, and every guard would
# be reported load-bearing on the strength of the harness failing to find itself.

@parent_only
def test_the_scan_finds_the_guards_the_page_declares():
    """A marker syntax that matched nothing would make every excision below vacuously
    absent, and this section would report a clean sweep of zero guards."""
    found = {g.name for g in _declared(PAGE.read_text())}
    assert "esc-escapes-what-reaches-the-page" in found, (
        f"the escaper is unmarked; found {sorted(found)}")
    assert "table-builds-every-cell-in-one-place" in found, (
        f"the table helper is unmarked; found {sorted(found)}")
    stale = [(g.name, s) for g in _declared(PAGE.read_text()) for s in (g.selectors or ("",))
             if not s or not selector_file(s, TESTS).exists()]
    assert not stale, (
        f"these brackets do not resolve to a file under tests/: {stale}. The bracket is one "
        f"or more pytest selectors relative to `tests/` -- see tests/guardlib.py.")


@parent_only
def test_a_guard_that_cannot_be_removed_cleanly_is_a_harness_error(node, tmp_path):
    """The trap. Excising a block can leave script that will not parse, every lifted
    function then fails, and that is indistinguishable from a load-bearing guard until
    something checks. `node --check` is what checks."""
    page = tmp_path / "brittle.html"
    page.write_text("<script>\nfunction f(xs) {\n  return xs.map(x =>\n"
                    "    // GUARD only-expression: removing this leaves an arrow with no body.\n"
                    "    x + 1\n"
                    "    // /GUARD\n  );\n}\n</script>\n")
    guard = _declared(page.read_text())[0]
    assert guard.name == "only-expression"
    broken = _parses(node, _script(_without(page.read_text(), guard)))
    assert broken.returncode != 0, "node accepted an arrow function with no body"


@parent_only
def test_the_pages_own_guards_survive_their_excision_syntactically(node):
    """Same check against the real markers: if one of them straddles a brace, the red run
    below would be a parse failure wearing a guard's name."""
    text = PAGE.read_text()
    for guard in _declared(text):
        got = _parses(node, _script(_without(text, guard)))
        assert got.returncode == 0, (
            f"removing guard {guard.name!r} leaves script that will not parse "
            f"({got.stderr.strip().splitlines()[-1] if got.stderr else '?'}). Widen the "
            f"markers so the block stands alone -- as written, a red run proves nothing.")


# --- the harness is looking at the mutant ------------------------------------

@parent_only
def test_the_child_run_passes_on_the_real_page(node):
    """The green half of the pair. A child that failed whatever it was handed would report
    every guard load-bearing, so the harness has to be able to come back green at all."""
    got = _run_module_against(PAGE)
    assert got.returncode == 0, f"the module does not pass on its own page:\n{got.stdout}"


@parent_only
def test_a_page_that_forgets_to_escape_is_caught_only_by_the_scan(node, tmp_path):
    """The red half, and it does two jobs.

    It proves the child is reading the mutant: the page handed over differs from the real one
    by a single unescaped `${r.player}`, and the run must go red where the run above went
    green. If the environment override were shadowed -- the exact way the Python harness
    fooled itself for a day -- the child would read the real page and this would pass green,
    which is why the assertion is written this way round.

    And it proves the static scan is load-bearing: nothing else in this module notices that
    interpolation, so deleting `test_every_feed_value_in_the_markup_goes_through_esc` turns
    this red. It is the sharpest of the three (#61): the thing that would catch a new
    unescaped interpolation, whose own removal was caught by nothing.
    """
    forgetful = PAGE.read_text().replace(
        "</script>", "const forgotten = `<td>${r.player}</td>`;\n</script>")
    assert forgetful != PAGE.read_text(), "the injection did not land"
    assert _parses(node, _script(forgetful)).returncode == 0, "the injected page must parse"

    got = _run_module_against(_mutant_site(tmp_path, forgetful))
    assert got.returncode != 0, (
        "a page with an unescaped feed value passed. Either the child is reading the real "
        f"page rather than the mutant, or the static scan is gone.\n{got.stdout}")
    assert _failed(got) == {"test_every_feed_value_in_the_markup_goes_through_esc"}, (
        f"expected the scan alone to notice, got {sorted(_failed(got))}. If it is empty the "
        f"child errored rather than failed, and this proves nothing.")


@parent_only
def test_a_table_that_stops_escaping_its_cells_is_caught(node, tmp_path):
    """The claim the marked block cannot make, made here instead.

    `table-builds-every-cell-in-one-place` wraps the two statements that build the header and
    the cells, which is the smallest span that contains the escaping and still leaves script
    `node --check` accepts. Deleting it therefore removes the escaping *and* the markup around
    it, so its red proves only that the helper stops rendering -- which is why its sentence
    claims the single place and stops there (#65). This mutates only the two `esc(...)`
    wrappers and leaves the table otherwise intact -- the change a new column would make by
    forgetting -- and is what proves the escaping itself is load-bearing.

    Written against the page's exact expressions, which can drift; if they do, the
    replacement is a no-op and the assertion below says so rather than passing quietly.
    """
    text = PAGE.read_text()
    unwrapped = text.replace("${esc(c.label)}", "${c.label}").replace(
        "${esc(c.get(r))}", "${c.get(r)}")
    assert unwrapped != text, (
        "neither `esc(c.label)` nor `esc(c.get(r))` is in the page any more; this test is "
        "mutating nothing. Re-point it at whatever `table` escapes with now.")
    assert _parses(node, _script(unwrapped)).returncode == 0

    got = _run_module_against(_mutant_site(tmp_path, unwrapped))
    assert got.returncode != 0, (
        f"`table` stopped escaping its cells and every test still passed.\n{got.stdout}")
    assert "test_the_table_escapes_every_cell_without_being_asked" in _failed(got), (
        f"something noticed, but not the test whose job this is: {sorted(_failed(got))}")


# --- the habit ---------------------------------------------------------------

@parent_only
@pytest.mark.parametrize("guard", _declared(PAGE.read_text()) or [None], ids=repr)
def test_removing_a_guard_turns_the_pages_tests_red(guard, node, tmp_path):
    """Delete the guard, run the tests it names against the mutant, require failure.

    A new guard marked on the page is covered here without a test of its own, which is the
    point: the mechanism is the habit, not a list. Red is evidence only when a test actually
    failed -- `guardlib.outcome` reads pytest's short summary to tell that from a child that
    errored, which is the same decision the other two harnesses make with the same code.
    """
    assert guard is not None, "no guards declared on the page"
    text = PAGE.read_text()
    mutant = _without(text, guard)
    parsed = _parses(node, _script(mutant))
    if parsed.returncode != 0:                          # pragma: no cover - marker error
        pytest.fail(f"removing guard {guard.name!r} leaves script that will not parse "
                    f"({parsed.stderr.strip()}). Widen the markers so the block stands "
                    f"alone -- as written, a red run would prove nothing.")

    got = outcome(_run_module_against(_mutant_site(tmp_path, mutant), guard.selectors))
    if got.errors and not got.failed:                   # pragma: no cover - harness error
        pytest.fail(
            f"removing guard {guard.name!r} made the child red without a test failing, which "
            f"is a harness error and not evidence: {got.why_not_evidence()}.\n{got.stdout}")
    assert got.is_evidence, (
        f"the page's tests still pass with guard {guard.name!r} removed, so nothing proves "
        f"it fires. It guards: {guard.why}\n"
        f"Either the tests assert the outcome rather than that the guard produced it, or the "
        f"guard is dead. ({got.why_not_evidence()})")


# --- the grammar's planted failure (#347) ------------------------------------

@parent_only
def test_a_panel_with_the_wrong_state_turns_the_grammar_red(node, tmp_path):
    """The planted failure. A page that labels a stale panel frozen -- the mislabel the
    grammar exists to stop -- must fail the stale test; so must one that drops the age; and
    an unconfirmed note drawn as a plain warning must fail its own."""
    text = PAGE.read_text()
    stale_test = "test_a_stale_artifact_renders_in_the_stale_state_with_its_age"
    for name, old, new, expected in [
        ("frozen-for-stale", "cls: stale ? STATE.stale : STATE.frozen",
         "cls: STATE.frozen", stale_test),
        ("no-age", ' <span class="stamp">artifact ${esc(age)}</span>', "", stale_test),
        ("plain-warning", '<p class="${STATE.unconfirmed}">These figures',
         '<p class="warn">These figures',
         "test_an_unconfirmed_pool_rule_renders_in_the_unconfirmed_state"),
        ("board-ignores-the-comparison", "const shows = boardShows(man);",
         "const shows = true;", "test_the_board_panel_is_gated_on_the_comparison"),
        ("board-retires-on-the-wrong-side", "Date.parse(a.generated_at) > built",
         "Date.parse(a.generated_at) < built",
         "test_the_board_retires_when_a_slate_is_newer_than_it"),
    ]:
        mutant = text.replace(old, new)
        assert mutant != text, f"{name}: the mutation did not land"
        assert _parses(node, _script(mutant)).returncode == 0
        got = _run_module_against(_mutant_site(tmp_path / name, mutant))
        assert expected in _failed(got), f"{name}: expected {expected}, got {sorted(_failed(got))}"
