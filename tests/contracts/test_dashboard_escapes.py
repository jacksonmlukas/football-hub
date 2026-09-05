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
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "site" / "index.html"

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
               "held", "gone", "start", "sit")


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
    ])
    return _run(node, f"{lifted}\nconsole.log(trackRecordBody({json.dumps(tr)}));")


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
    one, and the panel says why."""
    text = PAGE.read_text()
    joined = re.search(r"const live = [^\n]*", text)
    assert joined, "the slate no longer loads the live overlay where the test expects"
    assert "viewing" in joined.group(0), (
        f"{joined.group(0)!r} joins scores to whatever week is on screen; guard it on the "
        f"week the manifest names.")


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
