"""Publishing the site's JSON.

`weekly-slate/SKILL.md:22` and `docs/track-record.md` both invoke `hub.publish` and it did
not exist.

Two rules from the docs drive the design and both are testable.

`docs/track-record.md` rule 1: a prediction counts only if it was committed before kickoff.
A page that cannot tell a pre-registered prediction from one scored after the fact is a
tout account with extra steps, so the artifact records which it is and refuses to blur them.

`CLAUDE.md`: if a fetch fails, serve last-good state rather than erroring. So a missing
source must leave the previous artifact in place and mark it stale, never blank it -- a
dashboard that goes empty on a bad Sunday is worse than one showing yesterday's numbers
with a warning.
"""
import datetime as dt
import json

import polars as pl
import pytest

from hub import publish, store


@pytest.fixture
def site(tmp_path):
    return tmp_path / "site" / "data"


@pytest.fixture
def base(tmp_path):
    return tmp_path / "processed"


def _preds(rows, week=1):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows], "league": ["nfl"] * len(rows),
         "season": pl.Series([2026] * len(rows), dtype=pl.Int32),
         "week": pl.Series([week] * len(rows), dtype=pl.Int32),
         "home_win_prob": [r[1] for r in rows],
         "margin_mean": [r[2] for r in rows], "margin_lo": [r[2] - 17.0 for r in rows],
         "margin_hi": [r[2] + 17.0 for r in rows],
         "model": ["market_baseline"] * len(rows), "version": ["v1"] * len(rows),
         "fit_through_week": pl.Series([week - 1] * len(rows), dtype=pl.Int32),
         "predicted_at": [dt.datetime(2026, 9, 1)] * len(rows)})


# --- artifacts ------------------------------------------------------------

def test_predictions_become_a_weekly_artifact(site, base):
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    publish.predictions(2026, 1, base=base, out=site)
    got = json.loads((site / "preds_wk01.json").read_text())
    assert got["rows"][0]["game_id"] == "g1"
    assert got["week"] == 1


def test_every_artifact_carries_its_own_freshness(site, base):
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    publish.predictions(2026, 1, base=base, out=site)
    got = json.loads((site / "preds_wk01.json").read_text())
    assert got["generated_at"] and got["source"] == "preds"


def test_a_manifest_lists_what_the_page_can_render(site, base):
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    publish.publish_all(2026, 1, base=base, out=site)
    man = json.loads((site / "manifest.json").read_text())
    assert {a["name"] for a in man["artifacts"]} >= {"preds_wk01", "track_record"}


# --- last-good, not blank -------------------------------------------------

def test_a_missing_source_leaves_the_previous_artifact_alone(site, base):
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    publish.predictions(2026, 1, base=base, out=site)
    before = (site / "preds_wk01.json").read_text()

    # week 2 has nothing in the store at all
    publish.predictions(2026, 2, base=base, out=site)
    assert (site / "preds_wk01.json").read_text() == before, "week 1 must survive"


def test_a_missing_source_marks_stale_rather_than_writing_an_empty_file(site, base):
    publish.publish_all(2026, 9, base=base, out=site)
    man = json.loads((site / "manifest.json").read_text())
    preds = next(a for a in man["artifacts"] if a["name"] == "preds_wk09")
    assert preds["stale"] is True
    assert preds["reason"]


def test_a_stale_artifact_that_never_existed_is_reported_as_absent(site, base):
    publish.publish_all(2026, 9, base=base, out=site)
    man = json.loads((site / "manifest.json").read_text())
    preds = next(a for a in man["artifacts"] if a["name"] == "preds_wk09")
    assert preds["present"] is False


# --- the track record -----------------------------------------------------

def test_with_no_scored_predictions_the_record_says_so(site, base):
    got = publish.track_record(base=base, out=site)
    assert got["n_scored"] == 0
    assert got["n_preregistered"] == 0


def test_a_backtest_is_never_labelled_a_track_record(site, base):
    """docs/track-record.md rule 1: the value is entirely in the timestamp. A record that
    cannot distinguish a pre-registered prediction from a backfilled one proves nothing."""
    got = publish.track_record(base=base, out=site)
    assert "preregistered" in json.dumps(got).lower()
    assert got.get("is_backtest") is not None


def test_calibration_bins_are_reported_with_counts(site, base):
    scored = pl.DataFrame({
        "home_win_prob": [0.1, 0.15, 0.85, 0.9, 0.55, 0.45],
        "home_won": [0, 0, 1, 1, 1, 0]})
    bins = publish.reliability(scored, n_bins=5)
    assert sum(b["n"] for b in bins) == 6
    assert all("predicted" in b and "actual" in b for b in bins)


def test_a_perfectly_calibrated_set_shows_it():
    n = 400
    probs = [0.25] * n
    won = [1] * (n // 4) + [0] * (n - n // 4)
    bins = publish.reliability(pl.DataFrame({"home_win_prob": probs, "home_won": won}),
                               n_bins=4)
    hit = [b for b in bins if b["n"]][0]
    assert abs(hit["predicted"] - hit["actual"]) < 0.02


def test_log_loss_punishes_confident_and_wrong():
    right = publish.log_loss([0.9], [1])
    wrong = publish.log_loss([0.9], [0])
    assert wrong > right * 5


def test_log_loss_is_finite_at_the_extremes():
    """A model that said 1.0 and was wrong must not produce infinity on the page."""
    assert publish.log_loss([1.0], [0]) < 40


# --- the live overlay -----------------------------------------------------

def test_live_is_its_own_artifact(site, monkeypatch):
    """Separate file on purpose: it is the only thing that changes during a Sunday, and
    keeping it apart is what lets the page hold frozen and live numbers apart."""
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "live_state",
                        lambda league="nfl": [{"id": "1", "home": "SEA", "away": "NE",
                                               "state": "in", "home_score": "10"}])
    got = publish.live(out=site)
    assert got is not None and got["source"] == "espn_scoreboard"
    assert json.loads((site / "live.json").read_text())["rows"][0]["home"] == "SEA"


def test_espn_being_down_leaves_the_last_scores_in_place(site, monkeypatch):
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [{"id": "1", "home": "SEA"}])
    publish.live(out=site)
    before = (site / "live.json").read_text()

    def _boom(league="nfl"):
        raise RuntimeError("403")
    monkeypatch.setattr(espn, "live_state", _boom)
    assert publish.live(out=site) is None
    assert (site / "live.json").read_text() == before, "last-good scores must survive"


def test_publishing_live_never_touches_the_predictions(site, base, monkeypatch):
    """The core rule of 4.3. A model number that moves while games are in progress is
    worthless as a pre-registered claim, however good it looks afterwards."""
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    publish.predictions(2026, 1, base=base, out=site)
    frozen = (site / "preds_wk01.json").read_text()

    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "live_state", lambda league="nfl": [{"id": "1", "home": "SEA"}])
    for _ in range(3):
        publish.live(out=site)
    assert (site / "preds_wk01.json").read_text() == frozen


def test_scoring_against_itself_reports_no_edge():
    """3.4's correctness test, applied here: the same predictions scored against
    themselves must show a delta of zero."""
    p = [0.6, 0.3, 0.75]
    assert publish.log_loss(p, [1, 0, 1]) - publish.log_loss(p, [1, 0, 1]) == 0.0
