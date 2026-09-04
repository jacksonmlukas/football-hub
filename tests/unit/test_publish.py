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
    hit = next(b for b in bins if b["n"])
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


def test_scoring_accepts_numpy_arrays():
    """`if not probs` is ambiguous for a numpy array and raises. It went unnoticed while
    every caller passed lists; the model-comparison harness bootstraps over arrays, and hit
    it 4,000 times a run."""
    import numpy as np
    p = np.array([0.6, 0.3, 0.75])
    y = np.array([1, 0, 1])
    assert publish.log_loss(p, y) == pytest.approx(publish.log_loss(list(p), list(y)))
    assert publish.brier(p, y) == pytest.approx(publish.brier(list(p), list(y)))


def test_scoring_an_empty_set_is_not_an_error():
    import numpy as np
    assert publish.log_loss(np.array([]), np.array([])) != publish.log_loss([0.5], [1])


def test_the_week_defaults_to_the_latest_one_predicted(base, site):
    """`--week` defaulted to 1, so `make slate` with no week would have republished week 1
    every Sunday all season -- and looked like it worked."""
    store.write(_preds([("g1", 0.6, 3.0)], week=1), "preds", "nfl", 2026, 1, base=base)
    store.write(_preds([("g2", 0.7, 4.0)], week=5), "preds", "nfl", 2026, 5, base=base)
    assert publish.default_week(2026, base=base) == 5


def test_the_week_falls_back_to_one_when_nothing_is_predicted_yet(base):
    """Preseason. Falling back beats raising: a weekly refresh that needs an operator dies
    in October, which is this repo's own rule."""
    assert publish.default_week(2026, base=base) == 1


def test_survivor_is_published_now_that_the_solver_exists(site, base, monkeypatch):
    """The manifest hardcoded survivor as absent with "not built yet (foundation-plan 5.1)".
    5.1 shipped, and a panel that says a built thing is missing is worse than no panel."""
    import hub.season.survivor as sv
    monkeypatch.setattr(sv, "grid_from_schedule", lambda season, cache=None: pl.DataFrame(
        {"week": [1, 1, 2, 2], "team": ["KC", "LV", "SF", "SEA"],
         "win_prob": [0.8, 0.2, 0.7, 0.3]}))
    man = publish.publish_all(2026, 1, base=base, out=site)
    art = next(a for a in man["artifacts"] if a["name"] == "survivor")
    assert art["present"] is True and art["stale"] is False
    rows = json.loads((site / "survivor.json").read_text())["rows"]
    assert [r["team"] for r in rows] == ["KC", "SF"]


def test_a_survivor_solve_that_fails_leaves_the_panel_stale_not_broken(site, base, monkeypatch):
    """CLAUDE.md graceful degradation: a failing source marks stale and serves last-good,
    it does not take the page down."""
    import hub.season.survivor as sv

    def _boom(season, cache=None):
        raise RuntimeError("nflverse down")
    monkeypatch.setattr(sv, "grid_from_schedule", _boom)
    man = publish.publish_all(2026, 1, base=base, out=site)
    art = next(a for a in man["artifacts"] if a["name"] == "survivor")
    assert art["stale"] is True and art["reason"]


def test_the_survivor_artifact_carries_survival_and_unpriced_weeks_at_top_level(
        site, base, monkeypatch):
    """`_artifact` takes **kwargs; passing `extra={...}` nests them under an "extra" key and
    the page reads undefined. Caught by comparing the published JSON against what the
    dashboard indexes."""
    import hub.season.survivor as sv
    monkeypatch.setattr(sv, "grid_from_schedule", lambda season, cache=None: pl.DataFrame(
        {"week": [1, 1, 3, 3], "team": ["KC", "LV", "SF", "SEA"],
         "win_prob": [0.8, 0.2, 0.7, 0.3]}))
    publish.publish_all(2026, 1, base=base, out=site)
    got = json.loads((site / "survivor.json").read_text())
    assert "extra" not in got
    assert got["survival"] == pytest.approx(0.8 * 0.7)
    # Weeks 1 and 3 are priced; every other week of the season still needs a pick. Asking
    # coverage only about the weeks the grid already has would report none missing.
    assert got["unpriced_weeks"] == [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


# --- a broken query is not "no predictions" --------------------------------

def test_an_empty_store_reports_absence_by_asking_not_by_catching(tmp_path):
    """`store.tables` exists so a caller can tell "this clone has no predictions yet" from
    "the query is broken". A bare `except Exception` made a schema break, a DuckDB lock and a
    typo in the SQL all arrive as `stale: true, reason: no predictions`."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "preds" not in store.tables(empty)
    assert publish.predictions(2026, 1, base=empty, out=tmp_path / "site") is None
    assert publish.default_week(2026, base=empty) == 1


def test_a_real_query_failure_now_surfaces(site, base, monkeypatch):
    """The behaviour that changed. With predictions present, a broken query must raise rather
    than be reported to the page as an absence."""
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    assert "preds" in store.tables(base), "the fixture store has predictions"

    def boom(*a, **k):
        raise RuntimeError("schema drift")
    monkeypatch.setattr(store, "sql", boom)
    with pytest.raises(RuntimeError, match="schema drift"):
        publish.predictions(2026, 1, base=base, out=site)


def test_the_week_partition_has_one_spelling():
    """It had four in one 333-line file -- two filenames, a query parameter and the read-back
    -- because the format belonged to nobody."""
    assert store.week_key(1) == "01"
    assert store.week_key(14) == "14"
    assert store.LAYOUT.format(table="preds", league="nfl", season=2026, week=1,
                               name="x").split("/")[3] == f"week={store.week_key(1)}"


# --- the freshness contract, which every artifact keeps the same way ---

def _names(season=2026, week=1, base=None, out=None):
    return [a.name for a in publish.artifacts(season, week, base=base, out=out)]


def test_every_panel_the_page_reads_is_declared_once():
    """The manifest's order is the page's order, and adding a panel is one list entry."""
    assert _names() == ["preds_wk01", "track_record", "live", "roster",
                        "draft_board", "survivor"]


@pytest.mark.parametrize("name", ["preds_wk01", "track_record", "live", "roster",
                                  "draft_board", "survivor"])
def test_a_producer_returning_none_is_stale_with_a_reason(name, tmp_path):
    """`CLAUDE.md`'s degradation rule, asserted for all six rather than the four that used to
    go through `record`. `draft_board` and `survivor` bypassed it entirely and so could not
    be tested at all."""
    art = next(a for a in publish.artifacts(2026, 1, out=tmp_path) if a.name == name)
    got = publish.Artifact(art.name, lambda: None, art.reason).record(tmp_path)
    assert got["stale"] is True
    assert got["reason"], f"{name} went stale without saying why"
    assert got["present"] is False and got["generated_at"] is None


@pytest.mark.parametrize("name", ["preds_wk01", "roster", "draft_board", "survivor"])
def test_last_goods_timestamp_survives_a_failed_producer(name, tmp_path):
    """The panel shows yesterday's numbers *and* how old they are. `draft_board` and
    `survivor` always reported null here, so the page could not age them."""
    (tmp_path / f"{name}.json").write_text('{"generated_at": "2026-09-01T00:00:00+00:00"}')
    got = publish.Artifact(name, lambda: None, "why").record(tmp_path)
    assert got["present"] is True
    assert got["generated_at"] == "2026-09-01T00:00:00+00:00"


def test_a_list_shaped_artifact_does_not_crash_the_manifest(tmp_path):
    """`draft_board.json` is a bare list of rows, so a `.get` on it raises. That branch was
    unreachable while draft_board bypassed the contract; it is reachable now."""
    (tmp_path / "draft_board.json").write_text('[{"player": "x"}]')
    got = publish.Artifact("draft_board", lambda: None, "run `make draft`").record(tmp_path)
    assert got["present"] is True and got["generated_at"] is None


def test_a_present_board_is_not_stale_and_an_absent_one_is(tmp_path):
    board = next(a for a in publish.artifacts(2026, 1, out=tmp_path) if a.name == "draft_board")
    assert board.record(tmp_path)["stale"] is True
    (tmp_path / "draft_board.json").write_text('[{"player": "x"}]')
    assert board.record(tmp_path)["stale"] is False


# --- one prediction per game, not one per fitted version --------------------

def test_a_week_with_several_fitted_versions_publishes_each_game_once(site, base):
    """On the live store 2026 week 1 held five versions of the same sixteen games, and this
    page listed every one of them five times -- a reader counting games on it would have
    found eighty."""
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base, name="v1")
    later = _preds([("g1", 0.9, 9.0)]).with_columns(
        pl.lit("v2").alias("version"), pl.lit(dt.datetime(2026, 9, 3)).alias("predicted_at"))
    store.write(later, "preds", "nfl", 2026, 1, base=base, name="v2")

    got = publish.predictions(2026, 1, base=base, out=site)
    assert got is not None
    assert got["n"] == 1
    assert got["rows"][0]["version"] == "v2", "the page shows what the model now says"


def test_the_track_record_scores_each_game_once(site, base, monkeypatch):
    """Pooling versions counts every prediction as many times as it was re-fitted, and the
    reliability curve reports the inflated count as its sample size."""
    import nflreadpy as nfl
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base, name="v1")
    later = _preds([("g1", 0.6, 3.0)]).with_columns(
        pl.lit("v2").alias("version"), pl.lit(dt.datetime(2026, 9, 3)).alias("predicted_at"))
    store.write(later, "preds", "nfl", 2026, 1, base=base, name="v2")
    monkeypatch.setattr(nfl, "load_schedules", lambda: pl.DataFrame(
        {"game_id": ["g1"], "result": [7]}))
    assert publish.track_record(base=base, out=site)["n_scored"] == 1


# --- what a reader of the published record can check ------------------------

def _priced(rows, source="snapshot", week=1):
    return _preds(rows, week=week).with_columns(pl.lit(source).alias("price_source"))


def test_the_weekly_artifact_says_what_a_reader_can_obtain(site, base):
    """`priced_at` names a snapshot on one machine. A public record has to say that."""
    store.write(_priced([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    got = publish.predictions(2026, 1, base=base, out=site)
    assert got is not None
    prov = got["provenance"]
    assert set(prov) == {"snapshot"}
    assert prov["snapshot"]["reader_can_obtain"] is False
    assert ".gitignore" in prov["snapshot"]["why"]


def test_a_mixed_slate_classifies_every_source_it_used(site, base):
    """Near weeks price from a snapshot and far ones can fall back, so one artifact carries
    both -- and a reader needs the reason for each, not for whichever came first."""
    store.write(_priced([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base, name="a")
    store.write(_priced([("g2", 0.4, -1.0)], source="schedule"), "preds", "nfl", 2026, 1,
                base=base, name="b")
    got = publish.predictions(2026, 1, base=base, out=site)
    assert got is not None
    assert set(got["provenance"]) == {"snapshot", "schedule"}


def test_nothing_a_prediction_already_carried_is_removed(site, base):
    """This adds a qualification. A record that dropped provenance to look tidier would be
    the opposite of the point."""
    store.write(_priced([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    got = publish.predictions(2026, 1, base=base, out=site)
    assert got is not None
    row = got["rows"][0]
    for field in ("game_id", "home_win_prob", "margin_mean", "model", "version",
                  "predicted_at", "price_source"):
        assert field in row


def test_an_artifact_from_before_price_source_existed_still_publishes(site, base):
    """The store spans schemas by design. A week written before #6 has no source to classify
    and must not take the page down."""
    store.write(_preds([("g1", 0.6, 3.0)]), "preds", "nfl", 2026, 1, base=base)
    got = publish.predictions(2026, 1, base=base, out=site)
    assert got is not None and got["provenance"] == {}
