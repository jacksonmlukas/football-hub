"""An unfillable lineup publishes the gap, not last week's call (#531).

Found 2026-10-21: the week-7 roster publish declined because withholding (two byes and a
Questionable tight end) left no eligible TE. `lock` returned an empty Lock by design (#419),
`degenerate` marked the roster stale, and the site kept week 6's roster.json -- which started a TE
who is on bye this week. A gap is now published as this week's best partial lineup, the empty
slot named, and the best free agents at that position who are not on bye, labelled incomplete.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hub import publish
from hub.season import lineup_rule, stream
from hub.season.roster import lock, partial

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "roster_531_fillable.json"

# A fixed mtime for the parquet, so `generated_at` (dated from the file, #122) is a constant.
MTIME = 1_792_000_000           # 2026-10-15T...Z


def _frame(rows, **cols) -> pl.DataFrame:
    """`(player, pos, team, mu, starting)` rows, healthy unless a column says otherwise."""
    n = len(rows)
    base = {"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
            "nfl_team": [r[2] for r in rows], "mu": [r[3] for r in rows],
            "sd": [5.0] * n, "projected": [True] * n, "starting": [r[4] for r in rows],
            "injury_status": ["ACTIVE"] * n, "available": [True] * n,
            "can_start": [True] * n, "missing_games": [0] * n, "on_bye": [False] * n}
    for k, v in cols.items():
        base[k] = v
    return pl.DataFrame(base, schema_overrides={"missing_games": pl.Int64})


# QB1 / RB2 / WR3 / TE1 / FLEX1, with a bench. The set lineup starts "Bench WR" over "Good WR",
# so there is one swap to price, and "Bye WR" is on bye, withheld with the reason.
FILLABLE = [("Qb One", "QB", "KC", 18.0, True), ("Rb One", "RB", "SF", 15.0, True),
            ("Rb Two", "RB", "DET", 12.0, True), ("Wr One", "WR", "CIN", 16.0, True),
            ("Wr Two", "WR", "MIA", 13.0, True), ("Bench WR", "WR", "NYJ", 6.0, True),
            ("Te One", "TE", "DAL", 9.0, True), ("Flex RB", "RB", "LAR", 10.0, True),
            ("Good WR", "WR", "SEA", 11.0, False), ("Bye WR", "WR", "JAX", 14.0, False)]

# The week-7 shape: the only two tight ends are withheld -- one on bye, one projected short --
# and the bye one is set as the starter, which is the call last week's file would publish.
NO_TE = [("Qb One", "QB", "KC", 18.0, True), ("Rb One", "RB", "SF", 15.0, True),
         ("Rb Two", "RB", "DET", 12.0, True), ("Wr One", "WR", "CIN", 16.0, True),
         ("Wr Two", "WR", "MIA", 13.0, True), ("Wr Three", "WR", "NYG", 11.0, True),
         ("Bye TE", "TE", "JAX", 9.0, True), ("Flex RB", "RB", "LAR", 10.0, True),
         ("Short TE", "TE", "ATL", 7.0, False), ("Bench RB", "RB", "TEN", 5.0, False)]


def _no_te() -> pl.DataFrame:
    return _frame(NO_TE, on_bye=[r[0] == "Bye TE" for r in NO_TE],
                  available=[r[0] != "Short TE" for r in NO_TE],
                  missing_games=[1 if r[0] == "Short TE" else 0 for r in NO_TE],
                  injury_status=["QUESTIONABLE" if r[0] == "Short TE" else "ACTIVE"
                                 for r in NO_TE])


def _fillable() -> pl.DataFrame:
    return _frame(FILLABLE, on_bye=[r[0] == "Bye WR" for r in FILLABLE])


def _fa(player, team, pg, *, on_bye=False, blocked=None, pos="TE", rank=1):
    return {"player": player, "pos": pos, "nfl_team": team, "injury_status": "",
            "percent_owned": 12.5, "proj": pg * 11, "proj_per_game": pg, "value": pg - 7.0,
            "replaces": "Short TE", "on_bye_now": on_bye, "blocked": blocked, "rank": rank,
            "recommended": False}


WAIVER = {"label": "provisional", "week": 7, "as_of": "2026-10-20T12:00:00+00:00",
          "last_error": None, "stale": None, "calendar_reason": None, "missing": None,
          "rows": [_fa("Fa Bye", "BUF", 9.5, on_bye=True, rank=1),
                   _fa("Fa One", "KC", 8.0, rank=2), _fa("Fa Out", "DEN", 7.9,
                                                         blocked="OUT", rank=3),
                   _fa("Fa Two", "DET", 7.5, rank=4), _fa("Fa Three", "NYJ", 7.1, rank=5),
                   _fa("Fa Four", "CHI", 6.9, rank=6),
                   _fa("Fa Wr", "SEA", 12.0, pos="WR", rank=7)],
          "recommended": []}


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    """The K / D/ST heuristic reads the schedule and the waiver block the free-agent parquet;
    neither is under test here, and both are pinned so the bytes are."""
    monkeypatch.setattr(publish, "_roster_stream", lambda df: stream.unavailable("stubbed"))
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: json.loads(json.dumps(WAIVER)))


def _write(tmp_path: Path, df: pl.DataFrame, name: str = "roster.parquet") -> Path:
    src = tmp_path / name
    df.write_parquet(src)
    os.utime(src, (MTIME, MTIME))
    return src


def _publish(tmp_path: Path, df: pl.DataFrame, site: str = "site") -> tuple[Any, Path]:
    out = tmp_path / site
    out.mkdir(parents=True, exist_ok=True)
    got = publish.roster(out=out, path=_write(tmp_path, df))
    return got, out / "roster.json"


# --- a fillable week is byte-identical to before #531 --------------------------------------

def test_a_fillable_week_publishes_the_bytes_it_did_before_the_gap_existed(tmp_path):
    """The fixture was written by the publisher as it stood before #531 (commit 0449d2d), from
    this exact roster, stubs and mtime. A fillable week must not move a byte."""
    got, path = _publish(tmp_path, _fillable())
    assert isinstance(got, dict) and got["gain"] is not None
    assert path.read_text() == FIXTURE.read_text()


def test_a_fillable_week_never_asks_for_a_partial_lineup(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a fillable week reached the gap path")
    monkeypatch.setattr(lineup_rule, "partial_call", boom)
    got, _ = _publish(tmp_path, _fillable())
    assert isinstance(got, dict) and "incomplete" not in got
    entry = publish.Artifact("roster", lambda: got, "no roster yet").record(tmp_path / "site")
    assert "incomplete" not in entry and entry["stale"] is False


# --- the partial lineup ----------------------------------------------------------------------

NO_TE_LINEUP = ["Flex RB", "Qb One", "Rb One", "Rb Two", "Wr One", "Wr Three", "Wr Two"]


def test_the_week_seven_shape_declines_the_full_lineup():
    """The control the rest stand on: `lock` declines this roster, as it did on 2026-10-21."""
    lk = lock(_no_te())
    assert lk.gain is None and lk.best_lineup == []
    assert set(lk.withheld) == {"Bye TE", "Short TE"}


def test_partial_fills_every_slot_it_can_and_names_the_empty_one():
    got = partial(_no_te())
    assert got.empty == ["TE"]
    assert got.lineup == NO_TE_LINEUP
    assert "Bye TE" not in got.lineup and "Short TE" not in got.lineup


def test_partial_names_an_unfillable_flex_and_a_short_position():
    """One WR on the roster: two WR slots and the flex are empty, and the rest is filled."""
    rows = [("Qb", "QB", "KC", 18.0, True), ("Rb1", "RB", "SF", 15.0, True),
            ("Rb2", "RB", "DET", 12.0, True), ("Wr", "WR", "CIN", 16.0, True),
            ("Te", "TE", "DAL", 9.0, True)]
    got = partial(_frame(rows))
    assert got.empty == ["WR", "WR", "FLEX"]
    assert got.lineup == ["Qb", "Rb1", "Rb2", "Te", "Wr"]


def test_partial_of_nobody_is_all_empty_and_never_raises():
    got = partial(_frame([("Hurt", "QB", "KC", 18.0, True)], can_start=[False]))
    assert got.lineup == [] and got.empty[0] == "QB" and "FLEX" in got.empty


def test_partial_keeps_a_started_starter_and_drops_a_started_bench_player():
    """#516: a game that has kicked off closes the call in the partial lineup as in `lock`."""
    df = _no_te().with_columns(
        pl.col("player").is_in(["Wr Three", "Bench RB"]).alias("kicked_off"),
        pl.when(pl.col("player") == "Wr Three").then(1.0).otherwise(pl.col("mu")).alias("mu"))
    got = partial(df)
    assert "Wr Three" in got.lineup, "a started starter stays where he is, whatever he projects"
    assert "Bench RB" not in got.lineup


def test_partial_under_ranks_chooses_by_rank():
    ranks = {_key(n): float(i) for i, n in enumerate(
        ["Qb One", "Rb One", "Rb Two", "Wr One", "Wr Two", "Wr Three", "Bench RB", "Flex RB",
         "Bye TE", "Short TE"], start=1)}
    got = partial(_no_te(), ranks=ranks)
    assert got.empty == ["TE"] and "Bench RB" in got.lineup and "Flex RB" not in got.lineup


def _key(name: str) -> str:
    from hub.names import player_key
    return player_key(name)


def test_partial_call_sets_by_consensus_from_the_switch_week(tmp_path):
    ranks = {_key(n): float(i) for i, n in enumerate(
        ["Qb One", "Rb One", "Rb Two", "Wr One", "Wr Two", "Wr Three", "Bench RB"], start=1)}
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path, ecr=ranks)
    assert got.source == "ecr" and got.as_dict()["label"] == "provisional"
    assert "Bench RB" in got.partial.lineup and got.reason is None


def test_partial_call_falls_back_to_mu_when_ranks_leave_more_slots_empty(tmp_path):
    """A ranking with no quarterback in it would add a gap the roster does not have."""
    ranks = {_key(n): float(i) for i, n in enumerate(
        ["Rb One", "Rb Two", "Wr One", "Wr Two", "Wr Three", "Bench RB"], start=1)}
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path, ecr=ranks)
    assert got.source == "mu" and got.partial.lineup == NO_TE_LINEUP
    assert got.reason and "more slots empty" in got.reason


def test_partial_call_before_the_switch_is_mu_with_no_reason(tmp_path):
    got = lineup_rule.partial_call(_no_te(), 2026, 6, captures=tmp_path)
    assert got.source == "mu" and got.reason is None and got.as_dict()["label"] == "model"


def test_partial_call_with_no_ranks_at_all_says_so(tmp_path):
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path, ecr={},
                                   ecr_reason="no capture for week 7")
    assert got.source == "mu" and got.reason and got.reason.startswith("no capture for week 7")


# --- the published gap ----------------------------------------------------------------------

def test_a_roster_with_no_eligible_te_publishes_the_partial_lineup_the_gap_and_free_agents(
        tmp_path):
    got, path = _publish(tmp_path, _no_te())
    assert isinstance(got, dict), "a gap is published, not kept"
    gap = got["incomplete"]
    assert gap["empty_slots"] == ["TE"]
    assert gap["lineup"] == NO_TE_LINEUP
    # Not on bye (Fa Bye), not unable to play (Fa Out), best per game first, the top three.
    assert [r["player"] for r in gap["free_agents"]["TE"]] == ["Fa One", "Fa Two", "Fa Three"]
    assert gap["headline"].startswith("TE: no eligible player; add one -- best free agents at "
                                      "TE not on bye: Fa One (KC), Fa Two (DET), Fa Three (NYJ)")
    assert gap["free_agents_note"] is None
    assert gap["lineup_source"]["source"] == "mu"
    assert "no lineup could be filled" in gap["reason"]
    # This week's starters, never last week's: the bye TE set as a starter is sat, unticked.
    ticked = sorted(r["player"] for r in got["rows"] if r["best_start"])
    assert ticked == NO_TE_LINEUP
    assert "Bye TE" in got["sit"] and got["gain"] is None
    assert got["withheld_reasons"]["Bye TE"] == "bye"
    assert json.loads(path.read_text())["incomplete"] == gap


def test_the_gap_is_current_and_incomplete_in_the_manifest_never_stale(tmp_path):
    got, _ = _publish(tmp_path, _no_te())
    entry = publish.Artifact("roster", lambda: got, "no roster yet").record(tmp_path / "site")
    assert entry["stale"] is False and entry["reason"] is None
    assert entry["incomplete"].startswith("TE: no eligible player")


def test_a_gap_replaces_last_weeks_file_rather_than_keeping_it(tmp_path):
    """The planted control (#531). Last week's call is on disk, starting the TE now on bye. A
    gap must overwrite it with this week's partial lineup; keeping it is the bug."""
    site = tmp_path / "site"
    site.mkdir()
    last_week = {"name": "roster", "generated_at": "2026-10-17T16:58:11+00:00", "week": 6,
                 "gain": 0.0, "start": [], "sit": [], "withheld": [],
                 "rows": [{"player": "Bye TE", "pos": "TE", "best_start": True}]}
    (site / "roster.json").write_text(json.dumps(last_week))
    got = publish.roster(out=site, path=_write(tmp_path, _no_te()))
    assert not isinstance(got, publish.Kept), "last week's file was kept"
    on_disk = json.loads((site / "roster.json").read_text())
    assert on_disk.get("incomplete", {}).get("empty_slots") == ["TE"]
    assert not [r for r in on_disk["rows"] if r["player"] == "Bye TE" and r["best_start"]]
    assert on_disk["previous"]["generated_at"] == last_week["generated_at"]


def test_a_majority_withheld_is_still_stale_not_a_gap(tmp_path):
    """#419's guard is untouched: a misread of availability is not published as a lineup."""
    df = _no_te().with_columns(
        (~pl.col("player").is_in(["Qb One", "Rb One", "Bench RB"])).alias("on_bye"))
    got, _ = _publish(tmp_path, df)
    assert isinstance(got, publish.Kept) and "no lineup" in got.why


def test_a_gap_reads_free_agents_from_a_last_good_pool_and_says_so(tmp_path, monkeypatch):
    failed = dict(WAIVER, last_error="ConnectionError: the latest ESPN free-agent fetch failed")
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: json.loads(json.dumps(failed)))
    got, _ = _publish(tmp_path, _no_te())
    gap = got["incomplete"]
    assert [r["player"] for r in gap["free_agents"]["TE"]] == ["Fa One", "Fa Two", "Fa Three"]
    assert "last good pool" in gap["free_agents_note"]
    assert "2026-10-20T12:00:00+00:00" in gap["free_agents_note"]


def test_a_gap_with_no_free_agent_pool_still_publishes_and_says_why(tmp_path, monkeypatch):
    from hub.season import waiver
    monkeypatch.setattr(publish, "_roster_waiver",
                        lambda df: waiver.unavailable("no free agents fetched yet"))
    got, _ = _publish(tmp_path, _no_te())
    gap = got["incomplete"]
    assert gap["free_agents"] == {"TE": []}
    assert "no free agent at TE" in gap["headline"]
    assert gap["free_agents_note"].startswith("no free agents to suggest")


def test_a_gap_notes_byes_it_could_not_check(tmp_path, monkeypatch):
    blind = dict(WAIVER, calendar_reason="the schedule was unavailable (OSError)")
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: json.loads(json.dumps(blind)))
    got, _ = _publish(tmp_path, _no_te())
    assert "byes are not checked" in got["incomplete"]["free_agents_note"]


def test_a_flex_gap_draws_free_agents_from_every_flex_position():
    from hub.league import FLEX_FROM
    from hub.season.waiver import fill_candidates
    got, note = fill_candidates(WAIVER, ["FLEX"], FLEX_FROM)
    assert [r["player"] for r in got["FLEX"]] == ["Fa Wr", "Fa One", "Fa Two"]
    assert note is None


def test_a_gap_keeps_a_started_starter_out_of_the_moves(tmp_path):
    df = _no_te().with_columns((pl.col("player") == "Wr Three").alias("kicked_off"))
    got, _ = _publish(tmp_path, df)
    assert got["locked"] == ["Wr Three"]
    assert "Wr Three" in got["incomplete"]["lineup"]
    assert "Wr Three" not in got["start"] and "Wr Three" not in got["sit"]


def test_a_gap_whose_fields_are_refused_is_stale_with_the_refusal(tmp_path, monkeypatch):
    from hub import contracts
    from hub.contracts import ContractViolation

    def refuse(payload):
        if "incomplete" in payload:
            raise ContractViolation("planted")
    monkeypatch.setattr(contracts, "check_roster_artifact", refuse)
    got, _ = _publish(tmp_path, _no_te())
    assert isinstance(got, publish.Kept) and "roster fields refused: planted" in got.why


def test_a_gap_is_not_journalled_as_a_published_call(tmp_path, monkeypatch):
    """The journal records the decline as it always has: no `published` block for a gap."""
    seen = []
    monkeypatch.setattr(publish, "_roster_journal", lambda *a: seen.append(a) or "x")
    monkeypatch.setattr(publish, "SITE", tmp_path / "site")
    (tmp_path / "site").mkdir()
    got = publish.roster(path=_write(tmp_path, _no_te()))
    assert isinstance(got, dict) and "incomplete" in got
    assert seen == []


# --- the contract on the new fields ---------------------------------------------------------

def _gap_payload(tmp_path) -> dict:
    got, _ = _publish(tmp_path, _no_te())
    assert isinstance(got, dict)
    return json.loads(json.dumps(got))


def test_the_published_gap_passes_its_contract(tmp_path):
    from hub.contracts import check_roster_artifact
    check_roster_artifact(_gap_payload(tmp_path))


@pytest.mark.parametrize(("change", "says"), [
    (lambda p: p["incomplete"].update(empty_slots=[]), "no empty slots"),
    (lambda p: p["incomplete"].pop("lineup"), "no empty slots or no lineup"),
    (lambda p: p["incomplete"]["lineup"].append("Bye TE"), "withheld or unrostered"),
    (lambda p: p["incomplete"]["lineup"].append("Ghost"), "withheld or unrostered"),
    (lambda p: p["incomplete"].update(headline="Add a player."), "every empty slot"),
    (lambda p: p.update(gain=3.0), "full lineup's gain"),
    (lambda p: [r.update(best_start=True) for r in p["rows"] if r["player"] == "Short TE"],
     "best-XI ticks"),
])
def test_a_malformed_gap_is_refused(tmp_path, change, says):
    from hub.contracts import ContractViolation, check_roster_artifact
    payload = _gap_payload(tmp_path)
    change(payload)
    with pytest.raises(ContractViolation, match=says):
        check_roster_artifact(payload)


# --- the source order's later rungs, and the degradations ------------------------------------

_RANKED = ["Qb One", "Rb One", "Rb Two", "Wr One", "Wr Two", "Wr Three", "Bench RB"]
_LABEL = {"purpose": "lineup", "scrape_date": "2026-10-23",
          "captured_at": "2026-10-23T15:00:00+00:00", "note": "taken after Thursday's games"}


def _ranks() -> dict[str, float]:
    return {_key(n): float(i) for i, n in enumerate(_RANKED, start=1)}


def test_partial_call_reads_the_weeks_ranks_when_none_are_handed_over(tmp_path, monkeypatch):
    from hub.season import lineup_journal
    monkeypatch.setattr(lineup_journal, "ecr_ranks", lambda *a, **k: (_ranks(), None))
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path)
    assert got.source == "ecr" and got.ecr_week == 7 and "Bench RB" in got.partial.lineup


def test_partial_call_takes_the_lineup_capture_and_labels_it(tmp_path, monkeypatch):
    monkeypatch.setattr(lineup_rule, "lineup_ecr", lambda *a, **k: (_ranks(), dict(_LABEL)))
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path, ecr={})
    assert got.source == "ecr" and got.ecr_capture == _LABEL
    assert got.as_dict()["ecr_capture"] == _LABEL


def test_partial_call_serves_last_good_ranks_and_says_which_week(tmp_path, monkeypatch):
    monkeypatch.setattr(lineup_rule, "lineup_ecr", lambda *a, **k: ({}, None))
    monkeypatch.setattr(lineup_rule, "last_good_ecr", lambda *a, **k: (_ranks(), 6))
    got = lineup_rule.partial_call(_no_te(), 2026, 7, captures=tmp_path, ecr={})
    assert got.source == "ecr" and got.ecr_week == 6
    assert got.reason == "set on week 6's capture (last-good)"


def test_a_gap_set_on_the_lineup_capture_shows_those_ranks(tmp_path, monkeypatch):
    from hub.season import lineup_journal
    monkeypatch.setattr(lineup_journal, "roster_week", lambda df: 7)
    monkeypatch.setattr(lineup_journal, "ecr_ranks", lambda *a, **k: ({}, "no week capture"))
    monkeypatch.setattr(lineup_rule, "lineup_ecr", lambda *a, **k: (_ranks(), dict(_LABEL)))
    got, _ = _publish(tmp_path, _no_te())
    assert got["incomplete"]["lineup_source"]["ecr_capture"] == _LABEL
    assert got["ecr_reason"] is None
    assert {r["player"]: r["ecr"] for r in got["rows"]}["Bench RB"] == 7.0


def test_partial_never_raises_when_the_search_refuses(monkeypatch):
    from hub.season import lineup

    def refuse(*a, **k):
        raise lineup.TooManyLineups("planted")
    monkeypatch.setattr(lineup, "best_by_points", refuse)
    assert partial(_no_te()) == ([], ["TE"])


def test_a_stale_pool_is_noted_beside_the_free_agents():
    from hub.league import FLEX_FROM
    from hub.season.waiver import fill_candidates
    _, note = fill_candidates(dict(WAIVER, stale="the pool is 6 days old"), ["TE"], FLEX_FROM)
    assert note == "the pool is 6 days old"
