"""The weeks-out margin sd comparison (#249): the pre-registered design, held in code.

Every frame here is planted. No test reads the archive, a schedule or a result of the real
season, and the clock is never read: kickoffs and captures are fixed datetimes.
"""
from __future__ import annotations

import ast
import inspect
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.ledger import Ledger
from hub.models import lookahead_margin as LM
from hub.models.market import MARGIN_SD

ROOT = Path(__file__).resolve().parents[2]
KICK = datetime(2026, 11, 1, 18, 0)


def planted_rows(sds: dict[int, float], *, weeks: int = 40, games: int = 30, seed: int = 0,
                 buckets: tuple[int, ...] = (1, 2, 3)) -> pl.DataFrame:
    """Scored (game, bucket) rows whose margin is the spread plus noise of the bucket's own sd."""
    rng = np.random.default_rng(seed)
    out = []
    for wk in range(1, weeks + 1):
        for g in range(games):
            spread = float(np.round(rng.normal(0, 6.0) * 2) / 2)
            for b in buckets:
                result = spread + float(rng.normal(0, sds[b]))
                if result == 0:
                    result = 0.5
                out.append({"season": 2026, "week": wk, "game_id": f"{wk}-{g}", "bucket": b,
                            "spread": spread, "result": result, "home_won": int(result > 0),
                            "resid": result - spread})
    return pl.DataFrame(out)


GROWING = {0: MARGIN_SD, 1: 16.0, 2: 19.0, 3: 22.0}
FLAT = dict.fromkeys(range(4), MARGIN_SD)


def quiet(paired: pl.DataFrame):
    return LM.gate_run(paired, ledger=Ledger(path=None))


# --- the pre-registered constants -----------------------------------------------------------

def test_the_edges_are_the_ones_the_preregistration_states():
    doc = (ROOT / "docs" / "lookahead-margin-sd.md").read_text()
    stated = re.search(r"edges (\d+(?:, \d+)*)\.", doc)
    assert stated, "docs/lookahead-margin-sd.md no longer states the bucket edges"
    assert tuple(float(x) for x in stated.group(1).split(",")) == LM.BUCKET_EDGES


def test_no_function_takes_the_edges_as_an_argument():
    """Control 3: the edges are a constant, so they cannot be tuned after the data is seen."""
    tree = ast.parse(inspect.getsource(LM))
    names = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             for a in n.args.args + n.args.kwonlyargs}
    assert not {n for n in names if re.search(r"(^|_)edges?($|_)", n.lower())}


@pytest.mark.parametrize("w,expected", [(0.5, 0), (0.999, 0), (1.0, 1), (1.99, 1), (2.0, 2),
                                        (3.99, 2), (4.0, 3), (7.9, 3), (8.0, 4), (19.0, 4)])
def test_bucket_of_cuts_at_the_edges(w, expected):
    got = pl.DataFrame({"w": [w]}).select(LM.bucket_of(pl.col("w")))
    assert got.item() == expected


def test_label_names_each_range():
    assert [LM.label(b) for b in range(LM.N_BUCKETS)] == [
        "[0, 1)", "[1, 2)", "[2, 4)", "[4, 8)", "[8, inf)"]


# --- the rows -------------------------------------------------------------------------------

def lines_and_games():
    games = pl.DataFrame({
        "game_id": ["a", "b", "c", "d"], "season": [2026] * 4, "week": [5, 5, 5, 5],
        "kickoff": [KICK] * 4, "result": [10.0, 0.0, None, -3.0]})
    snaps = [  # (game, spread, weeks before kickoff)
        ("a", 3.0, 3.5), ("a", 4.0, 2.5), ("a", 5.0, 0.4), ("a", 9.0, -0.1), ("a", 9.0, 0.0),
        ("b", 2.0, 3.0), ("c", 2.0, 3.0), ("d", -1.0, 9.0)]
    lines = pl.DataFrame({
        "game_id": [s[0] for s in snaps], "close_spread": [s[1] for s in snaps],
        "captured_at": [KICK - timedelta(weeks=s[2]) for s in snaps]})
    return lines, games


def test_a_snapshot_at_or_after_kickoff_never_reaches_a_row():
    lines, games = lines_and_games()
    rows = LM.build_rows(lines, games)
    a = rows.filter(pl.col("game_id") == "a")
    assert 9.0 not in a["spread"].to_list()
    assert sorted(a["bucket"].to_list()) == [0, 2]


def test_the_last_snapshot_in_a_bucket_is_the_one_used():
    lines, games = lines_and_games()
    rows = LM.build_rows(lines, games)
    got = rows.filter((pl.col("game_id") == "a") & (pl.col("bucket") == 2))
    assert got["spread"].item() == 4.0 and got["resid"].item() == 6.0


def test_a_tied_and_an_unplayed_game_are_not_scored():
    lines, games = lines_and_games()
    rows = LM.build_rows(lines, games)
    assert set(rows["game_id"]) == {"a", "d"}
    assert rows.filter(pl.col("game_id") == "d")["bucket"].item() == 4


def test_no_snapshots_gives_no_rows():
    _, games = lines_and_games()
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                 "captured_at": pl.Datetime})
    assert LM.build_rows(empty, games).is_empty()


# --- the fit --------------------------------------------------------------------------------

def test_b0_is_the_closing_sd_and_a_thin_bucket_inherits_the_nearer_one():
    past = planted_rows(GROWING, weeks=3, games=15, buckets=(1, 2, 3))  # 45 rows each
    thin = past.filter(~((pl.col("bucket") == 1) & (pl.col("week") > 1)))  # b1: 15 rows
    sds = LM.bucket_sds(thin)
    assert sds[0].sd == MARGIN_SD and sds[0].kind == "closing"
    assert sds[1].kind == "inherited" and sds[1].sd == MARGIN_SD
    assert sds[2].kind == "fitted" and sds[3].kind == "fitted"
    assert sds[4].kind == "inherited" and sds[4].sd == sds[3].sd and sds[4].n == 0


def test_the_inheritance_chains_through_two_thin_buckets_to_the_closing_sd():
    past = planted_rows(GROWING, weeks=1, games=10, buckets=(3,))
    sds = LM.bucket_sds(past)
    assert [sds[b].sd for b in (1, 2, 3, 4)] == [MARGIN_SD] * 4
    assert {sds[b].kind for b in (1, 2, 3, 4)} == {"inherited"}


def test_a_bucket_at_the_floor_is_fitted():
    past = planted_rows(GROWING, weeks=1, games=LM.MIN_FIT_N, buckets=(2,))
    assert LM.bucket_sds(past)[2].kind == "fitted"
    short = past.head(LM.MIN_FIT_N - 1)
    assert LM.bucket_sds(short)[2].kind == "inherited"


# --- the walk-forward -----------------------------------------------------------------------

def test_the_first_week_has_no_past_and_b0_is_never_scored():
    rows = planted_rows(GROWING, weeks=4, games=20, buckets=(0, 1, 2))
    paired = LM.walk_forward(rows)
    assert sorted(paired["week"].unique().to_list()) == [2, 3, 4]
    assert 0 not in paired["bucket"].to_list()


def test_a_week_is_never_fitted_on_itself():
    """Plant a wild residual in the held-out week only: its own sd must not move."""
    base = planted_rows(GROWING, weeks=3, games=20, buckets=(2,))
    wild = base.with_columns(
        pl.when(pl.col("week") == 3).then(pl.col("resid") * 50).otherwise(pl.col("resid"))
        .alias("resid"))
    a = LM.walk_forward(base).filter(pl.col("week") == 3)["sd_b"]
    b = LM.walk_forward(wild).filter(pl.col("week") == 3)["sd_b"]
    assert a.to_list() == b.to_list()


def test_a_closing_end_only_archive_scores_nothing():
    rows = planted_rows(FLAT, weeks=4, games=10, buckets=(0,))
    assert LM.walk_forward(rows).is_empty()


def test_an_unfitted_bucket_scores_as_the_incumbent_to_the_last_digit():
    rows = planted_rows(GROWING, weeks=3, games=5, buckets=(1, 2))  # never reaches the floor
    paired = LM.walk_forward(rows)
    assert paired.height and set(paired["kind"]) == {"inherited"}
    assert paired["diff"].abs().max() == 0.0


# --- the controls (rule 18) ------------------------------------------------------------------

def test_control_1_a_planted_archive_where_sd_grows_with_weeks_out_is_adopted():
    paired = LM.walk_forward(planted_rows(GROWING))
    assert float(paired["diff"].to_numpy().mean()) > 0
    run = quiet(paired)
    assert run.verdict[0] == "ADOPT", run.verdict


def test_control_2_a_flat_archive_shows_no_gain_and_no_adoption():
    paired = LM.walk_forward(planted_rows(FLAT, seed=3))
    run = quiet(paired)
    assert run.verdict[0] != "ADOPT", run.verdict
    assert abs(float(paired["diff"].to_numpy().mean())) < 0.003


def test_the_gate_is_clustered_on_the_week_not_the_season():
    run = quiet(LM.walk_forward(planted_rows(GROWING, weeks=10)))
    assert run.summary["clusters"] == 9  # weeks 2..10, one cluster each


def test_the_small_archive_the_gate_actually_has_is_not_runnable():
    """The 2026 shape: four held-out weeks of 15 games. The reachable branch is NOT-RUNNABLE,
    which the pre-registration names as an exemption (rule 16)."""
    rows = planted_rows(GROWING, weeks=5, games=16, buckets=(1, 2, 3), seed=7)
    run = quiet(LM.walk_forward(rows))
    assert run.verdict[0] in {"NOT-RUNNABLE", "SHOW"}, run.verdict


def test_an_empty_frame_has_a_verdict_and_nothing_measured():
    run = quiet(LM.walk_forward(planted_rows(FLAT, weeks=1, games=3)))
    assert run.verdict[0] == "SHOW" and run.summary["n"] == 0


# --- the description --------------------------------------------------------------------------

def test_describe_states_n_interval_and_the_direction_from_the_incumbent():
    table = LM.describe(planted_rows(GROWING, weeks=20, games=30))
    row = {r["bucket"]: r for r in table.iter_rows(named=True)}
    assert row[3]["kind"] == "fitted" and row[3]["above"] is True
    assert row[3]["lo"] < row[3]["sd"] < row[3]["hi"]
    assert row[0]["kind"] == "closing" and row[0]["above"] is False
    assert row[4]["n"] == 0 and row[4]["kind"] == "inherited"
    assert row[3]["mde_ratio"] == pytest.approx(
        np.exp(LM.Z_ALPHA_POWER / np.sqrt(2 * (row[3]["n"] - 1))))


def test_describe_on_a_flat_archive_does_not_exclude_the_incumbent():
    table = LM.describe(planted_rows(FLAT, weeks=20, games=30, seed=11))
    assert not table["above"].any()


def test_describe_lines_say_inherited_beside_a_thin_bucket():
    out = "\n".join(LM.describe_lines(LM.describe(planted_rows(GROWING, weeks=1, games=8))))
    assert "INHERITED" in out and "the closing end" in out


def test_span_lines_report_the_span_and_the_calendar_weeks():
    lines, _ = lines_and_games()
    text = LM.span_lines(lines)[0]
    assert "capture instants" in text and "calendar weeks" in text
    assert LM.span_lines(lines.clear()) == ["  archive: no snapshots"]


# --- the entry point --------------------------------------------------------------------------

def loader():
    rows = planted_rows(GROWING, weeks=6, games=16, seed=5)
    games = rows.select("game_id", "season", "week", "result").unique("game_id").with_columns(
        pl.lit(KICK).alias("kickoff"))
    snaps = rows.select("game_id", "bucket", pl.col("spread").alias("close_spread")).with_columns(
        pl.col("bucket").map_elements(lambda b: KICK - timedelta(weeks=[0.5, 1.5, 3.0, 6.0][b]),
                                      return_dtype=pl.Datetime).alias("captured_at"))
    return snaps.select("game_id", "close_spread", "captured_at"), games


def test_main_runs_once_and_writes_one_ledger_entry(capsys):
    ledger = Ledger(path=None)
    assert LM.main(["--run"], load=loader, ledger=ledger) == 0
    out = capsys.readouterr().out
    assert "the fit, in sample" in out and "HELD" in out
    assert "NOT-RUNNABLE" in out or "SHOW" in out
    entries = ledger._read() or []
    assert [e.name for e in entries] == [LM.NAME]


def test_main_without_run_prints_help(capsys):
    assert LM.main([]) == 0
    assert "--run" in capsys.readouterr().out


def test_main_degrades_when_the_archive_cannot_be_read(capsys):
    def boom():
        raise OSError("no store")
    assert LM.main(["--run"], load=boom, ledger=Ledger(path=None)) != 0


def test_main_with_nothing_scored_says_so(capsys):
    lines, games = lines_and_games()
    assert LM.main(["--run"], load=lambda: (lines.clear(), games),
                   ledger=Ledger(path=None)) == 1


# --- not wired --------------------------------------------------------------------------------

def test_nothing_imports_this_module():
    """The survivor consumer is HELD while #379 is dormant: no price may read this."""
    me = ROOT / "src" / "hub" / "models" / "lookahead_margin.py"
    offenders = [str(p) for p in (ROOT / "src").rglob("*.py")
                 if p != me and "lookahead_margin" in p.read_text()]
    assert offenders == []
