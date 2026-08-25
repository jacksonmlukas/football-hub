"""Contract tests run against FROZEN fixtures, not the live API.

Live-API tests are flaky and useless in CI. These prove our parsing logic is correct.
A separate nightly job (tests/golden) hits the real API and diffs against the fixture,
which is what actually catches ESPN renaming a field.
"""
import polars as pl
import pytest

from hub.contracts import DRAFT_BOARD, Contract, ContractViolation


def _board(n=300, **override):
    """A frame carrying every column `DRAFT_BOARD` requires.

    Built from the contract itself rather than a hand-written column list, so adding a
    required column cannot leave this fixture quietly behind the thing it is testing.
    """
    fill = {pl.Utf8: [f"v{i}" for i in range(n)],
            pl.Float64: [float(i + 1) for i in range(n)],
            pl.UInt32: pl.Series([16] * n, dtype=pl.UInt32)}
    df = pl.DataFrame({c: fill[dt] for c, dt in DRAFT_BOARD.required.items()})
    return df.with_columns(**override) if override else df


def test_contract_passes_on_valid_frame():
    assert DRAFT_BOARD.validate(_board()).height == 300


def test_the_board_contract_names_the_columns_downstream_reads():
    """It declared three columns and was applied to nothing, while ~14 modules read the
    board by name. These are the ones read unconditionally."""
    assert {"player", "pos", "ecr", "xfp_per_game", "vor"} <= set(DRAFT_BOARD.required)


def test_the_consensus_spread_is_contracted_under_its_disambiguated_name():
    """`sd` meant a rank spread here and a points spread in `hub.models.predict`."""
    assert "sd" not in DRAFT_BOARD.required


def test_optional_columns_are_not_contracted():
    """`build` degrades on purpose when a fetch fails -- contracting `edge` or `td_luck`
    would turn a documented graceful degradation into a hard failure on draft night."""
    for optional in ("edge", "proj_blend", "td_luck", "wk15_17_sos", "vor_proj"):
        assert optional not in DRAFT_BOARD.required


def test_contract_catches_missing_column():
    df = pl.DataFrame({"player": ["A"], "ecr": [1.0]})
    with pytest.raises(ContractViolation, match="missing columns"):
        Contract("t", required={"player": pl.Utf8, "pos": pl.Utf8}).validate(df)


def test_contract_catches_duplicate_players():
    df = pl.DataFrame({"player": ["A", "A"], "pos": ["WR", "WR"], "ecr": [1.0, 2.0]})
    with pytest.raises(ContractViolation, match="not unique"):
        Contract("t", required={"player": pl.Utf8}, unique=("player",)).validate(df)


def test_contract_catches_out_of_range():
    """A scoring-rule change that doubles projections should fail loudly, not silently."""
    df = pl.DataFrame({"xfp": [500.0]})
    with pytest.raises(ContractViolation, match="range"):
        Contract("t", required={"xfp": pl.Float64}, ranges={"xfp": (-10, 80)}).validate(df)
