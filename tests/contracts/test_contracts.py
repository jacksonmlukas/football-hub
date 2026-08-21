"""Contract tests run against FROZEN fixtures, not the live API.

Live-API tests are flaky and useless in CI. These prove our parsing logic is correct.
A separate nightly job (tests/golden) hits the real API and diffs against the fixture,
which is what actually catches ESPN renaming a field.
"""
import polars as pl
import pytest
from hub.contracts import Contract, ContractViolation, DRAFT_BOARD


def test_contract_passes_on_valid_frame():
    df = pl.DataFrame({"player": [f"P{i}" for i in range(300)],
                       "pos": ["WR"] * 300,
                       "ecr": [float(i + 1) for i in range(300)]})
    assert DRAFT_BOARD.validate(df).height == 300


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
