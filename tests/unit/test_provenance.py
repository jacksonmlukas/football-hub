"""Prediction provenance.

`docs/foundation-plan.md` 3.5, whose done-when is precise: two runs differing only in a
hyperparameter must produce distinguishable rows.

`FitSpec.digest` already folded a config digest into every model's version string, and
`MarketBaseline.version` already returned `market-{digest}`. The part that was missing is
that nothing ever *put* the live config into the spec -- `ratings.fit` built
`FitSpec("nfl", season, wk - 1)` and took the `cfg_digest="default"` default, so every run
under every configuration produced the same version.

That is the failure mode this repo cares about most: not a crash, but two different models
writing indistinguishable rows into a public track record that claims a specific model made
a specific prediction.
"""
import polars as pl
import pytest

from hub.config import HubConfig, ModelConfig, config_digest
from hub.models import ratings
from hub.models.base import FitSpec


def test_the_config_digest_reaches_the_spec():
    a = FitSpec("nfl", 2026, 3, cfg_digest=config_digest(HubConfig()))
    b = FitSpec("nfl", 2026, 3, cfg_digest="default")
    assert a.digest != b.digest


def test_two_configs_differing_in_one_hyperparameter_are_distinguishable():
    """The done-when, stated directly."""
    base = HubConfig()
    tweaked = HubConfig(model=ModelConfig(conformal_alpha=0.05))
    assert config_digest(base) != config_digest(tweaked)
    assert (FitSpec("nfl", 2026, 3, cfg_digest=config_digest(base)).digest
            != FitSpec("nfl", 2026, 3, cfg_digest=config_digest(tweaked)).digest)


def test_the_same_config_is_stable_across_runs():
    """A digest that moved on its own would make every week look like a new model and the
    track record would never accumulate."""
    assert config_digest(HubConfig()) == config_digest(HubConfig())


def test_predictions_carry_the_config_digest_as_a_column():
    """Readable without decoding a hash. A provenance field nobody can query is decoration."""
    assert "cfg_digest" in ratings.PROVENANCE_COLUMNS


def test_fit_stamps_provenance_onto_every_row(monkeypatch, tmp_path):
    games = pl.DataFrame({
        "game_id": ["g1"], "league": ["nfl"], "season": pl.Series([2026], dtype=pl.Int32),
        "week": pl.Series([1], dtype=pl.Int32), "home_team": ["KC"], "away_team": ["LV"],
        "close_spread": [3.0], "result": [None]})
    monkeypatch.setattr(ratings, "games_for", lambda season, cache=None: games)
    got = ratings.fit(2026, 1, base=tmp_path)
    assert got.height == 1
    for col in ratings.PROVENANCE_COLUMNS:
        assert col in got.columns and got[col][0] is not None


def test_two_runs_under_different_config_do_not_overwrite_each_other(monkeypatch, tmp_path):
    """Distinguishable rows are not enough if the second run lands on the first one's file."""
    games = pl.DataFrame({
        "game_id": ["g1"], "league": ["nfl"], "season": pl.Series([2026], dtype=pl.Int32),
        "week": pl.Series([1], dtype=pl.Int32), "home_team": ["KC"], "away_team": ["LV"],
        "close_spread": [3.0], "result": [None]})
    monkeypatch.setattr(ratings, "games_for", lambda season, cache=None: games)

    monkeypatch.setattr(ratings, "live_config", lambda: HubConfig())
    ratings.fit(2026, 1, base=tmp_path)
    monkeypatch.setattr(ratings, "live_config",
                        lambda: HubConfig(model=ModelConfig(conformal_alpha=0.05)))
    ratings.fit(2026, 1, base=tmp_path)

    written = sorted((tmp_path / "preds").rglob("*.parquet"))
    assert len(written) == 2, "one file per configuration, not one overwritten twice"
    digests = set()
    for f in written:
        digests |= set(pl.read_parquet(f)["cfg_digest"].to_list())
    assert len(digests) == 2
