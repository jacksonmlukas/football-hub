"""The config exists to make model versions honest. These tests pin that."""
import pytest
from hub.config import (DraftConfig, HubConfig, PollConfig, RosterConfig,
                        config_digest, flex_share, starters)


def test_digest_is_stable_across_identical_configs():
    assert config_digest(HubConfig()) == config_digest(HubConfig())


def test_changing_a_hyperparameter_changes_the_digest():
    """The whole point: two lambdas must never share a model version."""
    a = HubConfig()
    b = HubConfig(draft=DraftConfig(projection_lambda=0.16))
    assert config_digest(a) != config_digest(b)


def test_operational_settings_do_not_change_the_digest():
    """Changing the poll interval must not invalidate a model version."""
    a = HubConfig()
    b = HubConfig(poll=PollConfig(scoreboard_interval=90))
    assert config_digest(a) == config_digest(b)


def test_roster_reflects_three_wr_league():
    assert starters(RosterConfig())["WR"] == 3


def test_flex_shares_sum_to_one():
    assert abs(sum(flex_share(RosterConfig()).values()) - 1.0) < 1e-9


def test_slot_is_three():
    assert RosterConfig().slot == 3
