import pytest
from hub.draft.picks import next_two, snake_picks


def test_slot_one_gets_the_turn():
    p = snake_picks(1, teams=12, rounds=4)
    assert p == [1, 24, 25, 48]


def test_last_slot_gets_the_wheel():
    p = snake_picks(12, teams=12, rounds=4)
    assert p == [12, 13, 36, 37]


def test_middle_slot_is_evenly_spaced():
    gaps = [b - a for a, b in zip(snake_picks(6, 12, 6), snake_picks(6, 12, 6)[1:])]
    assert max(gaps) - min(gaps) <= 2  # no long wait, no back-to-back


def test_every_pick_is_unique_and_covers_the_board():
    allp = sorted(p for s in range(1, 13) for p in snake_picks(s, 12, 16))
    assert allp == list(range(1, 12 * 16 + 1))


def test_slot_out_of_range_rejected():
    with pytest.raises(ValueError):
        snake_picks(13, teams=12)


def test_next_two_after_some_picks():
    assert next_two(snake_picks(3, 12, 5), drafted=10) == (22, 27)


def test_slot_three_alternates_long_and_short_waits():
    from hub.draft.picks import my_picks
    gaps = [b - a for a, b in zip(my_picks(8), my_picks(8)[1:])]
    assert gaps == [19, 5, 19, 5, 19, 5, 19]


def test_draft_mode_alternates():
    from hub.draft.picks import draft_mode
    assert draft_mode(3) == "scarcity"    # 19-pick wait follows
    assert draft_mode(22) == "value"      # next turn is 5 away
    assert draft_mode(27) == "scarcity"
    assert draft_mode(46) == "value"


def test_draft_mode_rejects_a_pick_that_is_not_yours():
    from hub.draft.picks import draft_mode
    with pytest.raises(ValueError):
        draft_mode(15)
