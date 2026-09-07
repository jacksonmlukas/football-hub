"""The reliability diagram, and the generalisation of it that bins on an input.

`reliability` bins a probability against its own value, which answers "when the model said
70%, did it happen 70% of the time". `reliability_by` bins on something else -- the spread a
survivor pick was chosen by -- because a miss concentrated in one range of the *input* is
invisible in a diagram whose bins are probabilities.

The two share a binning loop rather than holding two copies of it, so the first thing tested
here is that the shared version did not change what the published curve looks like.
"""
import polars as pl
import pytest

from hub.models import scoring_rules


def _frame(probs, won):
    return pl.DataFrame({"home_win_prob": [float(p) for p in probs],
                         "home_won": [int(w) for w in won]})


# --- what the published curve already promised ---------------------------

def test_the_bins_and_their_labels_are_unchanged():
    got = scoring_rules.reliability(_frame([0.05, 0.15, 0.95], [1, 0, 1]), n_bins=10)
    assert [b["bin"] for b in got] == [
        "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
        "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    assert [b["n"] for b in got] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]


def test_an_empty_bin_is_kept_and_says_nothing():
    got = scoring_rules.reliability(_frame([0.05], [1]), n_bins=2)
    assert got[1] == {"bin": "0.5-1.0", "n": 0, "predicted": None, "actual": None,
                      "gap": None}


def test_a_probability_of_exactly_one_lands_in_the_top_bin():
    """The last bin used to admit its upper edge by comparing against 1.01. It now includes
    the edge, which for a probability validated into [0, 1] is the same set of rows -- and
    a certainty falling out of the diagram entirely is the failure either spelling avoids."""
    got = scoring_rules.reliability(_frame([1.0], [1]), n_bins=10)
    assert got[-1]["n"] == 1


def test_an_empty_frame_is_no_diagram_rather_than_ten_empty_bins():
    assert scoring_rules.reliability(_frame([], [])) == []


def test_predicted_and_actual_are_the_two_things_the_page_compares():
    got = scoring_rules.reliability(_frame([0.72, 0.78], [1, 0]), n_bins=10)
    hit = next(b for b in got if b["n"])
    assert hit["predicted"] == pytest.approx(0.75)
    assert hit["actual"] == pytest.approx(0.5)
    assert hit["gap"] == pytest.approx(-0.25), "gap is actual minus predicted"


# --- binning on the input ------------------------------------------------

def test_it_bins_on_the_named_column_not_on_the_probability():
    """Two games with the same probability and different spreads must land in different
    bins, which is exactly what a probability-binned diagram cannot do."""
    df = pl.DataFrame({"spread": [1.0, 12.0], "p": [0.6, 0.6], "won": [0, 1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert [b["n"] for b in got] == [1, 1]
    assert [b["actual"] for b in got] == [0.0, 1.0]


def test_the_top_bucket_includes_its_upper_edge():
    df = pl.DataFrame({"spread": [20.0], "p": [0.9], "won": [1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert got[-1]["n"] == 1, "a game at the top of the range must not fall out"


def test_a_positive_gap_is_a_model_that_was_under_confident():
    df = pl.DataFrame({"spread": [10.0] * 10, "p": [0.7] * 10, "won": [1] * 9 + [0]})
    got = scoring_rules.reliability_by(df, [0.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert got[0]["gap"] == pytest.approx(0.2)


def test_rows_outside_the_edges_are_not_counted():
    df = pl.DataFrame({"spread": [-5.0, 3.0, 99.0], "p": [0.4, 0.6, 0.99],
                       "won": [0, 1, 1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0], on="spread", prob="p",
                                       outcome="won")
    assert sum(b["n"] for b in got) == 1


def test_the_bin_label_carries_the_places_it_is_asked_for():
    df = pl.DataFrame({"spread": [3.5], "p": [0.6], "won": [1]})
    got = scoring_rules.reliability_by(df, [0.0, 7.0], on="spread", prob="p",
                                       outcome="won", places=0)
    assert got[0]["bin"] == "0-7"
