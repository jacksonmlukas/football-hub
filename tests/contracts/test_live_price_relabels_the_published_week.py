"""The poll-age cut re-labels the runner's published week exactly as it was published.

#251 established, on the Actions runner's first output after #218, that `site/data/
preds_2026_wk02.json` carries sixteen rows, every one `price_source = live`, none adjusted.
That is the pinned production case for #297, which moved the cut from the quote's last move
to the age of the capture that priced the row: the runner starts every run with an empty
store and takes one fresh snapshot, so its every capture is seconds old and must read as
live under the new cut as it did under the old one, and the adjustment must reach none of
them.

A contract test rather than a unit test because the evidence is the committed artifact, not
a fixture: the artifact carries `priced_at` (the capture) and `predicted_at` (the moment the
cut was applied at), which is exactly the pair `hub.models.quarterback.live_price` reads, so
the published labels can be re-derived from the published rows and held to what the runner
wrote. It does not carry `unmoved_since`, which is why the old cut could not have been held
this way.
"""
import datetime as dt
import json
from pathlib import Path

import polars as pl

from hub.models import quarterback

ARTIFACT = Path(__file__).resolve().parents[2] / "site" / "data" / "preds_2026_wk02.json"

# The counts #251 quoted from the runner's stamp: 16 rows, 16 live, 0 adjusted.
PUBLISHED = {"n": 16, "live": 16, "adjusted": 0}


def _rows() -> pl.DataFrame:
    doc = json.loads(ARTIFACT.read_text())
    assert doc["n"] == PUBLISHED["n"] and len(doc["rows"]) == PUBLISHED["n"]
    rows = pl.DataFrame(doc["rows"])
    return rows.with_columns(
        pl.col("priced_at").str.to_datetime(time_unit="us"),
        pl.col("predicted_at").str.to_datetime(time_unit="us"))


def test_the_runners_week_two_is_sixteen_live_and_none_adjusted_as_published():
    rows = _rows()
    assert rows["price_source"].value_counts().to_dicts() == [
        {"price_source": "live", "count": PUBLISHED["live"]}]
    assert rows["adjusted_by"].null_count() == PUBLISHED["n"] - PUBLISHED["adjusted"]
    assert rows["qb_adjustment"].null_count() == PUBLISHED["n"] - PUBLISHED["adjusted"]
    assert set(rows["model"].to_list()) == {"market_baseline"}


def test_the_poll_age_cut_re_labels_every_published_row_identically():
    """Applied at each row's own `predicted_at`, the cut says live for every capture the
    runner priced from -- the same sixteen labels the artifact carries."""
    rows = _rows()
    moments = sorted(set(rows["predicted_at"].to_list()))
    assert len(moments) == 1, "one run stamped the week, so one moment applies to it"
    at: dt.datetime = moments[0]
    relabelled = rows.select(
        pl.when(quarterback.live_price(at)).then(pl.lit("live"))
          .otherwise(pl.lit("stale")).alias("relabelled"),
        pl.col("price_source"))
    assert relabelled["relabelled"].to_list() == relabelled["price_source"].to_list()
    # Every capture was seconds old at the moment the run priced from it.
    ages = rows.select((pl.col("predicted_at") - pl.col("priced_at"))
                       .dt.total_seconds().alias("age"))["age"].to_list()
    assert ages and max(ages) < 60


def test_the_poll_age_cut_leaves_the_adjustment_nothing_to_touch():
    """`quarterback.no_live_price` over the published rows selects none of them, so the
    adjustment count the runner printed -- 0 of 16 -- is what this cut reproduces."""
    rows = _rows().with_columns(pl.lit(3.0).alias("close_spread"))
    reached = rows.filter(quarterback.no_live_price(rows))
    assert reached.height == PUBLISHED["adjusted"]
