"""The poll-age cut re-labels the runner's published week exactly as it was published.

#251 established, on the Actions runner's first output after #218, that `site/data/
preds_2026_wk02.json` carried sixteen rows, every one `price_source = live`, none adjusted.
That was the pinned production case for #297, which moved the cut from the quote's last
move to the age of the capture that priced the row, and it is the case that first held
this test (2026-09-16: 16 rows, 16 live, 0 adjusted, every capture about two seconds old
at the moment the run priced from it).

The weekly runner regenerates that file, so what is held is the *rule* over whatever the
artifact holds on the day, not the week's counts: the label re-derived from each row's own
`priced_at` (the capture) and `predicted_at` (the moment the cut was applied at) -- exactly
the pair `hub.models.quarterback.live_price` reads -- equals the label the runner wrote, and
a row labelled live carries no adjustment. A contract test rather than a unit test because
the evidence is the committed artifact, not a fixture. The artifact does not carry
`unmoved_since`, which is why the old cut could not have been held this way.
"""
import json
from pathlib import Path

import polars as pl

from hub.models import quarterback

ARTIFACT = Path(__file__).resolve().parents[2] / "site" / "data" / "preds_2026_wk02.json"


def _rows() -> pl.DataFrame:
    doc = json.loads(ARTIFACT.read_text())
    rows = pl.DataFrame(doc["rows"])
    assert rows.height == doc["n"] > 0
    return rows.with_columns(
        pl.col("priced_at").str.to_datetime(time_unit="us"),
        pl.col("predicted_at").str.to_datetime(time_unit="us"))


def test_the_poll_age_cut_re_labels_every_published_row_identically():
    """Applied at each row's own `predicted_at`, the cut reproduces the label the runner
    wrote. Rows the moving field priced carry no capture and are not asked."""
    rows = _rows().filter(pl.col("price_source").is_in(["live", "stale"]))
    assert rows.height, "the artifact holds no snapshot-priced row to hold the cut to"
    relabelled = pl.concat([
        group.select(
            pl.when(quarterback.live_price(moment)).then(pl.lit("live"))
              .otherwise(pl.lit("stale")).alias("relabelled"),
            pl.col("price_source"))
        for (moment,), group in rows.group_by("predicted_at")])
    assert relabelled["relabelled"].to_list() == relabelled["price_source"].to_list()
    ages = rows.select((pl.col("predicted_at") - pl.col("priced_at"))
                       .dt.total_seconds().alias("age"))["age"].to_list()
    print(f"::notice::{ARTIFACT.name}: {rows.height} snapshot-priced rows, capture age at "
          f"the run's moment {min(ages)}-{max(ages)} s")


def test_a_row_labelled_live_carries_no_adjustment():
    """The two consumers of the cut agree on the published rows: where the label says a
    live price existed, `adjusted_by` and `qb_adjustment` are null, and `no_live_price`
    would not have reached the row."""
    rows = _rows()
    live = rows.filter(pl.col("price_source") == "live")
    assert live["adjusted_by"].null_count() == live.height
    assert live["qb_adjustment"].null_count() == live.height
    priced = rows.with_columns(pl.lit(3.0).alias("close_spread"))
    reached = priced.filter(quarterback.no_live_price(priced))
    assert set(reached["price_source"].to_list()) <= {"stale", "schedule"}
    assert reached.height == rows.filter(
        pl.col("price_source").is_in(["stale", "schedule"])).height
    print(f"::notice::{ARTIFACT.name}: {rows.height} rows, {live.height} live, "
          f"{rows.height - live.height} without a live price")
