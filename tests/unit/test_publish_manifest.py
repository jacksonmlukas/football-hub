"""What the manifest lists about the site beyond the week it was written for (#480).

`weeks` is every published `preds_*` artifact, and `bigten` / `cfbd` are manifest artifacts with
the same entry as every other. The page used to keep both facts itself -- a hard-coded archive
list and two side-file fetches -- and a second copy is the thing that rots.
"""
import json

import pytest

from hub import publish


def _preds(site, season, week, **extra):
    name = f"preds_{season}_wk{week:02d}"
    (site / f"{name}.json").write_text(json.dumps(
        {"name": name, "season": season, "week": week,
         "generated_at": f"{season}-09-0{week % 9 + 1}T00:00:00+00:00", **extra}))
    return name


@pytest.fixture
def site(tmp_path, monkeypatch):
    # The artifacts themselves are other tests' business; this file is about the listing.
    monkeypatch.setattr(publish, "artifacts", lambda *a, **k: [])
    return tmp_path


def test_weeks_lists_exactly_the_preds_files_present_newest_first(site):
    a, b, c = _preds(site, 2025, 18), _preds(site, 2026, 1), _preds(site, 2026, 2)
    (site / "track_record.json").write_text("{}")          # not a week
    got = publish.publish_all(2026, 2, out=site)["weeks"]
    assert [w["name"] for w in got] == [c, b, a]
    assert got[0] == {"name": c, "season": 2026, "week": 2,
                      "generated_at": "2026-09-03T00:00:00+00:00"}

    (site / f"{b}.json").unlink()
    publish.publish_all(2026, 2, out=site)
    got = json.loads((site / "manifest.json").read_text())["weeks"]
    assert [w["name"] for w in got] == [c, a], "a deleted week is still listed"


def test_a_week_that_cannot_be_read_is_still_listed_by_its_name(site):
    (site / "preds_2026_wk03.json").write_text("{not json")
    (site / "preds_2026_wk04.json").write_text("[]")
    got = publish.published_weeks(site)
    assert [(w["name"], w["season"], w["week"], w["generated_at"]) for w in got] == [
        ("preds_2026_wk04", 2026, 4, None), ("preds_2026_wk03", 2026, 3, None)]


def test_an_empty_site_lists_no_weeks(site):
    assert publish.publish_all(2026, 1, out=site)["weeks"] == []


def test_the_manifest_carries_each_side_files_own_stale_and_reason(site):
    (site / "bigten.json").write_text(json.dumps(
        {"fetched": True, "stale": True, "reason": "the page's shape changed",
         "generated_at": "2026-10-16T06:08:42+00:00"}))
    (site / "cfbd.json").write_text(json.dumps(
        {"fetched": False, "stale": True, "reason": "nothing was fetched: TypeError",
         "generated_at": "2026-10-14T15:41:51+00:00"}))
    by = {a["name"]: a for a in publish.publish_all(2026, 1, out=site)["artifacts"]}
    assert by["bigten"] == {"name": "bigten", "present": True, "stale": True,
                            "reason": "the page's shape changed",
                            "generated_at": "2026-10-16T06:08:42+00:00"}
    assert by["cfbd"]["stale"] is True and by["cfbd"]["present"] is True
    assert by["cfbd"]["reason"] == "nothing was fetched: TypeError"


def test_a_healthy_side_file_is_not_stale(site):
    (site / "bigten.json").write_text(json.dumps(
        {"fetched": True, "stale": False, "reason": None, "generated_at": "t"}))
    got = publish.side_record(site, "bigten")
    assert got["stale"] is False and got["reason"] is None and got["present"] is True


def test_an_unfetched_side_file_is_stale_even_if_it_forgot_to_say_so(site):
    (site / "cfbd.json").write_text(json.dumps({"fetched": False}))
    assert publish.side_record(site, "cfbd")["stale"] is True


def test_an_absent_side_file_is_not_present_with_a_standing_reason(site):
    by = {a["name"]: a for a in publish.publish_all(2026, 1, out=site)["artifacts"]}
    for name in ("bigten", "cfbd"):
        assert by[name] == {"name": name, "present": False, "stale": True,
                            "reason": f"no {name}.json has been published",
                            "generated_at": None}


@pytest.mark.parametrize("body", ["{not json", "[1]", "null"])
def test_a_damaged_side_file_degrades_to_a_stale_entry_not_an_error(site, body):
    (site / "bigten.json").write_text(body)
    got = publish.side_record(site, "bigten")
    assert got["present"] is True and got["stale"] is True
    assert got["reason"] == "bigten.json could not be read"
