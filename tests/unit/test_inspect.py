"""The summarizing CLI the data-guard hook points at.

`.claude/hooks/guard_data_reads.py` blocks raw data reads and tells the agent to run
`hub.inspect` instead. So this module is the escape hatch for the repo's most important
rule, and it has exactly one hard obligation: **it must never itself become the expensive
read it exists to prevent.**

Two properties carry that:

  * every mode goes through `pl.scan_parquet`, never `read_parquet`. Enforced here by
    making `read_parquet` raise -- if any mode reaches for it, these tests fail.
  * output is capped centrally. A 300-column play-by-play `--schema` is the case that
    would otherwise quietly blow past the limit.
"""
import polars as pl
import pytest
from hub import inspect as ins


@pytest.fixture
def base(tmp_path):
    """A stand-in for data/processed, with the three shapes resolve() must handle."""
    (tmp_path / "draft_board.parquet").parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"player": ["A", "B", "C"], "pos": ["RB", "WR", "TE"],
                  "adp": [1.5, None, 30.0]}).write_parquet(tmp_path / "draft_board.parquet")
    hive = tmp_path / "preds" / "league=nfl" / "season=2026" / "week=01"
    hive.mkdir(parents=True)
    pl.DataFrame({"game_id": ["g1"], "home_win_prob": [0.61]}).write_parquet(hive / "part.parquet")
    return tmp_path


@pytest.fixture(autouse=True)
def forbid_eager_reads(monkeypatch):
    """Nothing in this module may call read_parquet. That is the whole point of it."""
    def _boom(*a, **k):
        raise AssertionError("hub.inspect must use scan_parquet, never read_parquet")
    monkeypatch.setattr(pl, "read_parquet", _boom)


# --- resolution -----------------------------------------------------------

def test_resolves_an_explicit_path(base):
    p = base / "draft_board.parquet"
    assert ins.resolve(str(p), base=base) == [p]


def test_resolves_a_bare_name(base):
    assert ins.resolve("draft_board", base=base) == [base / "draft_board.parquet"]


def test_resolves_a_hive_partitioned_directory(base):
    found = ins.resolve("preds", base=base)
    assert len(found) == 1 and found[0].name == "part.parquet"


def test_unknown_dataset_names_what_is_available(base):
    with pytest.raises(ins.DatasetNotFound) as e:
        ins.resolve("does_not_exist", base=base)
    assert "draft_board" in str(e.value), "the error must list what the user could have meant"


def test_empty_hive_directory_is_not_a_match(base, tmp_path):
    (base / "empty_table").mkdir()
    with pytest.raises(ins.DatasetNotFound):
        ins.resolve("empty_table", base=base)


# --- the cap --------------------------------------------------------------

def test_cap_truncates_and_says_how_much_it_hid():
    out = ins.cap([f"line {i}" for i in range(200)], limit=10)
    assert len(out) == 10
    # 9 content lines plus the marker itself, so 191 are hidden, not 190.
    assert "191 more" in out[-1]


def test_cap_leaves_short_output_alone():
    assert ins.cap(["a", "b"], limit=10) == ["a", "b"]


def test_schema_of_a_wide_frame_stays_capped(tmp_path):
    """Full-width play-by-play is ~300 columns. This is the case that matters."""
    wide = pl.DataFrame({f"c{i}": [1.0] for i in range(300)})
    p = tmp_path / "wide.parquet"
    wide.write_parquet(p)
    assert len(ins.cap(ins.schema_lines([p]))) <= ins.MAX_LINES


def test_every_mode_respects_the_cap(tmp_path):
    wide = pl.DataFrame({f"c{i}": [1.0, None] for i in range(300)})
    p = tmp_path / "wide.parquet"
    wide.write_parquet(p)
    for lines in (ins.overview_lines([p]), ins.schema_lines([p]),
                  ins.null_lines([p]), ins.describe_lines([p]),
                  ins.head_lines([p], n=50, cols=None)):
        assert len(ins.cap(lines)) <= ins.MAX_LINES


# --- modes ----------------------------------------------------------------

def test_overview_reports_shape(base):
    text = " ".join(ins.overview_lines(ins.resolve("draft_board", base=base)))
    assert "3" in text and "rows" in text.lower()
    assert "3" in text and "column" in text.lower()


def test_schema_lists_columns_with_dtypes(base):
    text = "\n".join(ins.schema_lines(ins.resolve("draft_board", base=base)))
    assert "player" in text and "pos" in text and "adp" in text
    assert "str" in text.lower() or "utf8" in text.lower()


def test_head_selects_requested_columns_only(base):
    text = "\n".join(ins.head_lines(ins.resolve("draft_board", base=base), n=2, cols=["player"]))
    assert "player" in text
    assert "pos" not in text


def test_head_rejects_an_unknown_column(base):
    """Silently returning nothing would read as 'the column is empty'."""
    with pytest.raises(ins.DatasetNotFound):
        ins.head_lines(ins.resolve("draft_board", base=base), n=2, cols=["nope"])


def test_nulls_reports_the_column_that_has_them(base):
    text = "\n".join(ins.null_lines(ins.resolve("draft_board", base=base)))
    assert "adp" in text


def test_nulls_omits_clean_columns(base):
    text = "\n".join(ins.null_lines(ins.resolve("draft_board", base=base)))
    assert "player" not in text, "columns with no nulls are noise here"


def test_describe_covers_numeric_and_skips_strings(base):
    text = "\n".join(ins.describe_lines(ins.resolve("draft_board", base=base)))
    assert "adp" in text
    assert "player" not in text


def test_modes_work_on_a_multi_file_hive_dataset(base):
    paths = ins.resolve("preds", base=base)
    assert "home_win_prob" in "\n".join(ins.schema_lines(paths))
    assert "1" in " ".join(ins.overview_lines(paths))


# --- end to end -----------------------------------------------------------

def test_cli_runs_and_stays_within_the_cap(base, capsys):
    assert ins.main(["draft_board", "--schema", "--base", str(base)]) == 0
    out = capsys.readouterr().out.rstrip("\n").split("\n")
    assert len(out) <= ins.MAX_LINES


def test_cli_reports_a_missing_dataset_without_a_traceback(base, capsys):
    assert ins.main(["nope", "--base", str(base)]) == 1
    assert "nope" in capsys.readouterr().err


def test_cli_defaults_to_overview_when_no_mode_is_given(base, capsys):
    assert ins.main(["draft_board", "--base", str(base)]) == 0
    assert "rows" in capsys.readouterr().out.lower()
