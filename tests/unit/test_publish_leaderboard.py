"""The interim leaderboard is relayed from the forecasts branch into `site/data` (#543)."""
import json
import subprocess

import pytest

from hub import publish, publish_leaderboard

DOC = {
    "interim": True, "label": "Interim: descriptive, not a test.", "season": 2026,
    "first_scored_week": 10, "through_week": 10, "scored_weeks": [10],
    "contestants": [{"contestant": "m1:line_aware", "scorable": 14, "missing": 0,
                     "log_loss_diff": {"estimate": -0.01, "lo": -0.05, "hi": 0.03,
                                       "weeks": 1, "games": 14}}],
    "missing_games": [], "market_snapshot_ids": ["abc"], "packet_refused": 0,
}


def _published(tmp_path):
    return json.loads((tmp_path / "leaderboard.json").read_text())


def test_the_scorers_document_is_published_unchanged(tmp_path):
    got = publish_leaderboard.leaderboard(tmp_path, read=lambda: DOC)
    assert isinstance(got, dict)
    on_disk = _published(tmp_path)
    for key, value in DOC.items():
        assert on_disk[key] == value
    assert on_disk["name"] == "leaderboard" and on_disk["shape"] == "summary"
    assert on_disk["generated_at"] and on_disk["n_contestants"] == 1


def test_no_minimum_game_count_is_invented(tmp_path):
    """The prereg defines none, so the scorer writes none and the relay adds none."""
    publish_leaderboard.leaderboard(tmp_path, read=lambda: DOC)
    assert "min_games" not in _published(tmp_path)


def test_a_minimum_the_scorer_writes_is_relayed(tmp_path):
    publish_leaderboard.leaderboard(tmp_path, read=lambda: {**DOC, "min_games": 30})
    assert _published(tmp_path)["min_games"] == 30


def test_the_manifest_lists_it(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_leaderboard, "interim_from_git", lambda: DOC)
    art = next(a for a in publish.artifacts(2026, 1, out=tmp_path) if a.name == "leaderboard")
    entry = art.record(tmp_path)
    assert entry["present"] and not entry["stale"] and entry["generated_at"]


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"contestants": []},
                                 {"contestants": "no"}, {**DOC, "rows": [1]}],
                         ids=["unreadable", "list", "string", "empty-doc", "no-contestants",
                              "bad-contestants", "envelope-clash"])
def test_a_failed_or_unusable_read_leaves_the_published_board(tmp_path, bad):
    publish_leaderboard.leaderboard(tmp_path, read=lambda: DOC)
    before = (tmp_path / "leaderboard.json").read_text()
    got = publish_leaderboard.leaderboard(tmp_path, read=lambda: bad)
    assert isinstance(got, publish.Kept)
    assert (tmp_path / "leaderboard.json").read_text() == before


def test_with_nothing_published_and_nothing_read_the_manifest_says_so(tmp_path):
    got = publish_leaderboard.leaderboard(tmp_path, read=lambda: None)
    assert got is None
    art = next(a for a in publish.artifacts(2026, 1, out=tmp_path) if a.name == "leaderboard")
    entry = publish.Artifact(art.name, lambda: None, art.reason).record(tmp_path)
    assert entry["stale"] and not entry["present"] and entry["reason"]


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_the_git_reader_returns_the_file_on_the_ref(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "forecasts")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "leaderboard").mkdir()
    (tmp_path / "leaderboard" / "interim.json").write_text(json.dumps(DOC))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "x")
    assert publish_leaderboard.interim_from_git("forecasts", root=tmp_path) == DOC
    assert publish_leaderboard.interim_from_git("origin/nope", root=tmp_path) is None
    assert publish_leaderboard.interim_from_git(
        "forecasts", "leaderboard/missing.json", root=tmp_path) is None


def test_the_git_reader_survives_a_non_json_file(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "forecasts")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "leaderboard").mkdir()
    (tmp_path / "leaderboard" / "interim.json").write_text("{not json")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "x")
    assert publish_leaderboard.interim_from_git("forecasts", root=tmp_path) is None


def test_cli_leaderboard_flag(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(publish_leaderboard, "interim_from_git", lambda: DOC)
    assert publish.main(["--leaderboard", "--week", "1", "--out", str(tmp_path)]) == 0
    assert "1 contestants" in capsys.readouterr().out
    assert (tmp_path / "leaderboard.json").exists()


def test_the_default_root_is_the_repository_root():
    """Production calls the reader with no `root`; a wrong `ROOT` makes `git show` fail and
    the board is Kept forever, quietly."""
    root = publish_leaderboard.ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / ".git").exists()  # a directory, or a file in a worktree
    assert (root / "src" / "hub" / "publish_leaderboard.py").is_file()
