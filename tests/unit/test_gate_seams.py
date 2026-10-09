"""The `load=` / `ledger=` seam on the gates that read nflverse inline (issue #403).

Each gate's `main` takes its one network-bound reader as a keyword argument, defaulting to the
real one. These tests hand `main` an injected reader that raises and read two things off the
run: the reader is the one the gate used (not `nflverse.load` behind its back), and the failure
is reported as `unavailable` and leaves no ledger entry. The lifecycle behind each (compare, run,
print, write) is exercised in the gate's own test module.
"""
import pytest

from hub.draft import backtest
from hub.ledger import Ledger
from hub.models import injury, margin, spread


@pytest.mark.parametrize(("module", "argv", "who"), [
    (spread, ["--fit"], "hub.models.spread"),
    (injury, ["--fit"], "hub.models.injury"),
])
def test_a_reader_that_raises_is_unavailable_and_files_nothing(module, argv, who, capsys):
    asked = []

    def load(source, seasons, **kw):
        asked.append(source)
        raise OSError("no cache")

    ledger = Ledger(path=None)
    assert module.main(argv, load=load, ledger=ledger) == 1
    assert asked, "main did not use the reader it was handed"
    assert who in capsys.readouterr().err
    assert ledger._read() in (None, [])


def test_margin_reads_through_the_injected_loader(capsys):
    def load():
        raise OSError("no cache")

    assert margin.main(["--fit"], load=load) == 1
    assert "hub.models.margin" in capsys.readouterr().err


def test_backtest_reads_through_the_injected_loader(capsys):
    seen = []

    def load(seasons, build_board, *, on_season=None):
        seen.append(list(seasons))
        raise OSError("no cache")

    assert backtest.main(["--seasons", "2024"], load=load) == 1
    assert seen == [[2024]]
    assert "hub.draft.backtest" in capsys.readouterr().err
