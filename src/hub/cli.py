"""What a command-line entry point says when the thing it needed is not there.

Every CLI in this repo has to meet absent input with a sentence and a non-zero exit rather
than a traceback -- `tests/contracts/test_cli_surface.py` asserts exactly that, of every
module in the tree with a `main()`, and CLAUDE.md's degradation rule is why:

    If a fetch fails, serve last-good state from `data/processed/` rather than erroring.
    Systems that need an operator die in October.

Eighteen entry points had to grow that sentence at once, which is the whole reason this leaf
exists. Eighteen hand-written variants of "print something, return 1" is eighteen chances to
print nothing, to return 0 with a message, or to write to stdout where the caller is reading
stderr -- and the property being asserted is about the *shape* of the answer, so the shape
gets one owner rather than eighteen copies that agree today.

Deliberately not a decorator over `main`, and not a wrapper that swallows everything a CLI
does. Each call site names the input it was reaching for, because "hub.models.margin:
nflverse schedules unavailable" is the sentence an operator can act on and "hub.models.margin
failed" is not. What is shared is the format and the exit code; what a run was missing stays
with the code that went looking for it.

A leaf on purpose, like `hub.paths` and `hub.names`: it imports nothing from this repo, so
the modules at the bottom of the tree can answer in the same voice as the ones at the top.
"""
from __future__ import annotations

import sys


def unavailable(prog: str, what: str, exc: BaseException) -> int:
    """Report an input that could not be read, and hand back the exit code that says so.

    `what` names the input rather than the operation -- "nflverse schedules", "the live
    ESPN draft", "the processed store" -- because the reader's next question is what to go
    and get, not which line raised.

    The exception's type and message are both printed. The type alone loses the URL that
    failed and the file that was not there; the message alone reads as prose from this repo
    when it is prose from `nflreadpy`. Together they are the shortest thing that still says
    where the sentence came from.

    stderr, not stdout, and returned rather than raised: `make slate` puts a leading `-` on
    the optional sources and reads exit codes, and the record of what a run did is written
    by the caller that knows -- `hub.fetch.cfbd.record_run` is the one that does.
    """
    print(f"{prog}: {what} unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1
