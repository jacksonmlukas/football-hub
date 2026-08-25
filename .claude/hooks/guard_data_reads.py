#!/usr/bin/env python3
"""Block the single most expensive mistake: pulling a data file into context.

CLAUDE.md asks. This enforces. Exit 2 blocks the tool call and hands the reason
back to Claude, which then routes through a summarizing CLI instead.
"""
import json, re, sys

BLOCKED_READ = re.compile(r"(data/(raw|interim|processed)/|\.parquet$|\.csv$)")

# `[^|\n]*` rather than `[^|]*`: a reader command and its argument live on ONE line, but
# heredoc bodies and commit messages do not. Without the \n this matched a `tail` in a pipe
# on line 1 against a `data/` mention twenty lines later, and refused to write any document
# that merely describes the cache layout. A guard that fires while you are writing prose
# teaches you to ignore it, which is how a guardrail stops working.
# Accepted gap: a read split across a `\` line continuation slips through. Rare, and the
# Read branch above still catches file reads.
# Scoped to the same three directories as BLOCKED_READ. This used to match any `data/`,
# which caught `ls site/data/` -- the site's own published JSON, small and meant to be read.
# A guard that fires on the output directory teaches you to ignore it.
# The class also stops at `&` and `;`, not just `|` and newline. Those chain commands on ONE
# line, so the newline fix did not reach them: a reader in the first command matched a path
# three commands later that was being WRITTEN, not read. Same failure as the newline one, one
# separator down, and it blocked the command that produced a gate's results.
BLOCKED_BASH = re.compile(
    r"\b(cat|head|tail|less)\b[^|&;\n]*(data/(raw|interim|processed)/|\.parquet|\.csv)")

# The escape hatch cannot be caught by the trap. `hub.inspect --head 5 <path>.parquet`
# matches the reader pattern -- `--head` contains `head` -- so without this the guard
# refuses the exact command it tells you to run. Summarizing is the approved path by
# definition, so a command that invokes it is allowed whatever else it mentions.
# This is guidance, not a sandbox: someone determined to `cat` a parquet can still append
# a mention of hub.inspect. Worth it to keep the recommended path unobstructed.
#
# `git commit` earns the same exemption for a different reason: a commit message that
# describes data work puts words like `data/processed` on the same line as an unrelated
# `| tail -1`, and no `git commit` invocation has ever read a parquet file. Writing about
# the guard should not trip the guard.
ALLOWED = re.compile(r"\bhub\.inspect\b|\bgit\s+(-c\s+\S+\s+)*commit\b")

try:
    ev = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool = ev.get("tool_name", "")
inp = ev.get("tool_input", {}) or {}

hit = None
if tool == "Read" and BLOCKED_READ.search(str(inp.get("file_path", ""))):
    hit = inp.get("file_path")
elif tool == "Bash":
    cmd = str(inp.get("command", ""))
    if BLOCKED_BASH.search(cmd) and not ALLOWED.search(cmd):
        hit = cmd

if hit:
    print(
        f"BLOCKED: {hit}\n"
        "Reading data files into context costs 40k+ tokens (see CLAUDE.md rule 1).\n"
        "Use a CLI that prints a summary instead, e.g.:\n"
        # `uv run`, not bare `python`: the modern-python PATH shim intercepts `python`,
        # so an agent that follows this hint verbatim would hit a second wall.
        "  uv run python -m hub.inspect <dataset> --schema\n"
        "  uv run python -m hub.inspect <dataset> --head 5 --cols a,b,c\n"
        "  uv run python -m hub.inspect <dataset> --nulls",
        file=sys.stderr,
    )
    sys.exit(2)
sys.exit(0)
