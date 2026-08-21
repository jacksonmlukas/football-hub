#!/usr/bin/env python3
"""Block the single most expensive mistake: pulling a data file into context.

CLAUDE.md asks. This enforces. Exit 2 blocks the tool call and hands the reason
back to Claude, which then routes through a summarizing CLI instead.
"""
import json, re, sys

BLOCKED_READ = re.compile(r"(data/(raw|interim|processed)/|\.parquet$|\.csv$)")
BLOCKED_BASH = re.compile(r"\b(cat|head|tail|less)\b[^|]*\b(data/|\.parquet|\.csv)")

try:
    ev = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool = ev.get("tool_name", "")
inp = ev.get("tool_input", {}) or {}

hit = None
if tool == "Read" and BLOCKED_READ.search(str(inp.get("file_path", ""))):
    hit = inp.get("file_path")
elif tool == "Bash" and BLOCKED_BASH.search(str(inp.get("command", ""))):
    hit = inp.get("command")

if hit:
    print(
        f"BLOCKED: {hit}\n"
        "Reading data files into context costs 40k+ tokens (see CLAUDE.md rule 1).\n"
        "Use a CLI that prints a summary instead, e.g.:\n"
        "  python -m hub.inspect <dataset> --schema\n"
        "  python -m hub.inspect <dataset> --head 5 --cols a,b,c",
        file=sys.stderr,
    )
    sys.exit(2)
sys.exit(0)
