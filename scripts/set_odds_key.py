"""Move an API key from a scratch file into .env without it passing through a transcript.

Reads the key, writes it to .env, and shreds the scratch file. Prints only a length, never
the value -- the whole point is that the secret never appears anywhere it could be pasted,
logged or committed.

    uv run python scripts/set_odds_key.py ~/odds_key.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"
VAR = "ODDS_API_KEY"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: python {Path(__file__).name} <path-to-key-file>", file=sys.stderr)
        return 2
    src = Path(argv[1]).expanduser()
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 1

    key = src.read_text().strip().strip('"').strip("'").strip()
    if not key:
        print(f"{src} is empty", file=sys.stderr)
        return 1
    if any(c.isspace() for c in key):
        print("the key contains whitespace -- paste it on a single line with nothing else",
              file=sys.stderr)
        return 1

    lines = ENV.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == VAR:
            lines[i] = f"{VAR}={key}"
            break
    else:
        lines.append(f"{VAR}={key}")
    ENV.write_text("\n".join(lines) + "\n")
    os.chmod(ENV, 0o600)

    # Overwrite before unlinking rather than just deleting: a plain unlink leaves the bytes
    # on disk, and this file exists only to carry a secret a few inches.
    try:
        n = src.stat().st_size
        with src.open("wb") as fh:
            fh.write(b"\0" * n)
            fh.flush()
            os.fsync(fh.fileno())
        src.unlink()
        shredded = True
    except OSError:
        shredded = False

    print(f"  wrote {VAR} to .env ({len(key)} chars)")
    fate = "shredded and removed" if shredded else "COULD NOT be removed -- delete it yourself"
    print(f"  scratch file {fate}: {src}")
    print("  .env permissions set to 0600, and it is gitignored")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
