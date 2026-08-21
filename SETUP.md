# Setup

Exact sequence. Every step has a check. If a check fails, stop there rather than continuing.

Steps 1 through 6 take about 45 minutes and end with a working draft board. Step 7 onward can
wait until after Sep 3.

---

## 0. Before you start

You need macOS, a browser logged into your ESPN league, and about 45 minutes.

Download `football-hub.zip`, then:

```bash
mkdir -p ~/code && cd ~/code
unzip ~/Downloads/football-hub.zip
cd football-hub
ls
```

**Check:** you see `SETUP.md`, `pyproject.toml`, `src/`, `docs/`, `plugins/`.

---

## 1. Toolchain

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
npm install -g pyright
brew install gh
gh auth login
```

`pyright` is the language-server binary for the `pyright-lsp` plugin in step 6. It is separate
from Pyrefly, your type checker, and you need both: one gives Claude live diagnostics after every
edit, the other gates the TDD hook.

**Check:**

```bash
uv --version && pyright --version && gh auth status
```

All three print without error.

---

## 2. Install and run the tests

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
```

**Check:** `46 passed`. If anything fails, stop and paste the output to Claude before continuing.

```bash
uv run pyrefly check
```

**Check:** completes in under a second. Type errors are fine to fix later. A crash is not.

---

## 3. Private repo

```bash
git init
git add -A
git commit -m "scaffold: draft board, contracts, TDD gate, plugin, ADRs"
gh repo create football-hub --private --source=. --push
```

Private is deliberate. Your leaguemates are a mixed room and this board exists to beat them. It
flips public Sep 4. See `docs/going-public.md`.

**Check:**

```bash
gh repo view --json visibility --jq .visibility     # PRIVATE
./scripts/preflight_public.sh                       # PASS
```

Run preflight now even though you are not flipping yet. It establishes that the history is clean
*before* you add secrets, so a later failure points at exactly which commit introduced one.

---

## 4. ESPN cookies

Cannot be automated. ESPN added recaptcha to login years ago, so cookies are the only path.

1. Log into your league at `fantasy.espn.com`.
2. Open DevTools with Cmd+Option+I. Go to **Application > Cookies > fantasy.espn.com**.
3. Copy two values:
   - `espn_s2`, 250+ characters
   - `SWID`, about 38 characters, **including the curly braces**
4. League ID comes from the URL: `leagueId=XXXXXXX`

```bash
cp .env.example .env
open -e .env        # paste all three, save, close
```

**Check:**

```bash
awk -F= '/ESPN_S2/{print "espn_s2 length:", length($2)}' .env
```

Prints 200 or more. A small number means you copied the wrong cookie. Go back to step 4.

`.env` is gitignored. Never paste these into a chat, an issue, or a commit.

---

## 5. Build the board

```bash
make draft
```

**Check:** replacement levels print, roughly `QB 19.8  RB 12.4  WR 9.8  TE 9.6`, followed by
regression candidates.

If the output says **"ECR-only mode"**, the cookies did not work. The board still runs but you
lose the ADP diff, which is the main edge. Return to step 4.

```bash
uv run python -m hub.draft.board --show-slots
```

**Check:** reports 12 teams, 3 WR, slot 3. If your league page disagrees, the API is right and
`conf/config.yaml` needs updating.

```bash
make serve      # http://localhost:8000, Ctrl+C to stop
```

---

## 6. Claude Desktop

Download from `claude.com/download`. Open it, sign in, click the **Code** tab.

Start a session:

| Field | Value |
|---|---|
| Environment | Local |
| Project folder | `~/code/football-hub` |
| Model | Sonnet |
| Permission mode | Plan |

Trust the folder when prompted.

**First message.** Paste this exactly:

```
Read CLAUDE.md, docs/architecture.md, and SETUP.md. Then confirm:
1. Which plugins are loaded and which skills they provide
2. That both hooks are active
3. What you understand my top three objectives to be
Do not change any files.
```

**Check:** it reports `football-hub@football-hub-local` with four skills, both hooks active, and
your objectives as (1) win the league and pools, (2) understand the methods, (3) portfolio.

Install the official plugins: click `+` next to the prompt box, choose **Plugins**, then
**Add plugin**.

- `pyright-lsp` — live type diagnostics after every edit
- `security-guidance` — public repo handling session cookies
- `commit-commands` — git workflow

Then run `/reload-plugins`, adding `--force` if it warns about the prompt cache.

**Check:** `/plugin list` shows four entries, three official plus `football-hub`.

---

## 7. Project board (optional, can wait until after the draft)

```bash
gh auth refresh -s project,read:project
./scripts/bootstrap_project.sh <your-username>/football-hub
```

Seeds four gated milestones and ten issues.

---

## 8. Research Project (optional, 5 minutes)

In the **Chat** tab, create a Project called *Football Hub — research*.

Custom instructions:

```
Objective order for this project, highest first:
1. Win my fantasy league and pools
2. Actually understand the modeling techniques
3. Portfolio artifact
4. Demonstrable market edge

A model that fails its gate becomes a documented study, never a deletion.
Never propose anything that requires my attention on a Sunday.
```

Project knowledge: upload `docs/architecture.md`, `docs/modeling-research-action-items.md`,
`docs/track-record.md`, and `docs/cfbd-quota.md`. Not the code. The Code tab reads that live, and
a stale uploaded copy is worse than none.

---

## Then what

| When | Do |
|---|---|
| Aug 22-24 | Read the board with your own eyes. An absurd name is a join bug, not an insight. |
| Aug 25-28 | Tune lambda: `uv run python -m hub.draft.board -m draft.projection_lambda=0.02,0.04,0.08,0.16,0.32` against a 2024 to 2025 holdout |
| Aug 29-31 | Build `hub.draft.live`, the draft-day poller |
| Sep 1-2 | Dry run against last year's draft. Find the bug now, not at pick 3. |
| **Sep 3** | **Draft.** Second monitor. `draft_mode()` tells you scarcity vs value. |
| Sep 4 | `make preflight`, flip public, enable Pages |

Still outstanding, and only you can get it: your pool's entry count, payout structure, and
whether rebuys exist. Five-minute question to your commissioner, and it is the last thing
blocking the survivor optimizer.
