# Claude Desktop setup

**Short answer to "as a project?": no.** A claude.ai Project is a conversational workspace with
custom instructions and an uploaded knowledge base. It cannot execute code, touch local files, or
run scheduled tasks. Putting a repo in one gives you a chat about a snapshot of your code.

The repo belongs in the **Code tab**. A Project is still worth creating, but for the work that
isn't code. See the last section.

---

## Code tab: the repo

Three tabs in the desktop app: **Chat**, **Cowork**, **Code**. You want Code.

**Starting a session.** Four things before your first message:

| Setting | Pick |
|---|---|
| Environment | Local |
| Project folder | `football-hub` |
| Model | Sonnet by default, Opus for architecture |
| Permission mode | Plan for anything multi-file, then Accept edits |

The best-practice pattern is Plan first so Claude maps an approach before touching source, then
switch to Accept edits to execute. Set `permissions.defaultMode` in settings to make it stick.

**Worktrees are automatic here.** In the CLI you pass `--worktree`. In Desktop, every new
session gets its own isolated worktree for a git repo, stored under
`.claude/worktrees/`. So my earlier three-worktree plan is just three sessions:

```
session 1   main checkout   draft board + weekly slate
session 2   worktree        Track A ratings
session 3   worktree        Track C conformal + backtest
```

Cmd+N for a new session. Ctrl+Tab to cycle. Cmd-click a session in the sidebar to view two
side by side.

### Two gotchas that will bite you specifically

**1. `.worktreeinclude` (already added).** A worktree is a fresh checkout, so gitignored files
do not come along. Your ESPN cookies live in `.env`, which is gitignored. Without
`.worktreeinclude` listing `.env`, every parallel session silently loses them and the board
degrades to ECR-only mode with no obvious cause. The file is in the repo root now.

**2. Desktop does not inherit your full shell environment.** Launched from the Dock on macOS, it
reads your shell profile for `PATH` and a fixed set of Claude Code variables, but other exported
variables are not picked up. If you export anything in `~/.zshrc` that the pipeline needs, set it
in the local environment editor instead: environment dropdown → hover **Local** → gear icon.
Those are stored encrypted and apply to sessions and preview servers both.

### Preview pane

`.claude/launch.json` is configured to serve `site/` on port 8000, so the dashboard opens in the
Browser pane. Cmd+Shift+B toggles it.

`autoVerify` is off deliberately. It defaults on and screenshots after every edit, which makes
sense for frontend work and is pure waste here, since most of your edits are pipeline Python.
Ask for a preview when you want one.

### Token management in this UI

- **Usage ring** next to the model picker shows context usage for the session and plan usage
  across every Claude Code surface. Check it before starting anything heavy on a Wednesday.
- **Side chat** (Cmd+; or `/btw`) answers a question using the session's context without adding
  anything back to the main thread. Use it for "why is this failing" so the main session stays
  clean. Side chats are not saved to disk.
- **Summary view mode** (Ctrl+O to cycle) collapses the transcript when running several sessions.
- `/compact` before you hit auto-summarization, so you control what survives. The compact
  instructions in CLAUDE.md govern what gets kept.

### Plugins and skills

Everything already in the repo carries over. Desktop reads the same `CLAUDE.md`,
`.claude/settings.json`, hooks, and skills as the CLI, and you can run both at once on the same
project. The `+` button next to the prompt box opens Plugins, Connectors, and Slash commands.

`football-hub@football-hub-local` loads from `.claude/settings.json` automatically. Invoke its
skills as `/football-hub:weekly-slate`.

Install the official plugins through the plugin browser (`+` → Plugins → Add plugin) rather than
the terminal. Note that `enabledPlugins` in `.claude/settings.json` is what makes a plugin
available in cloud sessions, since the plugin browser does not reach them.

### Scheduled tasks

Desktop has scheduled tasks where the CLI has cron. Once the season starts, the Tuesday refit is
a candidate. Keep the Sunday live poller manual, since a scheduled task that fires while you are
out is exactly the failure the watchdog exists to catch.

---

## A Project, for everything that is not code

Create one called **Football Hub — research**. It is the right tool for the surrounding work,
which currently has nowhere to live.

**Custom instructions:** the objective ordering (win the league > understand the methods >
portfolio > market edge), the evenhandedness rule that gates fail into "stays a study" rather
than deletion, and a standing instruction to never propose a change that needs attention on a
Sunday.

**Project knowledge:** upload `docs/architecture.md`, `docs/modeling-research-action-items.md`,
`docs/track-record.md`, and `docs/cfbd-quota.md`. Not the code. The code is what the Code tab
reads live, and a stale uploaded copy is worse than none.

**Use it for:** weekly result reviews, deciding whether a gate passed, thinking through pool
strategy, drafting the write-ups you might publish later. All the reasoning that produces
decisions, none of the work that produces files.

**The split that matters:** the Code tab operates on the repo. The Project holds the thinking
about the repo. Conversations in a Project are also scoped separately, so research chatter does
not bleed into your coding sessions.
