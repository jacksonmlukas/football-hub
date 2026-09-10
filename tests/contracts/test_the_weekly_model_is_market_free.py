"""The **Weekly projection** reads no **Betting market** column, and nothing may add one.

Condition a player model on the market's implied team total, then aggregate those players
back into a team total, and the team total is the market's own number wearing this repo's
name. The comparison does not become biased -- it becomes *meaningless*, because the thing
being compared and the thing being compared against are the same number. It is standard
practice in the fantasy industry and the industry does not discuss it. Issue #212, finding
G7 of the research artifact of 2026-09-07.

**The tree is already clean, and by accident.** `own_spread` and `implied_total` appear in
`models/panel.py`, which builds them, and in `models/weekly_screen.py`, which screened them;
`models/weekly.py` touches neither. The reason on the record was not circularity: the joint
screen killed both as one finding wearing two hats at r = 0.83, since
`implied_total = total_line/2 + own_spread/2`. A correct outcome reached from unrelated
reasoning is exactly what a guard test is for -- the reasoning that produced it does not
constrain the next edit.

**Why a census and not a `# GUARD` block.** `tests/contracts/test_guards_are_load_bearing.py`
proves a guard by deleting it and watching the suite go red, which needs a *statement* to
delete. What is being asserted here is an **absence**: there is no refusal in `weekly.py` to
excise, and excising nothing proves nothing. So this is the other shape this repo uses --
`test_the_split_is_written_once` and
`tests/contracts/test_stage_columns_are_asked_of_the_report.py` -- an AST census over named
files with the survivors written down. The ticket names the second of those as the model.

**A rename must not satisfy it.** A list of column names in a test file is a list a future
author can walk away from by spelling the column differently, which is a guard in name only.
Two things stop that here:

* every name in `MARKET_COLUMNS` says which *declaration* it is read from, and
  `test_every_market_column_is_still_declared_where_this_file_says` fails the day one stops
  appearing there. A rename lands on that test before it lands anywhere else, and the fix is
  to re-point the entry -- at which point the scan covers the new spelling.
* `derived_from_the_line` walks `panel.game_context` and reports every column it *derives*
  from the schedule's market fields, seeded from the contract rather than from a literal. A
  ninth market column added to the Panel is in the scan the moment it is written, without
  anyone remembering this file exists.

**What is deliberately outside the constraint.** The **Consensus** prior is not the betting
market and does not contaminate a team total. `hub.models.weekly` regresses a thin sample
toward what a player's *preseason* FantasyPros ECR implies (`fit_consensus_prior`,
`consensus_target`), and `fit_shrink(target="market")` names that variant with the word this
repo reserves for three different things -- see the *Market signals* section of `CONTEXT.md`,
which distinguishes the **Draft market**, the **Consensus** and the **Betting market**. A
scan that read the word `market` would fire on `PURE_MARKET_K` and on every one of those
docstrings; this one reads column names off the AST, so the prose is free to discuss the
distinction and `test_the_consensus_prior_is_outside_the_constraint` holds it as a positive
statement rather than as an omission.

**What it costs, stated rather than discovered.** `MARKET_FREE` is three modules, not the
repo. `models/market.py`, `models/margin.py`, `models/ratings.py`, `models/coverage.py` and
`season/survivor.py` all read `close_spread` or `spread_line` and are meant to: they price
the market *as the benchmark*, and a benchmark is not laundered into a projection because
nothing aggregates it back out. The hole this leaves is a fourth module joining the
projection path without being added below, and the honest answer is that the scope only
ever under-covers -- adding to it strengthens the gate, which is the property a count of
allowed sites does not have.
"""
import ast
from pathlib import Path

import pytest

from hub import contracts
from hub.models import panel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"

# Each **Betting market** column, and the declaration it is read from. The value is what
# `test_every_market_column_is_still_declared_where_this_file_says` resolves, so an entry
# cannot claim a provenance it does not have and a rename cannot pass quietly.
MARKET_COLUMNS: dict[str, str] = {
    # The schedule's own two, which move -- `CONTEXT.md`'s *Snapshot* entry draws the
    # distinction between these and a dated capture. Everything the Panel offers is derived
    # from them.
    "spread_line": "contracts.SCHEDULES.ranges",
    "total_line": "contracts.SCHEDULES.ranges",
    # What the Panel makes of them, one row per (team, season, week). `implied_total` is the
    # market's own forecast of how many points this offence scores, which is precisely the
    # quantity a fantasy week is a share of -- and so precisely the one that launders.
    "own_spread": "panel.PRE_KICKOFF",
    "implied_total": "panel.PRE_KICKOFF",
    # The dated capture and its prices, from the odds snapshot. A projection could reach
    # past the Panel to the store, so the store's spelling is in the vocabulary too.
    "close_spread": "contracts.ODDS_SNAPSHOT.required",
    "close_total": "contracts.ODDS_SNAPSHOT.required",
    "spread_price": "contracts.ODDS_SNAPSHOT.required",
    "total_price": "contracts.ODDS_SNAPSHOT.required",
}

# Importing one of these is reaching for the betting market whatever it then names. Neither
# is a column, so the census above cannot see them, and `hub.models.market` in particular is
# one import away from being a feature rather than a benchmark.
MARKET_MODULES: tuple[str, ...] = ("hub.models.market", "hub.fetch.odds")

# The modules on the path from the Panel to a team-level aggregate, and why each must carry
# no market column. Three rather than one: the constraint is about what a team total is built
# out of, and a market term entering the distribution or the rebuild launders it exactly as
# one entering the mean would.
MARKET_FREE: dict[str, str] = {
    "models/weekly.py":
        "the Weekly projection itself -- the mean a team aggregate would sum. #212's "
        "subject",
    "models/components.py":
        "the rebuild the multiplier acts through: Usage counts to points, which is what "
        "makes the projection a number anything can add up",
    "models/predict.py":
        "the predictive distribution around that mean. A spread or a skew conditioned on "
        "the market launders it into the aggregate just as the centre would",
}

# Where the vocabulary is seeded from: the market fields the *schedule* publishes, read off
# the contract that declares them. `derived_from_the_line` grows the set from here, so the
# seed is the only thing written down and even that is checked against `hub.contracts`.
SEED_DECLARATION = "contracts.SCHEDULES.ranges"


def _declaration(name: str) -> set[str]:
    """The names one declaration declares, resolved from the string an entry records."""
    obj: object = {"contracts": contracts, "panel": panel}[name.split(".")[0]]
    for step in name.split(".")[1:]:
        obj = getattr(obj, step)
    return set(obj)  # type: ignore[call-overload]


def _strings(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _aliased(node: ast.AST) -> tuple[str, ast.expr] | None:
    """`expr.alias("x")` as (the column it binds, the expression behind it), else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "alias" and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return node.args[0].value, node.func.value
    return None


def derived_columns(fn: ast.AST, seed: set[str]) -> set[str]:
    """Every column `fn` derives from `seed`, whatever those are named.

    The auto-widening half. A column is market-derived when the expression it is aliased
    from mentions a market column, or mentions a local name that was itself assigned from
    one -- `spread = pl.col("spread_line") if home else -pl.col("spread_line")` is that
    second form and is how `own_spread` is written. Run to a fixed point, because
    `implied_total` is derived from `own_spread`, which is derived in turn.

    Takes the function node rather than a path so the control below can hand it source with
    a column spliced in and watch this find it, which is the only way to show that the scan
    widens rather than that it happens to agree with a list.
    """
    found: set[str] = set()
    tainted = set(seed)
    while True:
        before = len(tainted)
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _strings(node.value) & tainted:
                tainted |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            alias = _aliased(node)
            if alias is None:
                continue
            column, source = alias
            names = {n.id for n in ast.walk(source) if isinstance(n, ast.Name)}
            if (_strings(source) & tainted) or (names & tainted):
                found.add(column)
                tainted.add(column)
        if len(tainted) == before:
            return found


def derived_from_the_line(seed: set[str]) -> set[str]:
    """`derived_columns` against the Panel function that builds the market's columns."""
    return derived_columns(_function(SRC / "models" / "panel.py", "game_context"), seed)


def market_names() -> set[str]:
    """The whole vocabulary: what this file declares, plus what the Panel has since built."""
    return set(MARKET_COLUMNS) | derived_from_the_line(_declaration(SEED_DECLARATION)
                                                       & set(MARKET_COLUMNS))


def references(text: str, names: set[str]) -> list[str]:
    """Every reference to a market column or a market module in `text`, as reported lines.

    Read off the AST, never off the text, for the reason
    `test_guards_are_load_bearing.test_prose_naming_a_refusal_is_not_read_as_one` gives: a
    grep fires on the sentence explaining the constraint, and a gate that fires on its own
    explanation is a gate somebody deletes. `ast` tells `pl.col("implied_total")` from a
    paragraph about it, so the docstrings above are free to name what they forbid.

    Reported by line and by what was found, never as a count -- a report that says "1
    violation" is the number a person has to go and look up.
    """
    lines = text.splitlines()
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(text)):
        hit = None
        if isinstance(node, ast.Constant) and node.value in names:
            hit = f"the column {node.value!r}"
        elif isinstance(node, ast.Attribute) and node.attr in names:
            hit = f"the column {node.attr!r}, as an attribute"
        elif isinstance(node, ast.ImportFrom):
            if node.module in MARKET_MODULES:
                hit = f"an import from {node.module}"
            elif imported := {a.name for a in node.names} & names:
                hit = f"an import of {sorted(imported)}"
        elif isinstance(node, ast.Import) and (mods := {a.name for a in node.names}
                                               & set(MARKET_MODULES)):
            hit = f"an import of {sorted(mods)}"
        if hit is not None:
            found.append((getattr(node, "lineno", 0), hit))
    return [f"line {n}: {what} -- {lines[n - 1].strip()}" for n, what in sorted(set(found))]


# --- the premise, which is the half a list cannot keep -----------------------
#
# Everything below is vacuous if the vocabulary has gone stale, and a stale vocabulary passes
# exactly as loudly as a live one. That is the "passing tests, dead guard" shape of the week
# of 2026-09-04; these three make it fail instead.

@pytest.mark.parametrize(("column", "where"), sorted(MARKET_COLUMNS.items()))
def test_every_market_column_is_still_declared_where_this_file_says(column, where):
    """The anti-rename mechanism, and the reason the entries carry a provenance at all.

    Spell `implied_total` differently in `panel.game_context` and it leaves `PRE_KICKOFF`
    under that name; this goes red, the entry is re-pointed, and the scan covers the new
    spelling. Without it a rename silently empties the vocabulary and every assertion below
    passes over a model that now reads the market.
    """
    assert column in _declaration(where), (
        f"{column!r} is no longer declared in {where}, so the market vocabulary this file "
        f"scans for has gone stale by exactly one column -- and a scan that has gone stale "
        f"passes. Re-point the entry at wherever that quantity is declared now, or drop it "
        f"if the repo genuinely no longer carries it.")


def test_the_panel_still_derives_the_two_columns_that_prompted_this():
    """The auto-widening half, proved against the Panel it reads.

    If `derived_from_the_line` stops seeing `game_context` -- the function is renamed, the
    aliases are written some other way -- it returns the empty set and the vocabulary
    silently narrows to whatever is written down. So the denominator is asserted rather than
    assumed, the same way `test_the_watched_shapes_still_find_the_module_they_watch` does it.
    """
    derived = derived_from_the_line(_declaration(SEED_DECLARATION) & set(MARKET_COLUMNS))
    assert {"own_spread", "implied_total"} <= derived, (
        f"the scan of panel.game_context found {sorted(derived)}, which does not include "
        f"the two columns #212 is about. It has stopped reading the function that builds "
        f"them, so it would report nothing however many the Panel grows.")


def test_a_market_column_the_panel_grows_is_in_the_vocabulary_without_being_written_down():
    """What stops this file from being a list somebody has to remember to update.

    A ninth market column added to `game_context` is scanned for from the moment it is
    written. The probe is source handed to the scanner rather than an edit to the Panel,
    because the property being shown is that the *scanner* widens.
    """
    assert "moneyline_implied" not in market_names()
    src = '''
def game_context(seasons):
    spread = pl.col("spread_line") if home else -pl.col("spread_line")
    sides.append(s.select(spread.alias("own_spread"), pl.col("total_line"),
                          pl.col("roof").alias("roof")))
    return pl.concat(sides).with_columns(
        (pl.col("total_line") / 2 + pl.col("own_spread") / 2).alias("implied_total"),
        (pl.col("implied_total") * 0.42).alias("moneyline_implied"))
'''
    grown = derived_columns(ast.parse(src), {"spread_line", "total_line"})
    assert grown == {"own_spread", "implied_total", "moneyline_implied"}, grown
    assert "moneyline_implied" not in market_names(), "the probe leaked into the real scan"


# --- the constraint ----------------------------------------------------------

@pytest.mark.parametrize(("module", "why"), sorted(MARKET_FREE.items()))
def test_the_weekly_model_references_no_betting_market_column(module, why):
    """#212's first two criteria: red on a market column, green on the tree as it stands.

    The message says why the constraint exists rather than that it broke, because the next
    person to meet it will be adding a feature that measures well -- `implied_total` screened
    at +0.055 across five of five seasons -- and a message reading "forbidden column" invites
    deleting the test.
    """
    hits = references((SRC / module).read_text(), market_names())
    assert hits == [], (
        f"{module} now reads the betting market:\n  " + "\n  ".join(hits) + "\n"
        f"It may not, because it is {why}.\n"
        f"Conditioning a player projection on the market's implied team total and then "
        f"aggregating those players into a team total launders the market's own number into "
        f"what reads as an independent estimate -- the aggregate stops being a comparison at "
        f"all. The **Consensus** prior is not this and is deliberately allowed: see this "
        f"file's docstring and the *Market signals* section of CONTEXT.md.\n"
        f"If the answer really is that the projection should read the line, that is a "
        f"decision about what every team-level aggregate downstream means, and it belongs in "
        f"an ADR before it belongs in this module.")


def test_the_consensus_prior_is_outside_the_constraint():
    """#212's fourth criterion, as a statement rather than as an omission.

    `hub.models.weekly` regresses a thin sample toward what a player's preseason FantasyPros
    rank implies. That is the **Consensus** -- a fantasy market -- and it says nothing about
    how many points a defence will concede, so it cannot launder a team total. If the scan
    ever started reading it, this test says so in the same run rather than the constraint
    quietly widening into the thing the model actually does.
    """
    consensus = {"ecr", "preseason_ecr", "lead_days"}
    assert not consensus & market_names(), (
        f"{sorted(consensus & market_names())} is being scanned for as a betting-market "
        f"column. It is the Consensus, the model regresses toward it on purpose, and this "
        f"constraint is not about it.")
    weekly = (SRC / "models" / "weekly.py").read_text()
    assert "preseason_ecr" in _strings(ast.parse(weekly)), (
        "the Weekly projection no longer reads `preseason_ecr`, so the test above is "
        "carving out something the model does not do -- and the constraint's own scan has "
        "nothing to distinguish it from a file that mentions no column at all.")


def test_the_scan_sees_a_market_read_written_any_of_the_ordinary_ways():
    """The positive control. A census that silently matches nothing passes exactly as
    loudly as one that is right, which is what
    `tests/contracts/test_guards_are_load_bearing.py` was written after eight of.

    Four shapes that must be caught and three that must not: the two `market`-named things
    `weekly.py` genuinely contains, and prose naming the column.
    """
    src = '''
"""A docstring naming implied_total and close_spread, which is not a read of either."""
from hub.models.market import MarketBaseline
from hub.models.panel import PRE_KICKOFF

PURE_MARKET_K = 1e9

def project(df, coefs, *, target="market"):
    """Regresses toward what the preseason rank implies, the market-implied variant."""
    x = df.select(pl.col("implied_total"))
    y = df["own_spread"]
    return x, y, row.close_spread, PRE_KICKOFF
'''
    got = references(src, {"implied_total", "own_spread", "close_spread"})
    assert [g.split(" -- ")[0] for g in got] == [
        "line 3: an import from hub.models.market",
        "line 10: the column 'implied_total'",
        "line 11: the column 'own_spread'",
        "line 12: the column 'close_spread', as an attribute",
    ], got
