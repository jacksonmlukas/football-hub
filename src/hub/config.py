"""Structured Hydra config.

Every number that changes a prediction is *covered* by `config_digest()`, which folds into the
model version so two runs with different lambdas can never be mistaken for the same model in
the track record. The point is not tidiness; it is that the track record's claim about which
model made which prediction has to be true.

Covered is not the same as living here, and the difference is deliberate:

  * **Settings** are choices -- lambda, alpha, the roster shape. They live in the dataclasses
    below, where Hydra can override them from the command line.
  * **Fitted constants** are measurements with confidence intervals and write-ups in `docs/`.
    They live next to their provenance in the prediction modules listed in `FITTED_MODULES`,
    and `config_digest` hashes them from there. Making them Hydra-overridable would let a
    command-line flag quietly replace a measurement with a preference.

So: if you find yourself typing a number into a module, ask which kind it is. A choice belongs
in this file. A measurement belongs beside the evidence for it, in a module this file hashes.

That question is the one thing no scan can answer for you, and issue #201 is what happens when
one tries. Three times the digest's coverage was settled by a property of *how a line is
written* rather than of what it means -- whether the name was capitalised, whether the value
was a float, whether a re-export happened to be a number -- and each time a constant either
identified a model version it had nothing to do with, or failed to identify one it decided.
`fitted_constants` no longer guesses: everything a registered module assigns is covered, and
anything that should not be says so by name in `NOT_IN_DIGEST`, with the argument attached.

There is a third kind, and it is deliberately *not* covered. The bytes a run was scored
against are not part of the model that scored them, and they move on their own -- so
`data_digest()` sits beside `config_digest()` and never inside it. `data_digest` argues why
at the point of the decision.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import polars as pl
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from hub import declare, paths
from hub.declare import chosen


@dataclass
class RosterConfig:
    """The league's shape. The one declaration of it.

    It used to be five: this class, `board.SLOTS`, `season.STARTERS`, `evaluate.STARTERS`,
    and a bare `< 7` inside `optimize._need_score` that was `rb + wr + te + flex` worked out
    by hand. Four of the five agreed, which is the problem -- nothing made them agree, so
    the first commissioner change would have moved some and not others.
    """
    teams: int = 12
    slot: int = 3                       # your snake position
    rounds: int = 16
    qb: int = 1
    rb: int = 2
    wr: int = 3                         # three-WR league, not the ESPN default
    te: int = 1
    flex: int = 1
    # Who may fill the flex. **Eligibility is a league rule and stays here**; how the flex
    # actually gets *used* is not, and left this class on 2026-09-10 for
    # `hub.draft.board.FLEX_SHARES` (#184). The two used to sit adjacent as
    # `flex_from` and `flex_rb/wr/te`, separated by a comment saying they were different
    # kinds of thing -- which is the arrangement that let three replacement levels coexist:
    # a quantity nobody derived, wearing a Hydra override knob, next to one the commissioner
    # owns. ADR-0006 puts a number that changes a prediction beside its provenance and out of
    # reach of the command line, and the shares change every VOR at every flex-eligible
    # position. Eligibility genuinely is a choice, so it genuinely belongs here.
    flex_from: list[str] = field(default_factory=lambda: ["RB", "WR", "TE"])


# The season being drafted for. One owner, because it was five: `board.SEASON_AHEAD`,
# `espn.league_settings`'s `League(year=...)`, `state.sync_from_espn`'s `year or 2026`,
# `playoff_sos`'s default, and `publish --season`'s default. Nothing made them agree, and
# two of them sat on the same call path: `player_market(season=...)` filtered stats by a
# season the League object had never been asked for, so passing 2027 fetched the 2026 league,
# matched no stat block, and returned proj_ppg as all-null with no error.
SEASON_AHEAD = 2026

# The last completed season -- the one every model is *fitted* on, as against the one being
# drafted for. Split out on 2026-08-27 because only half the SEASON_AHEAD fix had landed:
# `playoff_sos` read `SEASON_AHEAD` from here and then carried `dvp_season: int = 2025` as a
# literal beside it, so a rollover had to move two numbers in lockstep or the defence-adjusted
# ratios would silently come from a two-year-stale season. Six more sites said 2025 by hand.
SEASON_COMPLETED = SEASON_AHEAD - 1

# How long a fantasy regular season is, and the weeks it is played over as a tuple, derived
# rather than restated. `tuple(range(1, 15))` was written out three times -- in `weekly_screen`,
# in `weekly_gate` and here -- a literal 15 in three files for something with one owner. It sits
# in `config` rather than in `draft.season` because `models/` may not reach into `draft/`
# (`test_models_does_not_reach_into_draft`), and the screen and the Weekly projection both need
# it. Weeks 15-17 are the playoffs, reported apart rather than pooled; week 18 is meaningless in
# a league that ends at 17.
REG_SEASON_WEEKS = chosen(14)
FANTASY_WEEKS: tuple[int, ...] = tuple(range(1, REG_SEASON_WEEKS + 1))


@dataclass
class DraftConfig:
    # Fraction of the room drafting off ESPN's board. Fit from league history where possible.
    espn_weight: float = 0.5
    # Consensus adjustment strength. Multiplicative, so z=1 moves ~8% up the board.
    # 0.0 by evidence, not by default. Six holdouts (2019->20 .. 2024->25): five of six
    # negative at 0.08, and at 0.32 all six are negative at t = -6.83. No lambda is
    # significantly positive under any grouping. See docs/lambda-sweep.md.
    projection_lambda: float = 0.0
    z_clip: float = 3.0
    availability_sims: int = 5000
    # A pick with a longer wait than this after it is in "scarcity" mode.
    long_wait_threshold: int = 10
    # How far a fitted correction may move a player from his ADP, as a fraction of that ADP.
    # A *choice*, not a measurement, so it lives here rather than beside a coefficient --
    # ADR-0006's distinction. Fixed at 0.20 before anything was fitted: about one round at
    # ADP 60, and 0.6 picks at ADP 3 where consensus is tightest. See ADR-0011.
    correction_clamp_frac: float = 0.20
    # Ridge penalty on the weeks 15-17 strength-of-schedule adjustment (#180), in games: one
    # game of shrinkage is the smallest that keeps an early-season system determined without
    # asserting a prior stronger than the evidence. A *choice*, so it lives here and is
    # covered by `config_digest`; `playoff_sos.RIDGE_PENALTIES` is the range the sensitivity
    # sweeps, and the measured stability across it is what licenses this as a default rather
    # than a decision -- see docs/decisions.md. `None` is the unadjusted metric the published
    # ranking was measured on.
    sos_ridge: float | None = 1.0


@dataclass
class ModelConfig:
    name: str = "market"
    seed: int = 0
    conformal_alpha: float = 0.2
    min_calibration_points: int = 20
    calibration_window_weeks: int = 6


@dataclass
class QuotaConfig:
    cfbd_monthly_budget: int = 1000
    cfbd_weekly_reserve: int = 40
    odds_monthly_credits: int = 500


@dataclass
class PollConfig:
    """Tiered cadence. See hub.fetch.espn.poll for why this shape."""
    scoreboard_interval: int = 45       # cheap: one request per league, all games
    summary_every_n_ticks: int = 4      # expensive: one request per game of interest
    max_games_of_interest: int = 12
    stale_after: int = 600              # watchdog threshold


# Which of `PoolConfig`'s rules nobody has confirmed with the commissioner. Named here, beside
# the settings themselves, so the caveat the survivor panel publishes cannot drift from the
# thing it is a caveat about -- a hand-written list in the publisher would be a second copy
# free to go stale the moment one of these is settled.
#
# `co_survivor_rule` and `co_elimination_rule` are the ones that matter: between them they
# decide what the pot pays at either end a pool can reach -- entries outlasting the final
# week, or the last entries standing going out together -- so every dollar figure on that
# panel is conditional on both.
#
# `playoff_continuation` left this tuple with the field it named (#160). It was an unconfirmed
# rule that reached no figure -- nothing read it, and nothing could: the grid this pool is
# simulated on is `schedule.priced_games`, which is the regular season, so a pool playing on
# past week 18 has no week 19 to be planned over. A caveat naming a rule that decides nothing
# is not a caveat, it is noise inside one, and it made the two that do decide something
# cheaper to read past.
UNCONFIRMED_POOL_RULES: tuple[str, ...] = ("co_survivor_rule", "co_elimination_rule",
                                           "buyback_cap")


@dataclass
class PoolConfig:
    """The survivor pool's rules, as the commissioner states them.

    Settings, not measurements: a rule is what someone decided, so a correction is a re-run
    rather than an edit. That is the whole reason these are here and not constants in
    `hub.season.survivor` -- three of them are still provisional, and a provisional number
    baked into a module is one nobody re-reads.

    **Confirmed** for 2026: winner-take-all at $20, buybacks at $20 running *through* week 6,
    and a buyback that restores the entry's used-team ledger rather than clearing it -- so
    re-entering late is worth less than re-entering early, because the teams are already
    spent. **Provisional**: the cap of four, and whether it counts per entry or per person.
    **Unconfirmed**: what happens when more than one entry survives, what happens when the
    last entries standing all go out in the same week, and whether the pool plays on past
    week 18. Each of those moves every dollar figure, because every dollar figure divides a
    pot among whoever the rule says it belongs to.

    `double_pick_weeks` is a tuple and not a set. `config_digest` builds a structured config
    and OmegaConf rejects a `set` annotation outright -- which is a startup error, not a
    survivor-path error, so the wrong type here would take down the board.

    **One field here is not a commissioner rule**, and it is called out rather than filed
    quietly among the ones that are. `field_concentration` is an assumption about how the
    *rivals* behave, which no commissioner states and nobody in this pool can observe: see
    its comment below. It lives here because #152 settled that it must be a `PoolConfig`
    field -- #160 requires `pool_digest` to cover it, which presupposes one -- and because
    ADR-0006 sends only *fitted* constants out to live beside their provenance. A stated
    assumption is a choice, and a choice belongs in a config.
    """
    entry_fee: float = 20.0
    buyback_fee: float = 20.0
    buyback_cap: int = 4                     # provisional; 0 means no buybacks, not unlimited
    buyback_cutoff_week: int = 6             # inclusive -- buybacks run *through* this week
    buyback_restores_ledger: bool = True     # confirmed: used teams survive a re-entry
    double_pick_weeks: tuple[int, ...] = (13, 14, 15, 16, 17, 18)
    field_size: int = 21
    co_survivor_rule: str = "split"          # unconfirmed: split | rollover | tiebreak
    # What the pot pays when the last entries standing all go out in the same week -- the
    # other way a survivor pool ends, and under a field that crowds onto chalk the *usual*
    # way. Same closed set as `co_survivor_rule`, read by `hub.season.pool.trial_share`:
    # `split` divides the pot among the entries eliminated last, `rollover` pays nobody
    # unless exactly one entry outlasted the rest. Until #157 this state was priced at zero
    # by omission, which is not a rule any pool has. An entry that outlives the whole field
    # and then loses is alone in this state and takes the pot under every spelling.
    co_elimination_rule: str = "split"       # unconfirmed: split | rollover | tiebreak
    # Three fields left this class on 2026-09-10 (#160), each because nothing read it and
    # nothing could: `tie_eliminates` names an outcome the simulation never draws -- a game is
    # a win or a loss off `win_prob`, with no tie state -- `max_entries_per_person` caps a
    # quantity the field model does not carry, since it simulates `field_size` entries and
    # not the people behind them, and `playoff_continuation` planned over weeks the priced
    # grid does not contain. A knob that changes nothing is worse than no knob: a reader takes
    # it for a lever, and a caveat naming it makes the caveats that do decide something cheaper
    # to read past. Each returns the day the simulation can honour it.
    # How hard the simulated field crowds onto the week's best team: an exponent applied to
    # `win_prob` before `hub.season.pool._pick` samples a rival's team. 1.0 is sampling
    # proportional to the raw probability, which is what this pool has always simulated and
    # which puts about a sixteenth of the field on the chalk team; real survivor fields
    # concentrate several times that, and concentration is what makes a field die *together*.
    # 0.0 is a field picking uniformly among the teams it may still take.
    #
    # **Stated, never fitted, and 1.0 is not a measurement.** No pick-popularity data is
    # fetched anywhere under `hub.fetch`, and under Hidden Picks nobody can observe this
    # pool's rivals before a deadline -- so there is nothing to fit against and a number
    # chosen here would be an invention wearing a measurement's clothes. 1.0 is the value
    # that reproduces the behaviour already in the tree, so adopting the knob changes no
    # published figure until somebody moves it. The deliverable is `pool.sensitivity`, a
    # sweep that reports what the pool's lifetime and our share do across the axis, per
    # ADR-0024: measure the alternatives side by side rather than pick one.
    field_concentration: float = 1.0


@dataclass
class HubConfig:
    roster: RosterConfig = field(default_factory=RosterConfig)
    draft: DraftConfig = field(default_factory=DraftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    poll: PollConfig = field(default_factory=PollConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)


cs = ConfigStore.instance()
cs.store(name="hub_config", node=HubConfig)

# Where the overrides live. `hub.paths` is the one declaration of `ROOT` and imports nothing
# but `pathlib`, so taking the directory from it costs nothing and keeps the eighth copy of
# `parents[2]` from being written here.
CONF_DIR = paths.ROOT / "conf"


def resolved_config() -> HubConfig:
    """The configuration a run operates under: the dataclasses with `conf/` applied.

    The one thing `HubConfig()` is not. ADR-0004 says the digest folded into every model
    version is over the *resolved* config, and a bare `HubConfig()` is the resolved config
    only for as long as `conf/config.yaml` overrides nothing that diverges from a default --
    which is true today (both digests are `eb32dd45`, re-measured 2026-09-10) and is a
    coincidence rather than a property. The first divergent override would have a gate print
    a model version for a model nobody ran: provenance present in the schema and absent in
    the data, which is the defect `models/ratings.py` records from when `cfg_digest`
    defaulted to `"default"`.

    Composed through Hydra rather than by merging the YAML by hand, because "the way a run
    resolves one" is the claim being made, and a second resolution path would be a second
    answer to defend. `defaults` and the `hydra` block are consumed by the composer, so what
    comes back is the structured schema and nothing else; `to_object` makes it a real
    `HubConfig`, so a caller gets the same type either way and a type checker still sees the
    fields rather than the `Any` a bare `DictConfig` would hand it.

    Degrades to the defaults when `conf/` is absent -- an installed wheel, a partial
    checkout -- because a missing override file means "nothing to override" and a fetch
    that cannot print provenance is worse than one that prints the defaults it ran under.
    A `conf/` that exists and does not compose is *not* degraded past: that is the typo
    ADR-0004 chose structured configs to turn into a startup error, and swallowing it would
    stamp rows with a digest for a configuration the run never had.
    """
    if not CONF_DIR.is_dir():
        return HubConfig()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return cast(HubConfig, OmegaConf.to_object(compose(config_name="config")))


# Where the fitted constants are, and what decides that a number identifies a model version.
#
# A **setting** is a choice and belongs in a dataclass above, where Hydra can override it
# from the command line. A **fitted constant** is a measurement with a confidence interval
# and a write-up in `docs/`, and it lives beside that provenance in the module that reads
# it (ADR-0006). Making it Hydra-overridable would invite `draft.talent_cv=0.9` on a command
# line -- silently replacing a measurement with a preference, the exact failure the digest
# exists to catch -- and it has 30 lines of provenance attached, which belongs next to the
# number and not in a config schema.
#
# The digest's claim -- "everything that can change a prediction" -- has to be true, and
# for a while it was not: refitting TALENT_CV from 0.35 to 0.42 changed every prediction in
# the repo and left the digest untouched. What decided coverage after that was a list of
# modules here, then a second list of individual names, then a third list of exclusions,
# then a module-level opt-out string in twenty-six modules (#187, #199, #201, #217) -- four
# mechanisms a module author had to learn to declare one number, and three separate
# escapes reached production through the seams between them.
#
# **Since #253 the constant declares its own coverage where it is written**, through one
# of `hub.declare`'s three spellings -- `fitted(v)`, `chosen(v)`, `not_an_input(v, why)`
# -- and this module derives the digest by walking those declarations. There is no list
# here to keep in step with the tree. `tests/contracts/test_every_number_is_declared.py`
# holds the other half: a module-level number nobody declared is refused by name, so the
# question is asked once per number and cannot be skipped by a module never having been
# registered.
#
# What the spellings mean is `hub.declare`'s docstring; what matters here is that `fitted`
# and `chosen` are both covered -- coverage is owed by anything that changes a prediction,
# measured or not (`FLEX_SHARES`, #184; the shape of a random draw, #201; the nfeloqb
# pin, #271) -- and `not_an_input` is the one way out, carrying its argument.


def _canonical(v: Any) -> str:
    """Deterministic text for a constant, whatever shape it has.

    Sorted by `repr` of the key rather than the key itself, because `TEAMMATE_RHO` is keyed
    by tuples and a plain sort across mixed key types raises.
    """
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: repr(kv[0]))
        return "{" + ",".join(f"{k!r}:{_canonical(x)}" for k, x in items) + "}"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(repr(x) for x in v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canonical(x) for x in v) + "]"
    if isinstance(v, float):
        # repr() of a float round-trips exactly, so a refit that moves the last bit still
        # moves the digest.
        return repr(v)
    return repr(v)


def _canonicalisable(v: Any) -> bool:
    """Whether `_canonical` can turn this value into text that means the same thing twice.

    The types `_canonical` handles, all the way down. A `numpy` array or an open file handle
    is not one of them: `repr` of the first is truncated and `repr` of the second carries a
    memory address, so either would make a model version depend on where an object happened
    to be allocated.
    """
    if isinstance(v, (bool, int, float, str, bytes, type(None))):
        return True
    if isinstance(v, dict):
        return all(_canonicalisable(k) and _canonicalisable(x) for k, x in v.items())
    if isinstance(v, (set, frozenset, list, tuple)):
        return all(_canonicalisable(x) for x in v)
    return False


def fitted_constants() -> dict[str, Any]:
    """Every covered constant, as {"stem.NAME": live value}: the declarations that said
    `fitted` or `chosen`, read off their modules now.

    Live, not captured at the declaration, so the value a module holds *now* is what the
    digest sees -- a test that rebinds `predict.TALENT_CV` moves it, as it always did. A
    value the digest cannot canonicalise raises rather than dropping out quietly, because
    dropping out quietly is how #187, #199 and #201 each happened.
    """
    out: dict[str, Any] = {}
    for key, v in declare.covered().items():
        if not _canonicalisable(v):
            raise RuntimeError(
                f"{key} is declared covered, so it identifies a model version, but a "
                f"{type(v).__name__} has no text `_canonical` can hash stably. Either give it "
                f"a shape the digest can read, or declare it `not_an_input` with why a "
                f"prediction cannot reach it. Skipping it silently is what #187, #199 and "
                f"#201 each were.")
        out[key] = v
    return out


def fitted_digest(constants: Mapping[str, Any] | None = None) -> str:
    """Stable 8-char hash of the fitted constants alone.

    `constants` is for asking what the digest *would* be -- a test that wants to know that
    one constant moves the version passes `{**fitted_constants(), "predict.TALENT_CV": 0.35}`
    rather than rebinding a module attribute. Absent, the constants are read live.
    """
    have = fitted_constants() if constants is None else constants
    text = "\n".join(f"{k}={_canonical(v)}" for k, v in sorted(have.items()))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def config_digest(cfg: HubConfig | object, *,
                  exclude: tuple[str, ...] = ("poll", "quota", "pool")) -> str:
    """Stable 8-char hash of everything that can change a prediction.

    Both halves of that: the settings in `cfg`, and the fitted constants in
    `FITTED_MODULES`. A model version that moved when you changed `conformal_alpha` but not
    when you refit `TALENT_CV` was recording the less important of the two.

    Operational settings are excluded on purpose: changing the poll interval must not
    invalidate a model version, or every version bump becomes meaningless noise.

    `pool` is excluded for a sharper version of the same reason. This digest is stamped on
    every prediction row and folded into `FitSpec`, so covering the survivor pool's rules
    would mean confirming the buyback cap invalidates every cached NFL fit and issues a new
    model version for the weekly prediction and the draft board -- neither of which can read
    a pool rule. `pool_digest` distinguishes two survivor runs instead, where the distinction
    belongs.
    """
    d = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
    for k in exclude:
        d.pop(k, None)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256((canonical + fitted_digest()).encode()).hexdigest()[:8]


class DataPin(Protocol):
    """What identifies one loaded source, structurally.

    Read-only on purpose, so `hub.fetch.nflverse.Pin` -- a frozen dataclass -- satisfies it,
    and stated here rather than imported so no digest in this module has to reach a data
    layer to be computed. Every digest here answers from its arguments alone and can be
    computed from an empty checkout; making one import a fetch layer would be the wrong
    shape and a circular import besides, since `hub.fetch.nflverse` reads `SEASON_COMPLETED`
    from here. (`resolved_config` does read one file -- `conf/config.yaml`, and degrades to
    the defaults without it -- because resolving a config is the one job here that is about
    what is on disk. It is not on any digest's path: `config_digest` hashes the config it is
    handed.)
    """

    @property
    def source(self) -> str: ...

    @property
    def as_of(self) -> str | None: ...

    @property
    def digest(self) -> str: ...


def pin_fold(source: str, as_of: str | None, digest: str) -> str:
    """What one pin hashes as: content, folded with the source name and the as-of.

    The one statement of it. `nflverse.pin_digest` builds a pin's own digest from a frame,
    and `data_digest` below folds a set of pins into a run's; both hash this exact form, and
    until now each wrote it out with a comment saying the other matched. A comment is not a
    mechanism -- the two would have gone on agreeing right up until one of them was edited,
    and the failure would be a digest that no longer named the thing it was compared against.

    All three parts, and only these three. Hashing the labels alone would be invariant to
    exactly the drift the pins exist to catch -- two runs at one as-of that fetched different
    bytes would agree. Hashing content alone would make an archive that has not moved between
    two pins indistinguishable, and a pin is a claim about a date and a source as well as
    about rows. `pinned_at` is deliberately absent: it is a wall-clock stamp that moves on
    every refetch, so folding it in would report drift on rows that had not changed.
    """
    return f"{source}\n{as_of or ''}\n{digest}"


# What `data_digest` answers for a run that pinned nothing. Not a hash: sixteen modules
# besides `hub.fetch.nflverse` itself still `import nflreadpy` and reach the archive directly
# (counted 2026-09-05) -- `models/panel.py` at eight call sites on its own -- so the
# unpinned run is the common case until U2 of
# docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md routes them. A
# plausible-looking eight hex characters for that case would be provenance present in the
# schema and absent in the data, which is the defect `models/ratings.py` records when
# `cfg_digest` defaulted to "default" for every run under every configuration.
#
# So it shares a *width* with a digest and nothing else, and that is the whole of the answer
# to "should the sentinel and an eight-character digest share a shape". Eight characters so
# the three line up where they are printed together; not eight *hex* characters, so a reader
# who did not write this can tell which is which without being told -- `unpinned` is not a
# hexadecimal number, and `test_config.py` holds that line rather than trusting the word to
# stay unhexish through a rename. The shared return type is the same trade: a caller printing
# provenance gets one column of strings and never has to branch, and the only thing that
# would make a `None` better here is a caller that wants to *ask*, which none does -- the
# three digests are printed and compared, and a comparison against this sentinel means
# "these runs pinned nothing", which is true and is what it says.
UNPINNED = "unpinned"


@dataclass(frozen=True)
class UnpinnedRead:
    """A read a run made that names no bytes, and which therefore unpins the whole digest.

    The sentinel above answers for a run that pinned *nothing*. The case it did not answer
    for is the partial one: a cache hit on an entry written before pinning existed, or one
    whose sidecar write was interrupted, contributed nothing at all -- so a run that read
    three sources with two pinned produced a digest over two. Eight hex characters that look
    entirely legitimate, naming the wrong set of bytes.

    That is worse than the total case rather than a smaller version of it. A run that pinned
    nothing says `unpinned` and a reader knows where they stand; a run that pinned two of
    three says nothing at all, and the digest compares equal to a later run that read only
    those two. It is the failure the sentinel was designed to prevent, arriving through the
    one door it left open.

    So a read with no pin is *recorded*, carrying this as its digest, and `data_digest` below
    turns the whole run's answer into the sentinel when it sees one. Not a `Pin` with a blank
    digest: a `Pin` is what was written beside a cache entry, and one this process invented
    would be indistinguishable from a record that came off disk. This type says what it is,
    and it is the only thing that satisfies `DataPin` without naming bytes.

    `as_of` is carried even though it does not reach the digest, because the reason a run went
    unpinned is the first thing anyone asks and the answer is usually which load it was.
    """

    source: str
    as_of: str | None = None
    digest: str = UNPINNED


def data_digest(pins: Iterable[DataPin]) -> str:
    """Stable 8-char hash of the data a run actually loaded, or `UNPINNED` for none.

    Deliberately a *sibling* of `config_digest` and not an ingredient of it. The two answer
    different questions and only one of them belongs in a model version:

      * `config_digest` answers "which model made this row". ADR-0006 requires that it move
        when, and only when, a setting or a fitted constant moves.
      * this answers "which bytes was it scored against". nflverse revises `player_stats` and
        `pbp` in place -- `nflverse._write_by_week` says so where it passes `replace=True` --
        so this moves on a Tuesday refetch with no line of code and no fitted constant having
        changed.

    Folding the second into the first would issue a new model version every time the archive
    was refetched. That is the mirror image of the failure ADR-0006 was written for: there,
    refitting TALENT_CV left the version byte-identical, so the record claimed one model had
    made both sets of rows; here the version would move while the model stood still, and it
    would stop discriminating hyperparameters, which is the one job ADR-0004 gave it. It would
    also make the version depend on the state of a cache directory, so a fresh clone would
    report a different model for identical code.

    So a gate prints both, and a moved archive shows up as a changed *data* digest beside an
    unchanged model version -- which is the distinction a reader needs to make.

    Each pin folds through `pin_fold`, which is the same call `nflverse.pin_digest` makes and
    argues there for what goes in and what stays out. Sorted and de-duplicated, so the answer
    does not depend on which source a gate happened to load first, or on one archive being
    reached through two call sites.

    All of the bytes, or none of them. One `UnpinnedRead` in the set unpins the answer, so a
    run that read three sources and pinned two says `unpinned` rather than publishing a hash
    of the two it managed. A digest over a subset is not a weaker claim than a digest over the
    whole; it is a false one, because it compares equal to a run that read only that subset.
    `UnpinnedRead` argues the case above.
    """
    read = list(pins)
    # GUARD partial-pinning-unpins-the-run [unit/test_config.py]: a read that names no bytes
    # takes the whole digest with it rather than dropping out of it
    if any(p.digest == UNPINNED for p in read):
        return UNPINNED
    # /GUARD
    rows = sorted({pin_fold(p.source, p.as_of, p.digest) for p in read})
    if not rows:
        return UNPINNED
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:8]


# What `commit()` answers when it cannot reach a git tree: an installed wheel, a tarball, a
# container with no `.git`. Same trade as `UNPINNED` -- eight characters so the stamps line up
# where they are printed together, and not eight *hex* characters, so a reader can tell a
# sentinel from a SHA without being told.
NO_COMMIT = "nocommit"


def commit() -> str:
    """The commit this code is running from, or `NO_COMMIT`, or a SHA marked dirty.

    `docs/track-record.md` rule 1 counts a prediction only if its commit predates kickoff, and
    `docs/gate-power.md` records an effect moving 8.07 points across 270 commits at an
    identical data digest with **no owning commit** -- because the runs that produced those
    figures recorded which *bytes* they read and never which *code* read them. Two of the three
    published magnitudes in that table cannot be attributed to a tree at all. This is the
    missing third of a run's identity: config, data, and the code.

    **A dirty tree returns `<sha>-dirty` and that is deliberate.** A SHA claims "this tree is
    that commit"; a tree with uncommitted changes is not, and printing the SHA alone would be
    the false-provenance failure `data_digest` argues against one layer up -- a stamp that
    looks entirely legitimate and names the wrong thing. It costs the eight-character
    alignment, and honest is worth more than aligned.

    Degrades rather than raises, per the repo's standing rule: a gate that dies because it
    could not find git is a gate that stops being run.
    """
    import subprocess
    try:
        run = ["git", "-C", str(paths.ROOT)]
        sha = subprocess.run([*run, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()[:8]
        dirty = subprocess.run([*run, "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return NO_COMMIT
    # No `if not sha` guard: `rev-parse` under `check=True` either prints a SHA or raises, so
    # a guard for the empty string would be a statement nothing can reach -- and this module's
    # coverage floor is zero, which is the mechanism that says so rather than a preference.
    return f"{sha}-dirty" if dirty else sha


def frame_digest(frame: pl.DataFrame) -> str:
    """Stable 8-char hash of a frame a gate was measured on: its rows and its column set.

    `data_digest` is a digest of the right object one layer too high. It covers the upstream
    source *bytes*, and what `backtest.compare` is handed is a **Board** -- built from those
    bytes by `board_as_of`, through joins, an as-of boundary, a `MIN_GAMES` filter and an xFP
    imputation, any of which can change membership without a single source byte moving. Three
    runs in `docs/gate-power.md` carry the digest `621cb5dd` and were played on frames nobody
    can now identify.

    **Row order is in the digest, not sorted out of it, and that is the point rather than an
    oversight.** Both stochastic quantities in this harness are drawn as arrays ordered by
    Board row -- `optimize.simulate_remaining_draft` draws one pick-noise normal per row, and
    `season.simulate_weeks` draws `(n_sims, weeks, rostered union)` over the rostered players
    in Board order (#260; it was every row until then). Position in the frame is therefore
    what pairs a player with his draw, so two frames holding the same players in a different
    order are two different experiments and must not compare equal.

    The column set is in it for the reason `correction_report` shows: which columns a frame
    carries decides which code path runs on it, so a frame that lost `adp` is a different frame
    even where every retained value is identical.

    Cell by cell through `repr`, which is round-trippable for floats and keeps a null distinct
    from the string that spells it. Not `hash_rows`, whose stability is a promise polars makes
    to itself across versions and not one to a digest that a test pins.
    """
    cols = sorted(frame.columns)
    lines = ["\x1f".join(cols)]
    lines += ["\x1f".join(repr(v) for v in row)
              for row in frame.select(cols).iter_rows()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:8]


def frames_digest(frames: Mapping[Any, pl.DataFrame]) -> str:
    """One digest over every frame a run played, keyed by season.

    Folded here rather than at the gate, so that the answer to "which Boards was this run
    measured on" is not re-derived once per harness -- the shape `experiment.gate` exists to
    prevent one level up, where three modules each remembered ADR-0019 and two of them
    remembered it wrong.

    Sorted by key, so the answer does not depend on the order a caller happened to build the
    mapping in; the key is folded in beside its digest, so two runs over the same frames
    assigned to different seasons do not compare equal.
    """
    rows = sorted(f"{k}={frame_digest(v)}" for k, v in frames.items())
    if not rows:
        return NO_FRAMES
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:8]


# What `frames_digest` answers for a run that played no frames at all. A hash of the empty
# string is eight legitimate-looking hex characters that compare equal across every such run
# and tell a reader nothing, which is the case the sentinels above exist for.
NO_FRAMES = "noframes"


def digests(cfg: HubConfig | object, pins: Iterable[DataPin]) -> dict[str, str]:
    """The three digests that identify a run, as {"cfg", "fitted", "data"}.

    One call, so that no gate decides for itself which of the three name it -- and so the
    "beside, not inside" choice above is made in one place rather than re-argued at each
    output. `cfg` and `fitted` describe the code; `data` describes what the code read.
    """
    return {"cfg": config_digest(cfg), "fitted": fitted_digest(), "data": data_digest(pins)}


def pool_digest(cfg: PoolConfig) -> str:
    """Stable 8-char hash of the survivor pool's rules.

    Beside `config_digest`, never inside it. Two survivor runs under different pool rules are
    different runs and their outputs must say so; two *model* runs under different pool rules
    are the same model, and saying otherwise would move a version for a number the model
    cannot read.

    Not folded into `digests()` either, for the same reason stated from the other side: that
    helper answers "which run produced this row" for every gate in the repo, and a draft
    board has no pool. The survivor artifact and the decision journal call this one directly.
    """
    d = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    canonical = OmegaConf.to_yaml(OmegaConf.create(d), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def starters(cfg: RosterConfig) -> dict[str, int]:
    """Every starting slot, flex included. What a draft board counts against."""
    return {"QB": cfg.qb, "RB": cfg.rb, "WR": cfg.wr, "TE": cfg.te, "FLEX": cfg.flex}


def required_starters(cfg: RosterConfig) -> dict[str, int]:
    """Starting slots by position, flex excluded.

    The flex is not a position, and a lineup optimiser that treats it as one will happily
    start a quarterback in it. `starters()` and this differ by exactly that key.
    """
    return {"QB": cfg.qb, "RB": cfg.rb, "WR": cfg.wr, "TE": cfg.te}


def flex_positions(cfg: RosterConfig) -> tuple[str, ...]:
    """Positions eligible for the flex."""
    return tuple(cfg.flex_from)


def drafted_positions(cfg: RosterConfig | None = None) -> tuple[str, ...]:
    """The positions this league actually drafts, in board order.

    Exactly `required_starters`' keys: a position with no starting slot is not drafted, and
    kickers and defences have none here on purpose (ADR-0008). The same tuple was written out
    twelve times under four names -- `DRAFTED_POSITIONS`, `SKILL`, `POSITIONS` and
    `SCORING_POSITIONS` -- which is the defect this module was written to fix for the roster
    shape, one level down: a superflex or a K/DST league would move `RosterConfig` and none of
    the twelve, and deleting any one copy breaks only its own module, which is how there came
    to be four names.
    """
    return tuple(required_starters(cfg or RosterConfig()))


# The league's own answer, for the modules that want a constant rather than a call. Callers
# inside `FITTED_MODULES` must use the function instead: any module-level name *assigned* in
# one of those files is swept into `config_digest`, and the roster shape is a setting. (It
# used to read "an upper-case module-level name", which was true and was the bug -- #201:
# spelling decided coverage, so binding this to `drafted_positions` under a lower-case or
# underscored name would have been a silent way out of the digest. It is not one now.)
DRAFTED_POSITIONS: tuple[str, ...] = drafted_positions()


def flex_capacity(cfg: RosterConfig) -> int:
    """How many flex-eligible players a team can start at once.

    Required flex-eligible slots plus the flex itself -- 2 + 3 + 1 + 1 = 7 in this league.
    `optimize._need_score` carried that 7 as a literal, so a commissioner adding a second
    flex would have left the draft valuing depth against the old roster.
    """
    return sum(n for p, n in required_starters(cfg).items()
               if p in flex_positions(cfg)) + cfg.flex


# `flex_share(cfg)` used to sit here, reading `cfg.flex_rb/_wr/_te`. It is now
# `hub.draft.board.FLEX_SHARES`, a plain dict beside the sensitivity that justifies it, and
# it is hashed through `FITTED_EXTRA` rather than through the config schema. There is no
# forwarding shim: a function here that read a constant in `hub.draft` would be a second
# address for the same number and an import cycle besides, since `board` imports this file.


# ESPN's own names for the slots this league starts, so a league read over the wire can be
# compared against the config. RB/WR and WR/TE flex variants exist; this league runs the
# plain one.
_ESPN_SLOT_NAMES: dict[str, str] = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "FLEX": "RB/WR/TE",
}


def roster_mismatch(league: dict[str, Any]) -> dict[str, tuple]:
    """Where a league's own roster slots disagree with `RosterConfig`, as {slot: (league, ours)}.

    The counterpart to `components.scoring_mismatch`, and it exists for the same reason: the
    roster shape belongs to the league, not to this repo. `espn.league_settings()` has always
    returned it and every caller discarded it, so a commissioner moving to two flex slots
    would have shifted replacement level, positional need and every VOR on the board without
    a word.

    Slots the league runs and this config does not (K, D/ST, bench, IR) are not a
    disagreement -- nothing here drafts them. A slot count that differs is.
    """
    ours = starters(RosterConfig())
    out: dict[str, tuple] = {}
    for slot, n in ours.items():
        theirs = league.get(_ESPN_SLOT_NAMES[slot])
        if theirs is not None and int(theirs) != int(n):
            out[slot] = (int(theirs), int(n))
    return out
