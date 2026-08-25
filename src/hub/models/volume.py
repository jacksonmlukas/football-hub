"""Volume projected from the market's own draft pick.

`docs/component-projection.md` screened a volume model that shrank toward a positional mean
and got a null -- RMSE 3.311 against 3.303 for regressing touchdowns alone. The diagnosis
was that a WR1 and a WR5 do not regress toward the same place, so shrinking both toward one
positional average over-shrinks the studs, and that anchoring on an ADP-implied prior was
"a different and more promising thing". This is that thing, and it is worth being precise
about what it did and did not do. Full numbers in `docs/volume-model.md`.

**It fixed the volume model.** Anchoring on the pick beats carrying volume forward by
3.474 against 3.720 RMSE on held-out seasons, 99.9% on a paired bootstrap. The earlier null
was the anchor, not the idea.

**It did not beat reading the pick on its own.** The market alone scores 3.578, and the full
model's edge over it is 87% -- below anything this repo calls a result. It also ties a
trivial blend of the two point predictions (3.482), which is the tell that the structure is
not buying anything the arithmetic did not.

So what ships is a **decomposition**, not a replacement projection. Given the market's own
points projection, `decompose` hands back the component line that produces it: the mean is
reproduced exactly and only the shape comes from here. That shape is what
`components.sample_weeks` needs and what a points projection cannot give, and it claims no
edge that nobody demonstrated.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from hub.models import components as C

# Fitted on 526 player-season pairs from this league's own drafts, 2022-25, joined through
# nflverse's ff_playerids crosswalk. Volume is log-log in pick -- roughly a power law and
# strictly non-negative. Efficiency is linear in log(pick): yards per carry goes negative
# for a receiver with one carry for -5, and log(y+1) on that is NaN, which is what the first
# version of the screen did to 188 of 272 rows before it was caught.
VOLUME_CURVE: dict[str, dict[str, tuple[float, float]]] = {
    "targets":  {"QB": (0.0256, -0.0026), "RB": (1.9641, -0.1678),
                 "WR": (2.7580, -0.1923), "TE": (2.8032, -0.2090)},
    "carries":  {"QB": (3.0117, -0.3259), "RB": (3.4282, -0.2500),
                 "WR": (0.3239, -0.0417), "TE": (-0.0627, 0.0299)},
    "attempts": {"QB": (3.5426, -0.0097), "RB": (0.0171, -0.0027),
                 "WR": (0.0250, -0.0037), "TE": (-0.0393, 0.0103)},
}
EFFICIENCY_CURVE: dict[str, dict[str, tuple[float, float]]] = {
    "catch_rate":   {"QB": (0.1005, 0.0008), "RB": (0.8286, -0.0135),
                     "WR": (0.7023, -0.0137), "TE": (0.7307, -0.0061)},
    "yd_per_tgt":   {"QB": (-1.4947, 0.4584), "RB": (6.7340, -0.2241),
                     "WR": (10.0152, -0.4144), "TE": (10.8073, -0.6997)},
    "yd_per_carry": {"QB": (8.0766, -0.8394), "RB": (4.9357, -0.1354),
                     "WR": (2.6365, 0.1539), "TE": (1.9721, -0.2902)},
    "yd_per_att":   {"QB": (8.8034, -0.3154), "RB": (1.5943, -0.3383),
                     "WR": (1.8536, -0.1613), "TE": (-0.7306, 0.1914)},
}

# Where each position was actually drafted. Outside it the curve is extrapolating with
# nothing behind it: unclamped, it puts a tight end at pick 3 on 10.8 targets a game, more
# than any real tight end sees, because this league has never drafted one before pick 11.
PICK_RANGE: dict[str, tuple[int, int]] = {
    "QB": (24, 202), "RB": (1, 204), "WR": (1, 190), "TE": (11, 191),
}

# How much of his own prior season to keep, chosen on held-out data. Volume is trusted less
# than efficiency because volume is what a changed situation moves most -- a new coach, a
# signing, a depth-chart change -- and the pick is the only input that knows the situation
# changed at all.
KEEP_VOLUME = 0.5
KEEP_EFFICIENCY = 0.7

_UNITS = {"targets": "rec", "carries": "rush", "attempts": "pass"}


def _curve(table: dict, quantity: str, position: str, pick: float,
           log_y: bool) -> float | None:
    per_pos = table.get(quantity, {})
    if position not in per_pos or position not in PICK_RANGE:
        return None
    lo, hi = PICK_RANGE[position]
    a, b = per_pos[position]
    v = a + b * math.log(min(max(float(pick), lo), hi))
    v = math.exp(v) - 1.0 if log_y else v
    return v if math.isfinite(v) else None


def pick_prior(pick: float, position: str) -> dict[str, float]:
    """The component line the market's pick implies, per game.

    Empty for a position never fitted (K, DST), rather than raising -- the caller is a draft
    board and half of it is not skill players.
    """
    if position not in PICK_RANGE:
        return {}
    vol = {q: max(_curve(VOLUME_CURVE, q, position, pick, True) or 0.0, 0.0)
           for q in VOLUME_CURVE}
    eff = {q: _curve(EFFICIENCY_CURVE, q, position, pick, False) or 0.0
           for q in EFFICIENCY_CURVE}
    return _assemble(vol, eff, position)


def _assemble(vol: Mapping[str, float], eff: Mapping[str, float],
              position: str) -> dict[str, float]:
    """Volume times efficiency, with touchdowns from the positional rate on the yardage."""
    rec_y = max(vol["targets"] * eff["yd_per_tgt"], 0.0)
    rush_y = max(vol["carries"] * eff["yd_per_carry"], 0.0)
    pass_y = max(vol["attempts"] * eff["yd_per_att"], 0.0)
    return {
        "targets": vol["targets"], "carries": vol["carries"],
        "attempts": vol["attempts"],
        "receptions": max(vol["targets"] * eff["catch_rate"], 0.0),
        "receiving_yards": rec_y, "receiving_tds": rec_y * C.td_rate(position, "rec"),
        "rushing_yards": rush_y, "rushing_tds": rush_y * C.td_rate(position, "rush"),
        "passing_yards": pass_y, "passing_tds": pass_y * C.td_rate(position, "pass"),
    }


def project(prior_season: Mapping[str, float], pick: float,
            position: str) -> dict[str, float]:
    """Blend a player's own prior season with what his pick implies.

    A player with no prior season -- a rookie -- gets the pick alone, which is the whole of
    what is known about him.
    """
    prior = pick_prior(pick, position)
    if not prior or not prior_season:
        return prior

    def blended(key: str, keep: float, own: float | None) -> float:
        base = prior.get(key, 0.0)
        return base if own is None else keep * float(own) + (1.0 - keep) * base

    vol = {q: max(blended(q, KEEP_VOLUME, prior_season.get(q)), 0.0)
           for q in VOLUME_CURVE}
    own_eff = _rates(prior_season)
    eff = {q: blended(q, KEEP_EFFICIENCY, own_eff.get(q)) for q in EFFICIENCY_CURVE}
    # Efficiency has to come from the prior line's own rates, not from its raw yardage:
    # blending yardage directly would double-count the volume already blended above.
    for q in EFFICIENCY_CURVE:
        if own_eff.get(q) is None:
            eff[q] = _curve(EFFICIENCY_CURVE, q, position, pick, False) or 0.0
    return _assemble(vol, eff, position)


def _rates(line: Mapping[str, float]) -> dict[str, float | None]:
    """Per-unit efficiency implied by a component line, where the volume exists to imply it."""
    def rate(num: str, den: str) -> float | None:
        d = float(line.get(den) or 0.0)
        return float(line.get(num) or 0.0) / d if d > 0.1 else None
    return {"catch_rate": rate("receptions", "targets"),
            "yd_per_tgt": rate("receiving_yards", "targets"),
            "yd_per_carry": rate("rushing_yards", "carries"),
            "yd_per_att": rate("passing_yards", "attempts")}


def decompose(pick: float, position: str, target_ppg: float) -> dict[str, float]:
    """The component line that produces `target_ppg`, shaped by what the pick implies.

    This is the shipped use. The market's mean is reproduced exactly rather than
    second-guessed -- the screen never showed this model beating it -- and only the shape
    comes from here. That shape is what `components.sample_weeks` needs and what a points
    projection on its own cannot give.
    """
    prior = pick_prior(pick, position)
    base = C.points(prior)
    if not prior or base <= 0 or target_ppg <= 0:
        return dict.fromkeys(prior, 0.0) if prior else {}
    scale = float(target_ppg) / base
    return {k: v * scale for k, v in prior.items()}
