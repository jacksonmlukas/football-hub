"""The shape a prior-season signal takes to reach a pick.

`hub.draft.durability` and `hub.draft.regression` are the same four functions in the same
order -- `prior_season` → `<signal>` → `attach` → `correct_projection` -- and
`docs/improvements.md` #15 filed the duplication. Two pieces of it are genuinely one thing
written twice, and they are here.

**What is not here, deliberately.** The two `correct_projection` bodies are *not* the same
function: durability prices a trait *and* today's designation, over two different position
sets, while touchdown luck prices one term. Folding them into one shape with flags would be
the abstraction that costs more than the duplication. Each module keeps its own, and its own
`BETA` -- which also keeps `config_digest` untouched, since the digest is over the constants
those modules declare.
"""
from __future__ import annotations

import polars as pl

from hub.names import player_key


def join_by_player(board: pl.DataFrame, signal: pl.DataFrame, column: str) -> pl.DataFrame:
    """Attach `column` from `signal` to `board`, matched on the normalised player name.

    On the key rather than the raw name because nflverse and ESPN disagree about punctuation
    -- `A.J. Brown` against `AJ Brown` -- and an exact join drops him silently.

    **A player with no prior season keeps a null, never a zero.** The two mean different
    things: zero is "played every game" or "scored exactly as expected", null is "we do not
    know", and filling one with the other would quietly call every rookie durable and every
    rookie lucky. Rookies are half an early board.
    """
    if column not in signal.columns or signal.is_empty():
        return board.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    keyed = (signal.with_columns(
                pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("_k"))
             .select("_k", column).unique(subset=["_k"], keep="first"))
    return (board.drop(column, strict=False)
                 .with_columns(
                     pl.col("player").map_elements(player_key, return_dtype=pl.Utf8)
                       .alias("_k"))
                 .join(keyed, on="_k", how="left").drop("_k"))


def priced(column: str, beta: dict[str, float]) -> pl.Expr:
    """`beta[position] * column`, as an expression, with an unlisted position priced at zero.

    Zero rather than a pooled default: a position absent from a `BETA` is one the fit found
    nothing for, and inventing a coefficient for it would ship an effect nobody measured.
    """
    return (pl.col("pos").replace_strict(beta, default=0.0, return_dtype=pl.Float64)
            * pl.col(column).fill_null(0.0))
