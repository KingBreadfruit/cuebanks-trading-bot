"""
engine_v2/strategies — strategy registry.

Importing this package eagerly registers the four baseline strategies.
resolve(name, **kwargs) is the only constructor the CLI calls.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import Strategy, StrategyContext, Decision, BaseStrategy
from .pure_rules_ict import PureRulesICT
from .ml_gated import MLGated
from .random_walk import RandomWalk
# NOTE: cheating is imported lazily inside resolve() — its top-level import
# guard raises unless cfg.allow_cheating is True.


STRATEGIES: Dict[str, Type] = {
    "pure_rules_ict": PureRulesICT,
    "ml_gated":       MLGated,
    "random":         RandomWalk,
}


def resolve(name: str, allow_cheating: bool = False, **kwargs) -> Strategy:
    """Construct a strategy by registry name.

    Cheating is special-cased: requires allow_cheating=True AND late-loads
    the module so the import-time guard fires only when intentionally enabled.
    """
    if name == "cheating":
        if not allow_cheating:
            raise PermissionError(
                "strategy 'cheating' requires allow_cheating=True "
                "(also: --allow-cheating on the CLI)"
            )
        from .cheating import Cheating
        return Cheating(**kwargs)

    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(list(STRATEGIES.keys()) + ['cheating'])}"
        )
    return STRATEGIES[name](**kwargs)


__all__ = [
    "Strategy", "StrategyContext", "Decision", "BaseStrategy",
    "PureRulesICT", "MLGated", "RandomWalk",
    "STRATEGIES", "resolve",
]
