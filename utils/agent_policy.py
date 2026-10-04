"""Safe strategy selection and sparse-node poker heuristics.

The trained node store is deliberately treated as one signal rather than an
absolute command.  Sparse nodes (most notably four-bet pots) are blended with
simple poker priors, and actions that PokerKit says are pointless or illegal
are removed before sampling.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

LOW_VISIT_THRESHOLD = 500


def can_meaningfully_fold(state) -> bool:
    """PokerKit permits a fold when checking is free, but warns against it."""
    actor = state.actor_index
    return actor is not None and state.bets[actor] < max(state.bets) and state.can_fold()


def legal_actions(state, raise_to=None) -> list[str]:
    actions = ["check/call"]
    if can_meaningfully_fold(state):
        actions.insert(0, "fold")
    if (state.can_complete_bet_or_raise_to() if raise_to is None
            else state.can_complete_bet_or_raise_to(raise_to)):
        actions.append("raise")
    return actions

def bucket_with_actions(bucket, actions):
    """Schema 2: distinguish the exact available actions, including chosen sizing."""
    mask = sum(1 << i for i, action in enumerate(('fold', 'check/call', 'raise')) if action in actions)
    return (*bucket, mask)


def node_for_actions(nodes, bucket, actions):
    # Only legacy keys omit the action mask; never fall back to another mask.
    node = nodes.get(bucket_with_actions(bucket, actions))
    return node if node is not None else nodes.get(bucket)


def _preflop_strength(hand: str) -> str:
    """Classify the exact bucket format used by exact_preflop_card_bucket."""
    if not isinstance(hand, str) or len(hand) < 3:
        return "unknown"
    ranks = hand[:2]
    rank_value = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
    high, low = sorted((rank_value.get(ranks[0], 0), rank_value.get(ranks[1], 0)), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = hand.endswith("s")

    if (pair and high >= 11) or (high == 14 and low >= 13) or (suited and high == 14 and low >= 12):
        return "premium"
    if pair or (high == 14 and low >= 10) or (high >= 12 and low >= 10) or (suited and high - low <= 2):
        return "playable"
    return "trash"


def heuristic_weights(state, bucket: tuple, raise_to=None) -> dict[str, float]:
    legal = legal_actions(state, raise_to)
    facing_bet = can_meaningfully_fold(state)
    street = state.street_index
    history = bucket[3] if street != 1 and len(bucket) > 3 else None

    if not facing_bet:
        weights = {"check/call": 0.72, "raise": 0.28}
    elif street == 0:
        strength = _preflop_strength(bucket[0])
        if history == "vs_4bet":
            weights = {
                "premium": {"fold": 0.0, "check/call": 0.25, "raise": 0.75},
                "playable": {"fold": 0.72, "check/call": 0.25, "raise": 0.03},
                "trash": {"fold": 0.97, "check/call": 0.03, "raise": 0.0},
            }[strength]
        else:
            weights = {
                "premium": {"fold": 0.0, "check/call": 0.35, "raise": 0.65},
                "playable": {"fold": 0.28, "check/call": 0.55, "raise": 0.17},
                "trash": {"fold": 0.78, "check/call": 0.20, "raise": 0.02},
            }[strength]
    else:
        equity_bucket = bucket[0][0] if isinstance(bucket[0], tuple) else 3
        if equity_bucket >= 6:
            weights = {"fold": 0.0, "check/call": 0.42, "raise": 0.58}
        elif equity_bucket >= 3:
            weights = {"fold": 0.28, "check/call": 0.58, "raise": 0.14}
        else:
            weights = {"fold": 0.72, "check/call": 0.26, "raise": 0.02}

    return {action: weights.get(action, 0.0) for action in legal}


def action_weights(state, bucket: tuple, node=None, raise_to=None) -> dict[str, float]:
    """Return normalized legal weights, blending sparse nodes with heuristics."""
    legal = legal_actions(state, raise_to)
    fallback = heuristic_weights(state, bucket, raise_to)
    visits = getattr(node, "times_visited", 0) if node is not None else 0
    raw: Mapping[str, float] = getattr(node, "strategy_sum", {}) if node is not None else {}
    raw_total = sum(max(float(raw.get(action, 0.0)), 0.0) for action in legal)

    confidence = min(visits / LOW_VISIT_THRESHOLD, 1.0) if raw_total > 0 else 0.0
    # Keep a small prior even for mature nodes; it prevents stale historical
    # fold mass from resurfacing in spots where checking is free.
    confidence *= 0.95
    trained = {action: max(float(raw.get(action, 0.0)), 0.0) / raw_total for action in legal} if raw_total else {}
    fallback_total = sum(fallback.values()) or 1.0
    weights = {
        action: confidence * trained.get(action, 0.0)
        + (1.0 - confidence) * fallback.get(action, 0.0) / fallback_total
        for action in legal
    }
    total = sum(weights.values()) or 1.0
    return {action: weight / total for action, weight in weights.items()}


def choose_action(state, bucket: tuple, node=None, rng=random, raise_to=None) -> str:
    weights = action_weights(state, bucket, node, raise_to)
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
