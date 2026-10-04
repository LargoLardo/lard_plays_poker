import unittest
from collections import defaultdict
from types import SimpleNamespace

from utils.agent_policy import action_weights, bucket_with_actions, node_for_actions


class FakeState:
    def __init__(self, facing_bet=True, street=0, can_raise=True):
        self.street_index = street
        self._facing_bet = facing_bet
        self._can_raise = can_raise
        self.actor_index = 0
        self.bets = [0, 1] if facing_bet else [1, 1]

    def can_fold(self):
        return self._facing_bet

    def can_complete_bet_or_raise_to(self):
        return self._can_raise


class AgentPolicyTests(unittest.TestCase):
    def test_never_folds_when_checking_is_free(self):
        node = SimpleNamespace(
            times_visited=100_000,
            strategy_sum=defaultdict(float, {"fold": 999, "check/call": 1}),
        )
        weights = action_weights(FakeState(facing_bet=False), ("7To", "BB", "deep", "root", "Limp"), node)
        self.assertNotIn("fold", weights)
        self.assertAlmostEqual(sum(weights.values()), 1)

    def test_sparse_four_bet_premium_raises(self):
        bucket = ("KAo", "SB", "deep", "vs_4bet", "~25.0bb raise")
        weights = action_weights(FakeState(), bucket, None)
        self.assertGreater(weights["raise"], weights["check/call"])
        self.assertEqual(weights["fold"], 0)

    def test_sparse_four_bet_trash_folds(self):
        bucket = ("27o", "SB", "deep", "vs_4bet", "~25.0bb raise")
        weights = action_weights(FakeState(), bucket, None)
        self.assertGreater(weights["fold"], 0.9)

    def test_lookup_separates_action_masks_and_supports_legacy_nodes(self):
        bucket = ('AKo', 'BB', 'short', 'vs_4bet', 'Jam (>25.00bb) raise')
        call, raise_node = object(), object()
        calling = ['fold', 'check/call']
        raising = calling + ['raise']
        nodes = {bucket_with_actions(bucket, calling): call,
                 bucket_with_actions(bucket, raising): raise_node}
        self.assertIs(node_for_actions(nodes, bucket, calling), call)
        self.assertIs(node_for_actions(nodes, bucket, raising), raise_node)
        del nodes[bucket_with_actions(bucket, calling)]
        self.assertIsNone(node_for_actions(nodes, bucket, calling))
        nodes[bucket] = call
        self.assertIs(node_for_actions(nodes, bucket, calling), call)


if __name__ == "__main__":
    unittest.main()
