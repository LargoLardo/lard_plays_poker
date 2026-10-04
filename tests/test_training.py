import pickle
import random
import tempfile
from pathlib import Path
import unittest

import full_game_mccfr as full
import pf_mccfr as pf
from pokerkit import Card
from utils import card_bucketer as cards
from utils.training import TrainingNode, NodeStore, load_nodes, save_nodes


class TrainingTests(unittest.TestCase):
    def tearDown(self):
        cards.configure_caches()

    def test_cache_limit_sample_count_and_hidden_cards(self):
        cards.configure_caches(2)
        state = full.create_state()
        hero, board, unknown = cards._equity_inputs(state, 10)
        self.assertEqual(len(unknown), 50)
        opponent = [cards._to_treys(repr(c)) for c in state.hole_cards[1 - state.actor_index]]
        self.assertTrue(all(c in unknown for c in opponent))
        for i in range(5):
            cards._store(cards._ehs_cache, (i,), i)
        self.assertEqual(list(cards._ehs_cache), [(3,), (4,)])
        state.check_or_call()
        state.check_or_call()
        cards.compute_ehs(state, 1)
        cards.compute_ehs(state, 2)
        self.assertEqual({key[2] for key in cards._ehs_cache}, {1, 2})
        cards.configure_caches(0)
        cards.compute_potential(state, 2)
        self.assertFalse(cards._pot_cache)

    def test_straight_draw_rank_order_and_wheel(self):
        self.assertEqual(cards._max_straight_draw(list(Card.parse('3cAcAdKs'))), 2)
        self.assertTrue(cards.straight_draw_completed(list(Card.parse('As2d8c')), list(Card.parse('As2d8c3h'))))
        self.assertTrue(cards.straight_draw_completed(list(Card.parse('AsKd7c')), list(Card.parse('AsKd7cQh'))))

    def test_old_dictionary_node_state_loads_into_slots(self):
        node = full.Node.__new__(full.Node)
        node.__setstate__({'regret_sum': {'raise': 3.0}, 'strategy_sum': {'raise': 1.0}, 'times_visited': 7})
        self.assertFalse(hasattr(node, '__dict__'))
        restored = pickle.loads(pickle.dumps(node))
        self.assertEqual(restored.regret_sum, node.regret_sum)
        self.assertEqual(restored.times_visited, 7)
        self.assertEqual(restored.clone().strategy_sum, node.strategy_sum)
        # The bundled file records __main__.Node and must remain loadable.
        old = load_nodes('FULLGAME_10m_iters.pkl')
        self.assertTrue(old)
        self.assertTrue(all(isinstance(node, TrainingNode) for node in old.values()))

    def test_resume_remainder_limits_and_export_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'nodes.pkl'
            first = pf.train(3, seed=7, output=path)
            self.assertEqual(first.iterations, 3)
            resumed = pf.train(2, resume=path)
            continuous = pf.train(5, seed=7, output=Path(directory) / 'continuous.pkl')
            self.assertEqual(resumed.iterations, 5)
            self.assertEqual(load_nodes(path).iterations, 5)
            self.assertEqual(set(resumed), set(continuous))
            for key in resumed:
                self.assertEqual(resumed[key].regret_sum, continuous[key].regret_sum)
                self.assertEqual(resumed[key].strategy_sum, continuous[key].strategy_sum)
            limited = pf.train(100, max_nodes=1, output=Path(directory) / 'limited.pkl')
            self.assertEqual(limited.iterations, 1)
            self.assertTrue(limited)
            with self.assertRaises(FileExistsError):
                pf.train(1, output=path)
            with self.assertRaises(ValueError):
                full.train(1, resume=path, output=path)
            raw = pickle.loads(path.read_bytes())
            self.assertIsInstance(raw, dict)
            self.assertEqual(raw.iterations, 5)
            self.assertFalse(path.with_name(path.name + '.tmp').exists())

    def test_worker_snapshot_does_not_grow_on_regret_reads(self):
        node = pf.Node()
        state = pf.create_state()
        bucket = pf.Bucketer().exact_preflop_bucket(state, [])
        base = {bucket: node}
        random.seed(1)
        pf.play_hand(state, 0, base, {}, pf.Bucketer())
        self.assertFalse(node.regret_sum)

    def test_opponent_averaging_and_legacy_resume_guard(self):
        for trainer in (pf, full):
            state = trainer.create_state()
            bucket = trainer.Bucketer().exact_preflop_bucket(state, [])
            base = trainer.Node()
            base.regret_sum['fold'] = 100
            delta = {}
            value = trainer.play_hand(state, 0, {bucket: base}, delta, trainer.Bucketer(1))
            self.assertEqual(value, .5)
            self.assertEqual(delta[bucket].strategy_sum['fold'], 1)
            self.assertEqual(delta[bucket].times_visited, 1)
            self.assertFalse(delta[bucket].regret_sum)
            self.assertFalse(base.strategy_sum)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy.pkl'
            save_nodes(NodeStore({bucket: base}), legacy)
            with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                full.train(0, resume=legacy)
            reset = full.train(0, resume=legacy, reset_average=True)
            self.assertEqual(reset.algorithm, 2)
            self.assertFalse(reset[bucket].strategy_sum)
            self.assertEqual(reset[bucket].regret_sum['fold'], 100)

    def test_raise_to_includes_existing_bet(self):
        state = full.create_state()
        state.complete_bet_or_raise_to(3)
        state.complete_bet_or_raise_to(9)
        state.complete_bet_or_raise_to(27)
        bucket = full.Bucketer().exact_preflop_bucket(state, ['raise'] * 3)
        self.assertEqual(full.get_pf_raise_size(state, bucket), 100)
        self.assertTrue(state.can_complete_bet_or_raise_to(100))


if __name__ == '__main__':
    unittest.main()
