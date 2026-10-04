import pickle
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import full_game_mccfr as full
import pf_mccfr as pf
from pokerkit import Card
from utils import card_bucketer as cards
from utils.training import TrainingNode, NodeStore, load_nodes, save_nodes
from tools.audit_model import audit
from utils.agent_policy import legal_actions, bucket_with_actions


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
            cards._store(cards._feature_cache, (i,), i)
        self.assertEqual(list(cards._feature_cache), [(3,), (4,)])
        state.check_or_call()
        state.check_or_call()
        cards.compute_ehs(state, 1)
        cards.compute_ehs(state, 2)
        self.assertEqual({key[2] for key in cards._feature_cache}, {1, 2})
        cards.configure_caches(0)
        cards.compute_potential(state, 2)
        self.assertFalse(cards._feature_cache)

    def test_straight_draw_rank_order_and_wheel(self):
        self.assertEqual(cards._max_straight_draw(list(Card.parse('3cAcAdKs'))), 2)
        self.assertTrue(cards.straight_draw_completed(list(Card.parse('As2d8c')), list(Card.parse('As2d8c3h'))))
        self.assertTrue(cards.straight_draw_completed(list(Card.parse('AsKd7c')), list(Card.parse('AsKd7cQh'))))

    def test_river_collision_is_split_by_actual_legal_actions(self):
        state = full.create_state()
        # Common path reaches one fixed river deal; both continuations share cards.
        for action, amount in [('c', 0), ('c', 0), ('c', 0), ('r', 1), ('c', 0),
                               ('c', 0), ('r', 2), ('r', 5), ('c', 0)]:
            state.check_or_call() if action == 'c' else state.complete_bet_or_raise_to(amount)
        histories = [[], [], ['check/call', 'raise', 'raise', 'check/call']]
        keys, base_keys = [], []
        bucketer = full.Bucketer(10)
        for line in ([('c', 0), ('r', 7), ('r', 18), ('r', 38), ('r', 93)],
                     [('r', 7), ('r', 18), ('r', 38)]):
            current = pickle.loads(pickle.dumps(state))
            history = []
            for action, amount in line:
                current.check_or_call() if action == 'c' else current.complete_bet_or_raise_to(amount)
                history.append('check/call' if action == 'c' else 'raise')
            bucket = bucketer.river_bucket(current, history, histories[2])
            actions = legal_actions(current, full.get_halfp_raise_size(current, bucket))
            base_keys.append(bucket)
            keys.append(bucket_with_actions(bucket, actions))
        self.assertEqual(base_keys[0], base_keys[1])
        self.assertEqual([key[-1] for key in keys], [3, 7])
        self.assertNotEqual(keys[0], keys[1])

    def test_joint_equity_potential_uses_one_sample_pass(self):
        state = SimpleNamespace(actor_index=0,
                                hole_cards=[list(Card.parse('AhAd')), list(Card.parse('8h9h'))],
                                board_cards=[[c] for c in Card.parse('2h7dTs')])
        samples = [[cards._to_treys(c) for c in text.split()] for text in (
            'Ks Kd Ac 3s', 'Th Tc As Ac', 'Kh Kd Kc 3s', '5s 6s 3d 4d')]
        cards.configure_caches(2)
        with patch('utils.card_bucketer.random.sample', side_effect=samples) as sample:
            self.assertEqual(cards.compute_ehs(state, 4), .5)
            self.assertEqual(cards.compute_potential(state, 4), (1, 2 / 3))
            self.assertEqual(sample.call_count, 4)
        self.assertEqual(len(cards._feature_cache), 1)
        cards.configure_caches(0)
        with patch('utils.card_bucketer.random.sample', side_effect=samples) as sample:
            self.assertEqual(cards.flop_card_bucket(state, 4), (4, 3, 2, 'rainbow', False))
            self.assertEqual(sample.call_count, 4)

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

    def test_web_export_preserves_action_masks_and_legacy_keys(self):
        preflop = ('AKo', 'SB', 'deep', 'root', 'Limp')
        river = ((5, False, False, False), 'SB', 'vs_4bet', 'small', 'short', False)
        nodes = NodeStore()
        for base in (preflop, river):
            for mask in (None, 3, 7):
                node = TrainingNode()
                node.strategy_sum.update({'fold': 1, 'check/call': 2, 'raise': 0 if mask == 3 else 3})
                node.times_visited = 10
                nodes[base if mask is None else (*base, mask)] = node
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pre, post = [root / name for name in ('nodes.pkl', 'pre.json', 'post.json')]
            source.write_bytes(pickle.dumps(nodes))
            result = subprocess.run([sys.executable, 'tools/export_web_model.py', str(source), str(pre),
                                     '--postflop-output', str(post)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(pre.read_text())), {
                '|'.join(preflop), '|'.join(preflop) + '|3', '|'.join(preflop) + '|7'})
            exported = json.loads(post.read_text())
            self.assertEqual({len(json.loads(key)) for key in exported}, {6, 7})
            self.assertEqual({json.loads(key)[-1] for key in exported if len(json.loads(key)) == 7}, {3, 7})

    def test_iteration_snapshots_resume_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'nested' / 'run.pkl'
            snapshots = output.parent / 'run-snapshots'
            pf.train(7, seed=9, output=output, snapshot_every=3)
            self.assertEqual(sorted(p.name for p in snapshots.iterdir()), ['iter-3.pkl', 'iter-6.pkl'])
            pf.train(3, resume=output, snapshot_every=3)
            for count, path in ((3, snapshots / 'iter-3.pkl'), (6, snapshots / 'iter-6.pkl'),
                                (9, snapshots / 'iter-9.pkl'), (10, output)):
                reference_path = root / f'reference-{count}.pkl'
                reference = pf.train(count, seed=9, output=reference_path)
                saved = load_nodes(path)
                self.assertEqual(saved.iterations, count)
                self.assertEqual(saved.rng_state, reference.rng_state)
                self.assertEqual(set(saved), set(reference))
                for key in saved:
                    self.assertEqual(saved[key].regret_sum, reference[key].regret_sum)
                    self.assertEqual(saved[key].strategy_sum, reference[key].strategy_sum)
                    self.assertEqual(saved[key].times_visited, reference[key].times_visited)
            self.assertFalse((root / 'reference-10-snapshots').exists())
            original = (snapshots / 'iter-6.pkl').read_bytes()
            pf.train(3, resume=snapshots / 'iter-3.pkl', output=output,
                     snapshot_every=3, reset_average=True)
            self.assertEqual((snapshots / 'iter-6.pkl').read_bytes(), original)
            self.assertFalse(list(snapshots.glob('*.tmp')))
            with self.assertRaises(ValueError):
                pf.train(0, snapshot_every=-1, output=root / 'invalid.pkl')

    def test_parallel_snapshots_stop_batches_at_exact_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'parallel.pkl'
            result = pf.train(9, n_workers=2, merge_every=3, seed=9,
                              output=output, snapshot_every=4)
            self.assertEqual(result.iterations, 9)
            snapshots = output.parent / 'parallel-snapshots'
            self.assertEqual(sorted(p.name for p in snapshots.iterdir()), ['iter-4.pkl', 'iter-8.pkl'])
            for count in (4, 8):
                self.assertEqual(load_nodes(snapshots / f'iter-{count}.pkl').iterations, count)

    def test_worker_snapshot_does_not_grow_on_regret_reads(self):
        node = pf.Node()
        state = pf.create_state()
        bucket = pf.Bucketer().exact_preflop_bucket(state, [])
        bucket = bucket_with_actions(bucket, legal_actions(state, 2))
        base = {bucket: node}
        random.seed(1)
        pf.play_hand(state, 0, base, {}, pf.Bucketer())
        self.assertFalse(node.regret_sum)

    def test_opponent_averaging_and_legacy_resume_guard(self):
        for trainer in (pf, full):
            state = trainer.create_state()
            bucket = trainer.Bucketer().exact_preflop_bucket(state, [])
            bucket = bucket_with_actions(bucket, legal_actions(state, 3))
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
            with self.assertRaisesRegex(ValueError, 'buckets merge legal actions'):
                full.train(0, resume=legacy, reset_average=True)
            compatible = NodeStore({bucket: base})
            compatible.schema, compatible.algorithm = 2, 2
            save_nodes(compatible, legacy)
            reset = full.train(0, resume=legacy, reset_average=True)
            self.assertEqual(reset.algorithm, 2)
            self.assertFalse(reset[bucket].strategy_sum)
            self.assertEqual(reset[bucket].regret_sum['fold'], 100)

    def test_model_audit_reports_sparse_weights_and_unknown_legacy_metadata(self):
        node = TrainingNode()
        node.strategy_sum['raise'] = 3
        node.times_visited = 500
        nodes = NodeStore({('AAo', 'SB', 'deep', 'root', '~2.0bb raise'): node,
                           ((1, 0, 0, 'rainbow', False, False, False),
                            'BB', 'root', 'small', 'short', False): TrainingNode()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.pkl'
            save_nodes(nodes, path)
            original = path.read_bytes()
            report = audit(path)
            self.assertIsNone(report['metadata']['iterations'])
            self.assertIsNone(report['metadata']['samples'])
            self.assertEqual(report['streets']['turn']['no_average'], 1)
            self.assertEqual(report['streets']['turn']['under_100'], 1)
            self.assertEqual(report['streets']['preflop']['under_500'], 0)
            self.assertEqual(report['sb_root_combo_weighted']['raise'], 1)
            self.assertEqual(path.read_bytes(), original)
            nodes.iterations, nodes.samples, nodes.algorithm = 20, 50, 2
            save_nodes(nodes, path)
            report = audit(path)
            self.assertEqual(report['metadata']['iterations'], 20)
            self.assertEqual(report['metadata']['samples'], 50)
            self.assertEqual(report['metadata']['algorithm'], 2)

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
