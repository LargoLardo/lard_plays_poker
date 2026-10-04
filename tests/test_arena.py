import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from agent_arena import run_match, python_policy
from cpp.run import build
from utils.training import NodeStore, TrainingNode, save_nodes
from utils.shared_nodes import packed_bucket


class ArenaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build(arena=True)

    def test_native_python_pairing_reproducibility_and_read_only_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, python = root / 'native.bin', root / 'python.pkl'
            subprocess.run([str(build()), '--iterations', '0', '--samples', '3',
                            '--output', str(native)], capture_output=True, check=True)
            nodes = NodeStore()
            nodes.schema, nodes.algorithm, nodes.trainer, nodes.samples = 2, 2, 'full-game', 3
            # The SB folds every starting hand; its later decisions use the shared fallback.
            ranks = '23456789TJQKA'
            for i, first in enumerate(ranks):
                for second in ranks[i:]:
                    hand = ''.join(sorted(first + second))
                    for suit in ('o',) if first == second else ('o', 's'):
                        node = TrainingNode()
                        node.strategy_sum['fold'] = 5000
                        node.times_visited = 5000
                        nodes[(hand + suit, 'SB', 'deep', 'root', '~2.0bb raise', 7)] = node
            save_nodes(nodes, python)
            before = {path: hashlib.sha256(path.read_bytes()).digest() for path in (native, python)}
            result = run_match(native, python, hands=200, seed=9, fallback='call')
            repeated = run_match(native, python, hands=200, seed=9, fallback='call')
            reversed_agents = run_match(python, native, hands=200, seed=9, fallback='call')
            self.assertEqual(result['a'], repeated['a'])
            self.assertEqual(result['b'], repeated['b'])
            self.assertEqual(result['a'], reversed_agents['b'])
            self.assertEqual(result['b'], reversed_agents['a'])
            self.assertEqual((result['hands'], result['duplicate_pairs'], result['samples']), (200, 100, 3))
            self.assertEqual(result['a']['net_bb'], -result['b']['net_bb'])
            self.assertGreater(result['b']['coverage']['trained'], 0)
            for agent in (result['a'], result['b']):
                self.assertEqual(agent['wins'] + agent['losses'] + agent['ties'], 200)
                coverage = agent['coverage']
                self.assertEqual(coverage['decisions'], sum(coverage[key] for key in ('trained', 'missing', 'no_average', 'preflop_runout')))
                self.assertLessEqual(agent['ci95_bb_per_100'][0], agent['bb_per_100'])
                self.assertGreaterEqual(agent['ci95_bb_per_100'][1], agent['bb_per_100'])
            selfplay = run_match(python, python, hands=200, samples=3, seed=9)
            self.assertEqual(selfplay['a']['net_bb'], 0)
            self.assertEqual(selfplay['a']['ci95_bb_per_100'], [0, 0])
            single_pair = run_match(native, native, hands=2, samples=3)
            self.assertIsNone(single_pair['a']['ci95_bb_per_100'])
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
            with self.assertRaises(ValueError):
                run_match(native, python, hands=3)
            with self.assertRaises(RuntimeError):
                run_match(native, python, hands=2, swap_a_legacy_positions=True)
            output = root / 'result.json'
            command = [sys.executable, 'agent_arena.py', str(native), str(python), '--hands', '20', '--output', str(output)]
            cli = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(cli.stdout), json.loads(output.read_text()))
            self.assertEqual(json.loads(output.read_text())['hands'], 20)
            command[-1] = str(python)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(hashlib.sha256(python.read_bytes()).digest(), before[python])

    def test_legacy_keys_and_illegal_weight_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = NodeStore()
            legacy = ('AKo', 'BB', 'deep', 'root', 'Limp')
            node = TrainingNode()
            node.strategy_sum.update({'fold': 3, 'check/call': 1, 'raise': 2})
            node.times_visited = 50
            nodes[legacy] = node
            source, converted = root / 'legacy.pkl', root / 'legacy.policy'
            save_nodes(nodes, source)
            metadata = python_policy(source, converted)
            self.assertIsNone(metadata['iterations'])
            with converted.open('rb') as stream:
                self.assertEqual(stream.read(8), b'LARDPOL1')
                schema, _, _, _, count = struct.unpack('<IIIQQ', stream.read(28))
                self.assertEqual((schema, count), (1, 1))
                key, *values = struct.unpack('<I3dQ', stream.read(36))
                self.assertEqual(key, packed_bucket((*legacy, 2)))
                self.assertEqual(values, [3, 1, 2, 50])
            result = run_match(source, source, hands=100, samples=3,
                               swap_a_legacy_positions=True, swap_b_legacy_positions=True)
            self.assertTrue(result['a']['legacy'])
            self.assertTrue(result['a']['seat_correction'])
            self.assertIsNone(result['a']['iterations'])
            self.assertEqual(result['a']['net_bb'], 0)
            # Legacy fold mass must be ignored when checking is free.
            nodes = NodeStore()
            ranks = '23456789TJQKA'
            for i, first in enumerate(ranks):
                for second in ranks[i:]:
                    hand = ''.join(sorted(first + second))
                    for suit in ('o',) if first == second else ('o', 's'):
                        node = TrainingNode()
                        node.strategy_sum['fold'] = 5000
                        node.times_visited = 5000
                        nodes[(hand + suit, 'SB', 'deep', 'limped', 'Limp')] = node
            save_nodes(nodes, source)
            illegal = run_match(source, source, hands=100, samples=3, fallback='call',
                                swap_a_legacy_positions=True, swap_b_legacy_positions=True)
            self.assertGreater(illegal['a']['coverage']['no_average'], 0)
            self.assertEqual(illegal['a']['net_bb'], 0)
            fixed = ('AKo', 'SB', 'deep', 'root', 'Limp', 7)
            nodes.schema = 2
            nodes[fixed] = TrainingNode()
            save_nodes(nodes, source)
            with self.assertRaises(ValueError):
                python_policy(source, converted)


if __name__ == '__main__':
    unittest.main()
