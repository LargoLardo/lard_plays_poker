"""V5 frozen abstraction, checkpoint, inference and cross-model checks."""

import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from cpp.run import build
from agent_arena import run_match
from tools.audit_model import audit
from treys import Card, Evaluator


def read_v5(path):
    data = path.read_bytes()
    if data[:8] != b'LARDCPP5':
        raise ValueError('Not a V5 checkpoint')
    mode, samples, workers, chunk, hands, count, touches, linear, every, periods, length = struct.unpack_from('<IIIIQQQIQQI', data, 8)
    offset = 72 + length
    feature_samples, seed, examples = struct.unpack_from('<IQQ', data, offset)
    offset += 20
    for _ in range(3):
        clusters, = struct.unpack_from('<I', data, offset)
        offset += 4 + clusters * 32 * 8
    nodes = {}
    for _ in range(count):
        history, context, *values = struct.unpack_from('<QI10dQ', data, offset)
        nodes[history, context] = values
        offset += 100
    if offset != len(data) or samples != feature_samples or mode != 0:
        raise ValueError('Invalid V5 metadata')
    return dict(hands=hands, touches=touches, linear=linear, every=every, periods=periods,
                workers=workers, chunk=chunk, rng=data[72:72 + length],
                abstraction=data[72 + length:len(data) - count * 100], nodes=nodes,
                records=len(data) - count * 100, seed=seed, examples=examples)


class CppV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = build(model='v5')
        cls.engine = build(test=True, model='v5')
        subprocess.run([str(cls.engine)], check=True, capture_output=True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.asset = cls.directory / 'cards.abs'
        cls.train('--build-abstraction', cls.asset, '--examples', 64, '--clusters', '8,8,8',
                '--samples', 32, '--cluster-rounds', 2, '--workers', 2)
        cls.checkpoint = cls.directory / 'model.bin'
        cls.train('--abstraction', cls.asset, '--iterations', 2000, '--workers', 3,
                '--chunk-size', 4, '--discount-every', 1000, '--output', cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def train(cls, *arguments, success=True, input=None):
        result = subprocess.run([str(cls.binary), '--memory-mb', '128', '--cache-mb', '0',
                                 *map(str, arguments)], input=input, text=True, capture_output=True)
        if success and result.returncode:
            raise AssertionError(result.stderr)
        if not success and not result.returncode:
            raise AssertionError('Invalid operation succeeded')
        return result

    def test_asset_is_worker_independent_and_frozen_on_resume(self):
        other = self.directory / 'single-worker.abs'
        self.train('--build-abstraction', other, '--examples', 64, '--clusters', '8,8,8',
                 '--samples', 32, '--cluster-rounds', 2, '--workers', 1)
        self.assertEqual(self.asset.read_bytes(), other.read_bytes())
        original = self.checkpoint.read_bytes()
        for options in (('--samples', 99), ('--algorithm', 'vanilla'),
                        ('--discount-every', 2000), ('--abstraction', other)):
            self.train('--resume', self.checkpoint, '--iterations', 1, *options, success=False)
            self.assertEqual(self.checkpoint.read_bytes(), original)
        self.train('--build-abstraction', self.asset, success=False)
        self.train('--abstraction', self.asset, '--output', self.checkpoint, success=False)

    def test_serial_and_snapshot_boundary_parallel_resume(self):
        for workers, split, total in ((1, 31, 80), (3, 24, 48)):
            first = self.directory / f'part-{workers}.bin'
            whole = self.directory / f'whole-{workers}.bin'
            common = ('--abstraction', self.asset, '--workers', workers, '--chunk-size', 8,
                      '--discount-every', 1000, '--snapshot-every', split)
            self.train(*common, '--iterations', total, '--output', whole)
            self.train(*common, '--iterations', split, '--output', first)
            self.train('--resume', first, '--iterations', total - split, '--snapshot-every', split)
            a, b = read_v5(first), read_v5(whole)
            self.assertEqual(a, b)
            snapshot = first.parent / (first.stem + '-snapshots') / f'iter-{split}.bin'
            self.assertEqual(read_v5(snapshot)['hands'], split)
        limited = self.directory / 'limited.bin'
        self.train('--abstraction', self.asset, '--workers', 3, '--chunk-size', 8,
                 '--max-nodes', 1, '--iterations', 24, '--output', limited)
        empty = self.directory / 'empty.bin'
        self.train('--abstraction', self.asset, '--workers', 3, '--chunk-size', 8,
                 '--iterations', 0, '--output', empty)
        self.assertEqual(read_v5(limited), read_v5(empty), 'Failed batches preserve RNG and model')

    def test_malformed_checkpoint_and_read_only_export(self):
        data = self.checkpoint.read_bytes()
        metadata = read_v5(self.checkpoint)
        mutations = [data[:30], data[:-1], data + b'extra']
        malformed = bytearray(data)
        struct.pack_into('<Q', malformed, metadata['records'], 0)
        mutations.append(malformed)
        malformed = bytearray(data)
        struct.pack_into('<d', malformed, metadata['records'] + 12, float('nan'))
        mutations.append(malformed)
        malformed = bytearray(data)
        struct.pack_into('<d', malformed, metadata['records'] + 52, -1)
        mutations.append(malformed)
        bad = self.directory / 'bad.bin'
        for mutation in mutations:
            bad.write_bytes(mutation)
            before = bad.read_bytes()
            self.train('--resume', bad, '--iterations', 0, success=False)
            self.assertEqual(bad.read_bytes(), before)
        exported = self.directory / 'web'
        self.train('--resume', self.checkpoint, '--iterations', 0, '--export', exported)
        self.assertEqual(self.checkpoint.read_bytes(), data)
        rows = json.loads((exported / 'preflop-model.json').read_text())
        self.assertTrue(rows)
        self.assertTrue(all(len(row) == 7 for row in rows.values()))
        self.assertTrue(all(row[:3] == [0, 0, 0] for row in rows.values() if row[3] < 1000))
        self.assertEqual(json.loads((exported / 'postflop-model.json').read_text()), {})
        report = audit(self.checkpoint)
        self.assertEqual(report['metadata']['format'], 'LARDCPP5')
        self.assertEqual(report['metadata']['clusters'], [8, 8, 8])
        self.assertEqual(sum(row['nodes'] for row in report['streets'].values()), len(metadata['nodes']))
        self.assertEqual(len(report['metadata']['actions']), 5)

    def test_export_retains_each_raise_size_without_rewriting_checkpoint(self):
        metadata = read_v5(self.checkpoint)
        data = bytearray(self.checkpoint.read_bytes())
        index = next(i for i, (history, context) in enumerate(metadata['nodes']) if history == 1)
        offset = metadata['records'] + index * 100
        struct.pack_into('<5dQ', data, offset + 52, 1, 2, 3, 4, 5, 1000)
        checkpoint = self.directory / 'raise-split.bin'
        checkpoint.write_bytes(data)
        exported = self.directory / 'raise-split-web'
        self.train('--resume', checkpoint, '--iterations', 0, '--export', exported)
        rows = json.loads((exported / 'preflop-model.json').read_text())
        mature = next(row for row in rows.values() if row[3] == 1000)
        for actual, expected in zip(mature, [1 / 15, 2 / 15, 12 / 15, 1000, 3 / 15, 4 / 15, 5 / 15]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(mature[4:7]), mature[2])
        self.assertTrue(all(row[4:7] == [0, 0, 0] for row in rows.values() if row[3] < 1000))
        self.assertEqual(checkpoint.read_bytes(), data)

    def test_river_features_match_exact_treys_opponent_classes(self):
        rng = random.Random(17)
        deck = [rank + suit for rank in '23456789TJQKA' for suit in 'shdc']
        deals = [rng.sample(deck, 9) for _ in range(12)]
        result = subprocess.run([str(self.engine), 'features'], check=True, capture_output=True, text=True,
                                input='\n'.join(' '.join(deal) + ' 0 3 32' for deal in deals))
        evaluator = Evaluator()
        for deal, line in zip(deals, result.stdout.splitlines()):
            actual = json.loads(line)['features']
            known = deal[:2] + deal[4:]
            available = [c for c in deck if c not in known]
            board = [Card.new(c) for c in deal[4:]]
            hero = evaluator.evaluate(board, [Card.new(c) for c in deal[:2]])
            wins, totals = [0.0] * 8, [0] * 8
            for a, b in itertools.combinations(available, 2):
                other = evaluator.evaluate(board, [Card.new(a), Card.new(b)])
                high, low = sorted(('23456789TJQKA'.index(a[0]), '23456789TJQKA'.index(b[0])), reverse=True)
                suited = a[1] == b[1]
                group = (0 if high >= 8 else 1) if high == low else (2 if suited else 3) if low >= 8 else (4 if suited else 5) if high == 12 else (6 if suited else 7)
                wins[group] += 1 if hero < other else .5 if hero == other else 0
                totals[group] += 1
            self.assertEqual(sum(totals), 990)
            equity = sum(wins) / 990
            bin_index = min(15, int(equity * 16))
            self.assertEqual(actual[:16], [0] * bin_index + [1] * (16 - bin_index))
            for group in range(8):
                self.assertAlmostEqual(actual[16 + group], (wins[group] + 2 * equity) / (totals[group] + 2), places=6)

    def test_inference_uses_saved_average_and_survives_invalid_requests(self):
        original = hashlib.sha256(self.checkpoint.read_bytes()).digest()
        request = 'As Kd 0 0 1 0'
        result = self.train('--resume', self.checkpoint, '--infer', input=request + '\ninvalid\n' + request + '\n')
        first, invalid, last = map(json.loads, result.stdout.splitlines())
        self.assertEqual(first, last)
        self.assertIn('error', invalid)
        self.assertEqual(first['history'], '1')
        self.assertEqual(first['mask'], 31)
        self.assertEqual(first['amounts'][2:], [2.5, 3, 100])
        node = read_v5(self.checkpoint)['nodes'].get((1, first['bucket'] | 1 << 16 | 31 << 19))
        weights = node[5:10] if node else [0] * 5
        total = sum(weights)
        self.assertEqual(first['trained'], total > 0)
        for observed, weight in zip(first['weights'], weights):
            self.assertAlmostEqual(observed, weight / total if total else .2)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).digest(), original)

    def test_browser_actions_match_entire_native_tree(self):
        result = subprocess.run([str(self.engine), 'tree'], check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 13608)
        script = """
import assert from 'node:assert/strict';
import {v5ActionMask, v5ActionAmounts} from './public/model-policy.js';
let data = '';
for await (const chunk of process.stdin) data += chunk;
for (const row of JSON.parse(data)) {
  assert.equal(v5ActionMask(row.game), row.mask, row.history);
  assert.deepEqual(v5ActionAmounts(row.game), row.amounts, row.history);
}
"""
        subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(rows),
                       check=True, text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1])

    def test_arena_cross_models_baselines_and_duplicate_symmetry(self):
        identical = run_match(self.checkpoint, self.checkpoint, hands=200, seed=11)
        self.assertEqual(identical['a']['net_bb'], 0)
        self.assertEqual(identical['a']['ci95_bb_per_100'], [0, 0])
        self.assertEqual(identical['a']['coverage']['off_tree_decisions'], 0)
        self.assertEqual(identical['a']['coverage']['translated_actions'], 0)
        random_match = run_match(self.checkpoint, Path('baseline:random'), hands=200, seed=11)
        self.assertEqual(random_match['a']['net_bb'], -random_match['b']['net_bb'])
        self.assertGreater(random_match['a']['coverage']['translated_actions'], 0)
        self.assertEqual(random_match['a']['feature_samples'], 32)
        legacy = self.directory / 'v4.bin'
        subprocess.run([str(build()), '--iterations', '200', '--samples', '3', '--output', str(legacy)], check=True, capture_output=True)
        cross = run_match(self.checkpoint, legacy, hands=200, seed=4, samples=3)
        self.assertEqual(cross['a']['model_version'], 5)
        self.assertEqual(cross['b']['model_version'], 4)
        self.assertEqual(cross['a']['feature_samples'], 32, 'Arena samples cannot change frozen V5 buckets')

    def test_hosted_v1_export_preserves_original_strategy_weights(self):
        from agent_arena import python_policy
        from tools.export_arena_policy import export
        source = Path('checkpoints/v1.pkl')
        original = self.directory / 'original-v1.policy'
        hosted = self.directory / 'hosted-v1.policy'
        python_policy(source, original)
        export(source, hosted)
        expected, actual = original.read_bytes(), hosted.read_bytes()
        self.assertEqual(actual[:20], expected[:20])
        self.assertEqual(actual[28:], expected[28:], 'Every bucket, strategy weight and visit count is unchanged')
        self.assertEqual(struct.unpack_from('<Q', actual, 20)[0], 10_000_000)


if __name__ == '__main__':
    unittest.main()
