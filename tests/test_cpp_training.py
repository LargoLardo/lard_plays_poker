"""Differential checks against the original trainer's poker libraries."""

import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from pokerkit import Automation, Mode, NoLimitTexasHoldem
from treys import Card as TreysCard, Evaluator
from utils.bucketer import Bucketer
from utils.agent_policy import legal_actions, bucket_with_actions
from full_game_mccfr import get_pf_raise_size, get_halfp_raise_size

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / 'cpp'


def run(*arguments):
    return subprocess.run([str(CPP / 'run.sh'), *map(str, arguments)], cwd=ROOT, text=True, capture_output=True)


def read_checkpoint(path):
    with open(path, 'rb') as stream:
        if stream.read(8) not in (b'LARDCPP1', b'LARDCPP2', b'LARDCPP3'):
            raise ValueError('invalid magic')
        mode, samples, iterations, count, length = struct.unpack('<IIQQI', stream.read(28))
        rng = stream.read(length)
        nodes = {}
        for _ in range(count):
            key, *values = struct.unpack('<I6dQ', stream.read(60))
            nodes[key] = values
        return mode, samples, iterations, rng, nodes


class CppTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([str(CPP / 'test.sh')], cwd=ROOT, check=True, capture_output=True)

    def test_evaluator_matches_treys_for_5_6_7_cards(self):
        rng = random.Random(8)
        deck = [r + s for r in '23456789TJQKA' for s in 'shdc']
        hands = [rng.sample(deck, size) for size in (5, 6, 7) for _ in range(4000)]
        # Explicit wheel, full house with two trips, board ties and straight flush.
        hands += [text.split() for text in [
            'As 2s 3s 4s 5s Kd Qh', 'As Ah Ad Ks Kh Kd 2c', 'Ts Js Qs Ks As 2d 3h',
            'As Ah Ad Ac Ks Kh Kd', '2s 3d 4h 5c 6s 7d 8h']]
        result = subprocess.run([str(CPP / 'build/test'), 'eval'], input='\n'.join(' '.join(h) for h in hands), text=True, capture_output=True, check=True)
        scores = list(map(int, result.stdout.splitlines()))
        evaluator = Evaluator()
        expected = [evaluator.evaluate([TreysCard.new(c) for c in hand[:2]], [TreysCard.new(c) for c in hand[2:]]) for hand in hands]
        # Equal ranks must tie, and every distinct rank must have the same ordering.
        ranks = {}
        for native, treys in zip(scores, expected):
            if treys in ranks:
                self.assertEqual(native, ranks[treys])
            ranks[treys] = native
        ordered = [ranks[rank] for rank in sorted(ranks)]
        self.assertTrue(all(a > b for a, b in zip(ordered, ordered[1:])))

    def test_betting_and_buckets_match_pokerkit(self):
        rng = random.Random(72)
        deck = [r + s for r in '23456789TJQKA' for s in 'shdc']
        cases, expected, browser_cases = [], [], []
        for case in range(250):
            deal = rng.sample(deck, 9)
            automations = tuple(a for a in Automation if a not in (
                Automation.HOLE_DEALING, Automation.BOARD_DEALING, Automation.CARD_BURNING))
            state = NoLimitTexasHoldem.create_state(automations, False, 0, (.5, 1), 1, (100, 100), 2, mode=Mode.CASH_GAME)
            state.deal_hole(''.join(deal[:2]), 0)
            state.deal_hole(''.join(deal[2:4]), 1)
            history = [[], [], [], []]
            last_raise = 1
            actions, snapshots = [], []
            burns = iter(c for c in deck if c not in deal)
            bucketer = Bucketer(1)

            def settle_deal():
                while state.actor_index is None and state.status:
                    if state.can_burn_card():
                        state.burn_card(next(burns))
                    elif state.can_deal_board():
                        n = len(state.board_cards)
                        count = 3 if n == 0 else 1
                        state.deal_board(''.join(deal[4 + n:4 + n + count]))
                    else:
                        self.fail('PokerKit stalled')

            def snapshot():
                if state.actor_index is None:
                    return {'street': 4, 'payoff': state.stacks[0] - 100}
                street = state.street_index
                if street == 0:
                    bucket = bucketer.exact_preflop_bucket(state, history[0])
                elif street == 1:
                    bucket = bucketer.flop_bucket(state, history[1])
                elif street == 2:
                    bucket = bucketer.turn_bucket(state, history[2], history[1])
                else:
                    bucket = bucketer.river_bucket(state, history[3], history[2])
                amount = get_pf_raise_size(state, bucket) if street == 0 else get_halfp_raise_size(state, bucket)
                bucket = bucket_with_actions(bucket, legal_actions(state, amount))
                browser_cases.append({
                    'game': {'street': street, 'agent': state.actor_index,
                             'stacks': list(state.stacks), 'bets': list(state.bets),
                             'pot': state.total_pot_amount - sum(state.bets),
                             'histories': [list(h) for h in history], 'lastRaise': last_raise},
                    'raiseTo': amount, 'mask': bucket[-1]})
                minimum = state.min_completion_betting_or_raising_to_amount
                return {'street': street, 'actor': state.actor_index,
                        'stacks': [int(x * 2) for x in state.stacks],
                        'bets': [int(x * 2) for x in state.bets],
                        'pot': int(state.total_pot_amount * 2),
                        'minimum': int(minimum * 2) if minimum is not None else -1,
                        'bucket': json.loads(json.dumps(bucket))}

            snapshots.append(snapshot())
            while state.actor_index is not None:
                street, actor = state.street_index, state.actor_index
                legal = [1]
                if state.bets[actor] < max(state.bets):
                    legal.append(0)
                if state.can_complete_bet_or_raise_to():
                    legal.append(2)
                # Include check-through, all-in, min raises, arbitrary legal sizing.
                action = 1 if case < 10 else rng.choice(legal)
                amount = 0
                if action == 0:
                    state.fold()
                elif action == 1:
                    state.check_or_call()
                else:
                    old_high = max(state.bets)
                    low = int(state.min_completion_betting_or_raising_to_amount * 2)
                    high = int(state.max_completion_betting_or_raising_to_amount * 2)
                    amount = rng.choice([low, high, rng.randint(low, high)])
                    state.complete_bet_or_raise_to(amount / 2)
                    last_raise = max(last_raise, amount / 2 - old_high)
                history[street].append(['fold', 'check/call', 'raise'][action])
                actions += [action, amount]
                settle_deal()
                if state.street_index != street:
                    last_raise = 1
                snapshots.append(snapshot())
            cases.append(' '.join(deal + list(map(str, actions))))
            expected.append(snapshots)
        result = subprocess.run([str(CPP / 'build/test'), 'trace'], input='\n'.join(cases), text=True, capture_output=True, check=True)
        native_cases = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(native_cases), len(expected))
        for index, (actual, wanted) in enumerate(zip(native_cases, expected)):
            self.assertEqual(len(actual), len(wanted), cases[index])
            for native, python in zip(actual, wanted):
                for key, value in python.items():
                    if key == 'bucket' and python['street']:
                        self.assertEqual(native[key][1:], value[1:], cases[index])
                        skip = 1 if python['street'] == 3 else 3
                        self.assertEqual(native[key][0][skip:], value[0][skip:], cases[index])
                    else:
                        self.assertEqual(native[key], value, (cases[index], key))

        browser = subprocess.run(['node', '--input-type=module', '-e', '''
import assert from 'node:assert/strict';
import { modelActionMask, modelRaiseTo } from './public/model-policy.js';
let input = '';
for await (const chunk of process.stdin) input += chunk;
for (const { game, raiseTo, mask } of JSON.parse(input)) {
    assert.equal(modelRaiseTo(game), raiseTo, JSON.stringify(game));
    assert.equal(modelActionMask(game), mask, JSON.stringify(game));
}
'''], cwd=ROOT, input=json.dumps(browser_cases), text=True, capture_output=True)
        self.assertEqual(browser.returncode, 0, browser.stderr)

    def test_iteration_snapshots_resume_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'nested' / 'run.bin'
            snapshots = output.parent / 'run-snapshots'
            first = run('--iterations', 7, '--samples', 10, '--seed', 9,
                        '--snapshot-every', 5, '--output', output)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual([p.name for p in snapshots.iterdir()], ['iter-5.bin'])
            resumed = run('--resume', output, '--iterations', 5, '--snapshot-every', 5)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(sorted(p.name for p in snapshots.iterdir()), ['iter-10.bin', 'iter-5.bin'])
            for count, path in ((5, snapshots / 'iter-5.bin'),
                                (10, snapshots / 'iter-10.bin'), (12, output)):
                reference = root / f'reference-{count}.bin'
                result = run('--iterations', count, '--samples', 10, '--seed', 9, '--output', reference)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(read_checkpoint(path), read_checkpoint(reference))
            self.assertFalse((root / 'reference-12-snapshots').exists())
            fork = root / 'fork.bin'
            result = run('--resume', snapshots / 'iter-5.bin', '--iterations', 7, '--output', fork)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_checkpoint(fork), read_checkpoint(output))
            original = (snapshots / 'iter-10.bin').read_bytes()
            rewind = run('--resume', snapshots / 'iter-5.bin', '--reset-average',
                         '--iterations', 5, '--snapshot-every', 5, '--output', output)
            self.assertEqual(rewind.returncode, 0, rewind.stderr)
            self.assertIn('Keeping existing snapshot', rewind.stdout)
            self.assertEqual((snapshots / 'iter-10.bin').read_bytes(), original)
            self.assertFalse(list(snapshots.glob('*.tmp')))
            self.assertNotEqual(run('--snapshot-every', -1, '--iterations', 0,
                                    '--output', root / 'invalid.bin').returncode, 0)
            limited = root / 'limited.bin'
            result = run('--iterations', 10, '--samples', 10, '--max-nodes', 1,
                         '--snapshot-every', 1, '--output', limited)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_checkpoint(limited)[2], 0)
            self.assertFalse((root / 'limited-snapshots').exists())

    def test_cli_resume_exports_and_memory_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a.bin', root / 'b.bin'
            first = run('--iterations', 40, '--samples', 10, '--seed', 9, '--output', a)
            self.assertEqual(first.returncode, 0, first.stderr)
            resumed = run('--resume', a, '--iterations', 40, '--export', root / 'models')
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            continuous = run('--iterations', 80, '--samples', 10, '--seed', 9, '--output', b)
            self.assertEqual(continuous.returncode, 0, continuous.stderr)
            self.assertEqual(read_checkpoint(a), read_checkpoint(b))
            for file in ('preflop-model.json', 'postflop-model.json'):
                nodes = json.loads((root / 'models' / file).read_text())
                self.assertTrue(nodes)
                for key, values in nodes.items():
                    self.assertTrue(all(math.isfinite(x) and x >= 0 for x in values))
                    self.assertAlmostEqual(sum(values[:3]), 1)
                    self.assertGreater(values[3], 0)
                    if file.startswith('post'):
                        self.assertIsInstance(json.loads(key)[0], list)
                    else:
                        self.assertEqual(len(key.split('|')), 6)
                        self.assertIn(int(key.split('|')[-1]), (2, 3, 6, 7))
            data = a.read_bytes()
            mode, samples, iterations, count, length = struct.unpack('<IIQQI', data[8:36])
            key, row = next(iter(read_checkpoint(a)[4].items()))
            for version in (1, 2):
                legacy = root / f'legacy-{version}.bin'
                legacy.write_bytes(f'LARDCPP{version}'.encode() +
                    struct.pack('<IIQQI', mode, samples, iterations, 1, length) + data[36:36 + length] +
                    struct.pack('<I6dQ', key & ((1 << 30) - 1), *row))
                original = legacy.read_bytes()
                rejected = run('--resume', legacy, '--iterations', 1, '--reset-average')
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn('buckets merge legal actions', rejected.stderr)
                exported = run('--resume', legacy, '--iterations', 0, '--export', root / f'legacy-model-{version}')
                self.assertEqual(exported.returncode, 0, exported.stderr)
                self.assertEqual(legacy.read_bytes(), original)
            self.assertNotEqual(run('--iterations', 1, '--output', a).returncode, 0)
            self.assertNotEqual(run('--resume', a, '--samples', 11).returncode, 0)
            limited = run('--iterations', 100_000, '--samples', 10, '--memory-mb', 4, '--output', root / 'limit.bin')
            self.assertEqual(limited.returncode, 0, limited.stderr)
            self.assertIn('limit reached', limited.stderr.lower())
            checkpoint = read_checkpoint(root / 'limit.bin')
            self.assertLess(checkpoint[2], 100_000)
            self.assertTrue(checkpoint[4])
            root.joinpath('broken.bin').write_bytes(a.read_bytes()[:-4])
            self.assertNotEqual(run('--resume', root / 'broken.bin', '--iterations', 0).returncode, 0)
            self.assertFalse(a.with_suffix('.bin.tmp').exists())


if __name__ == '__main__':
    unittest.main()
