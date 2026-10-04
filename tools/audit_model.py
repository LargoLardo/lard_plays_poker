"""Read-only coverage/weight audit for trusted Python and C++ training stores."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.training import load_nodes

ACTIONS = ('fold', 'check/call', 'raise')
STREETS = ('preflop', 'flop', 'turn', 'river')
RANKS = '23456789TJQKA'


def cpp_records(source):
    with open(source, 'rb') as stream:
        magic = stream.read(8)
        if magic not in (b'LARDCPP1', b'LARDCPP2', b'LARDCPP3', b'LARDCPP4'):
            raise ValueError('Unsupported C++ checkpoint')
        mode, samples = struct.unpack('<II', stream.read(8))
        workers, chunk = struct.unpack('<II', stream.read(8)) if magic == b'LARDCPP4' else (1, 64)
        iterations, count, length = struct.unpack('<QQI', stream.read(20))
        if length > 20_000:
            raise ValueError('Invalid RNG metadata')
        stream.read(length)
        metadata = dict(format=magic.decode(), mode='preflop' if mode else 'full',
                        samples=samples, iterations=iterations, algorithm=1 if magic == b'LARDCPP1' else 2,
                        schema=2 if magic in (b'LARDCPP3', b'LARDCPP4') else 1, workers=workers, chunk_size=chunk)
        yield metadata
        # Records stream without constructing a second node store.
        for _ in range(count):
            key, *row = struct.unpack('<I6dQ', stream.read(60))
            street = (key >> 17) & 3
            bucket = None
            if street == 0:
                hand = key & 65535
                suited = hand % 2
                hand //= 2
                bucket = (RANKS[hand // 13] + RANKS[hand % 13] + ('s' if suited else 'o'),
                          'SB' if (key >> 16) & 1 else 'BB',
                          ('short', 'medium', 'deep')[(key >> 27) & 3],
                          ('root', 'limped', 'vs_open', 'vs_3bet', 'vs_4bet')[(key >> 19) & 7])
            yield STREETS[street], bucket, row[:3], row[3:6], row[6]
        if stream.read(1):
            raise ValueError('Unexpected trailing checkpoint data')


def python_records(source):
    nodes = load_nodes(source)
    # Class defaults keep legacy stores loadable, but are not recorded metadata.
    metadata = vars(nodes)
    yield dict(format='pickle', iterations=metadata.get('iterations'), algorithm=nodes.algorithm,
               samples=metadata.get('samples'), schema=nodes.schema, mode=nodes.trainer or 'legacy/unknown')
    for key, node in nodes.items():
        street = 'preflop' if isinstance(key[0], str) else {5: 'flop', 7: 'turn', 4: 'river'}[len(key[0])]
        yield street, key, [node.regret_sum.get(a, 0) for a in ACTIONS], \
            [node.strategy_sum.get(a, 0) for a in ACTIONS], node.times_visited


def audit(source, swap_positions=False):
    records = cpp_records(source) if Path(source).suffix == '.bin' else python_records(source)
    report = {'source': str(source), 'metadata': next(records), 'streets': {}, 'root_hands': {}}
    visits = defaultdict(list)
    counters = defaultdict(lambda: defaultdict(int))
    roots = report['root_hands']
    for street, bucket, regrets, weights, count in records:
        visits[street].append(count)
        totals = counters[street]
        invalid = count < 0 or any(not math.isfinite(x) for x in regrets + weights) or any(x < 0 for x in weights)
        totals['invalid_nodes'] += invalid
        total = sum(weights)
        totals['no_average'] += total <= 0
        if invalid or total <= 0:
            continue
        probabilities = [x / total for x in weights]
        totals['near_pure_99pct'] += max(probabilities) >= .99
        if street == 'preflop' and bucket[3] == 'root':
            position = bucket[1]
            if swap_positions:
                position = 'SB' if position == 'BB' else 'BB'
            roots[bucket[0] + '|' + position] = dict(visits=count, probabilities=probabilities)
    for street in STREETS:
        values = visits[street]
        if not values:
            continue
        report['streets'][street] = dict(
            nodes=len(values), zero_visits=sum(v == 0 for v in values),
            under_100=sum(v < 100 for v in values), under_500=sum(v < 500 for v in values),
            median_visits=statistics.median(values), max_visits=max(values),
            **counters[street])
    sb_roots = [(hand, row) for hand, row in roots.items() if hand.endswith('|SB')]
    if sb_roots:
        combinations = [6 if hand[0] == hand[1] else 4 if hand[2] == 's' else 12 for hand, _ in sb_roots]
        report['sb_root_combo_weighted'] = {
            action: sum(n * row['probabilities'][a] for n, (_, row) in zip(combinations, sb_roots)) / sum(combinations)
            for a, action in enumerate(ACTIONS)}
        report['sb_root_hand_classes'] = len(sb_roots)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', default='FULLGAME_10m_iters.pkl')
    parser.add_argument('--swap-legacy-positions', action='store_true')
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args()
    report = audit(args.source, args.swap_legacy_positions)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'root_hands'}, indent=2))


if __name__ == '__main__':
    main()
