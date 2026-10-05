"""Compare two saved poker policies using duplicate deals and the native engine."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile

from cpp.run import build


def python_policy(source, output):
    from utils.training import load_nodes
    from utils.shared_nodes import packed_bucket

    nodes = load_nodes(source)
    schemas = {2 if len(key) == (6 if isinstance(key[0], str) or len(key[0]) == 5 else 7) else 1 for key in nodes}
    if len(schemas) > 1:
        raise ValueError('Cannot compare a checkpoint mixing legacy and action-mask buckets')
    schema = next(iter(schemas), nodes.schema)
    if nodes.trainer not in (None, 'full-game', 'preflop'):
        raise ValueError('Unsupported Python trainer')
    with Path(output).open('wb') as destination:
        destination.write(b'LARDPOL1')
        destination.write(struct.pack('<IIIQQ', schema, nodes.trainer == 'preflop', nodes.samples,
                                      nodes.iterations, len(nodes)))
        for bucket, node in nodes.items():
            key = packed_bucket(bucket if schema == 2 else (*bucket, 2))
            weights = [float(node.strategy_sum.get(action, 0)) for action in ('fold', 'check/call', 'raise')]
            if any(not math.isfinite(weight) or weight < 0 for weight in weights) or node.times_visited < 0:
                raise ValueError('Invalid strategy weights or visits')
            destination.write(struct.pack('<I3dQ', key, *weights, node.times_visited))
    return {'iterations': vars(nodes).get('iterations')}


def run_match(a, b, *, hands=10_000, seed=1, samples=0, fallback='uniform',
              memory_mb=None, swap_a_legacy_positions=False, swap_b_legacy_positions=False):
    if hands < 2 or hands % 2:
        raise ValueError('Hands must be even and at least 2 (two legs per duplicate deal)')
    if samples < 0 or samples > 1_000_000 or seed < 0 or seed >= 2**64:
        raise ValueError('Samples must be 0..1000000 and seed an unsigned 64-bit integer')
    if fallback not in ('uniform', 'call'):
        raise ValueError('Fallback must be uniform or call')
    paths = [str(path) if str(path) in ('baseline:call', 'baseline:random', 'baseline:pot') else Path(path).resolve() for path in (a, b)]
    for path in paths:
        if isinstance(path, str):
            continue
        if path.suffix not in ('.bin', '.pkl') or not path.is_file():
            raise ValueError(f'Choose an existing .bin or .pkl checkpoint: {path}')
    with tempfile.TemporaryDirectory(prefix='lard-arena-') as directory:
        policies = []
        metadata = {}
        for index, path in enumerate(paths):
            if isinstance(path, str):
                policies.append(path)
            elif path.suffix == '.pkl':
                converted = Path(directory) / f'{index}.policy'
                metadata[index] = python_policy(path, converted)
                policies.append(converted)
            else:
                policies.append(path)
        if memory_mb is None:
            memory_mb = max(256, (max((path.stat().st_size for path in policies if isinstance(path, Path)), default=0) * 8 + 1_048_575) // 1_048_576)
        if memory_mb < 4 or memory_mb > 1_048_576:
            raise ValueError('Memory budget must be 4..1048576 MiB per agent')
        command = [str(build(arena=True)), '--a', str(policies[0]), '--b', str(policies[1]),
                   '--hands', str(hands), '--seed', str(seed), '--samples', str(samples),
                   '--fallback', fallback, '--memory-mb', str(memory_mb)]
        if swap_a_legacy_positions:
            command.append('--swap-a-legacy-positions')
        if swap_b_legacy_positions:
            command.append('--swap-b-legacy-positions')
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            while True:
                try:
                    stdout, stderr = process.communicate()
                    break
                except KeyboardInterrupt:
                    if os.name != 'nt':
                        process.send_signal(signal.SIGINT)
            if process.returncode:
                raise RuntimeError(stderr.strip() or 'Arena failed')
        report = json.loads(stdout)
        for index, (label, path) in enumerate(zip(('a', 'b'), paths)):
            report[label]['checkpoint'] = str(path)
            report[label].update(metadata.get(index, {}))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', help='Checkpoint A (.bin/.pkl) or baseline:call/random/pot')
    parser.add_argument('b', help='Checkpoint B (.bin/.pkl) or baseline:call/random/pot')
    parser.add_argument('--hands', type=int, default=10_000, help='Even hand count, split into duplicate pairs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples', type=int, default=0, help='Equity samples; 0 uses the larger recorded training count')
    parser.add_argument('--fallback', choices=('uniform', 'call'), default='uniform', help='Policy for missing/unaveraged nodes')
    parser.add_argument('--memory-mb', type=int, help='Native working budget per agent; defaults based on checkpoint size')
    parser.add_argument('--swap-a-legacy-positions', action='store_true')
    parser.add_argument('--swap-b-legacy-positions', action='store_true')
    parser.add_argument('--output', type=Path, help='Optional .json report file')
    args = parser.parse_args()
    if args.output and args.output.suffix != '.json':
        parser.error('--output must be a .json report path')
    try:
        report = run_match(args.a, args.b, hands=args.hands, seed=args.seed, samples=args.samples,
                           fallback=args.fallback, memory_mb=args.memory_mb,
                           swap_a_legacy_positions=args.swap_a_legacy_positions,
                           swap_b_legacy_positions=args.swap_b_legacy_positions)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, args.output)
    print(text, end='')


if __name__ == '__main__':
    main()
