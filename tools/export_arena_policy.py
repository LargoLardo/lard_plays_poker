"""Export the committed legacy V1 policy for hosted matches using only stdlib."""

import math
from pathlib import Path
import pickle
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.shared_nodes import packed_bucket


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'Node':
            return SimpleNamespace
        return super().find_class(module, name)


def export(source, output):
    # The build reads only the trusted, committed V1 pickle; uploads are not accepted.
    with Path(source).open('rb') as stream:
        nodes = LegacyUnpickler(stream).load()
    with Path(output).open('wb') as destination:
        destination.write(b'LARDPOL1')
        destination.write(struct.pack('<IIIQQ', 1, 0, 100, 10_000_000, len(nodes)))
        for bucket, node in nodes.items():
            weights = [float(node.strategy_sum.get(action, 0)) for action in ('fold', 'check/call', 'raise')]
            if any(not math.isfinite(weight) or weight < 0 for weight in weights) or node.times_visited < 0:
                raise ValueError('Invalid V1 strategy weights or visits')
            destination.write(struct.pack('<I3dQ', packed_bucket((*bucket, 2)), *weights, node.times_visited))


if __name__ == '__main__':
    export(*sys.argv[1:])
