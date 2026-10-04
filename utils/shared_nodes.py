"""One packed, read-only regret snapshot for multiprocessing workers."""

from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
import pickle
import struct
from types import SimpleNamespace

RANKS = '23456789TJQKA'
HISTORIES = ('root', 'limped', 'vs_open', 'vs_3bet', 'vs_4bet')
SIZES = ('Limp', '~2.0bb raise', '~2.75bb raise', '~6.0bb raise',
         '~10.0bb raise', '~25.0bb raise', 'Jam (>25.00bb) raise')
ENTRY = struct.Struct('<I4x3d')
KEY = struct.Struct('<I')
ACTIONS = ('fold', 'check/call', 'raise')


class BudgetExceeded(RuntimeError):
    pass


class NodeDelta(dict):
    limit = None
    counter = None
    total_limit = 0

    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        if key not in self:
            if self.limit is not None and len(self) >= self.limit:
                raise BudgetExceeded('Worker update limit reached; reduce --chunk-size or raise --memory-mb')
            if self.counter is not None:
                with self.counter.get_lock():
                    if self.counter.value >= self.total_limit:
                        raise BudgetExceeded('Batch update limit reached; reduce --chunk-size or raise --memory-mb')
                    self.counter.value += 1
        super().__setitem__(key, value)


def packed_bucket(bucket):
    """Use the native schema-2 key layout, including legal-action flags."""
    feature, position = bucket[:2]
    stack = spr = previous = 0
    if isinstance(feature, str):
        street = 0
        hand = (RANKS.index(feature[0]) * 13 + RANKS.index(feature[1])) * 2 + (feature[2] == 's')
        stack = ('short', 'medium', 'deep').index(bucket[2])
        history, size = HISTORIES.index(bucket[3]), SIZES.index(bucket[4])
    else:
        street = {5: 1, 7: 2, 4: 3}[len(feature)]
        if street == 3:
            hand = ((feature[0] * 2 + feature[1]) * 2 + feature[2]) * 2 + feature[3]
        else:
            texture = ('monotone', 'two_tone', 'rainbow').index(feature[3])
            hand = (((feature[0] * 4 + feature[1]) * 4 + feature[2]) * 3 + texture) * 2 + feature[4]
            if street == 2:
                hand = (hand * 2 + feature[5]) * 2 + feature[6]
        history = bucket[2] if street == 1 else HISTORIES.index(bucket[2])
        size = ('small', 'medium', 'large', 'overbet').index(bucket[3])
        spr = ('short', 'mid', 'mid_deep', 'deep').index(bucket[4])
        if street > 1:
            previous = int(bucket[5])
    mask = bucket[-1]
    if mask not in (2, 3, 6, 7) or len(bucket) != (7 if street > 1 else 6):
        raise ValueError('Shared training requires schema-2 bucket keys')
    return (int(hand) | ((position == 'SB') << 16) | (street << 17) | (history << 19)
            | (size << 22) | (spr << 25) | (stack << 27) | (previous << 29)
            | ((mask & 1) << 30) | ((mask & 4) << 29))


class SharedNodes:
    def __init__(self, slots, name=None):
        self.slots = slots
        self.memory = SharedMemory(name=name, create=name is None, size=slots * ENTRY.size)

    def _offset(self, key):
        # Same integer hash as the C++ table; no Python hash randomization.
        value = key ^ (key >> 16)
        value = (value * 0x7feb352d) & 0xffffffff
        value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
        slot = (value ^ (value >> 16)) & (self.slots - 1)
        while KEY.unpack_from(self.memory.buf, slot * ENTRY.size)[0] not in (0, key + 1):
            slot = (slot + 1) & (self.slots - 1)
        return slot * ENTRY.size

    def get(self, bucket, default=None):
        key = packed_bucket(bucket)
        stored, *regrets = ENTRY.unpack_from(self.memory.buf, self._offset(key))
        return SimpleNamespace(regret_sum=dict(zip(ACTIONS, regrets))) if stored else default

    def close(self, unlink=False):
        self.memory.close()
        if unlink:
            self.memory.unlink()


class SharedSnapshot:
    """Publish a growing model only after every worker finishes the prior batch."""

    def __init__(self, budget):
        self.budget = budget
        self.table = None

    def publish(self, nodes, delta_limit):
        slots = self.check_size(len(nodes))
        if self.table is None or self.table.slots < slots:
            replacement = SharedNodes(slots)
            if self.table:
                self.table.close(unlink=True)
            self.table = replacement
        for bucket, node in nodes.items():
            key = packed_bucket(bucket)
            ENTRY.pack_into(self.table.memory.buf, self.table._offset(key), key + 1,
                            *(node.regret_sum.get(action, 0.0) for action in ACTIONS))
        return self.table.memory.name, self.table.slots, delta_limit

    def check_size(self, count):
        slots = 1024
        while count > slots * 7 // 10:
            slots *= 2
        if self.table is None or self.table.slots < slots:
            old_bytes = self.table.memory.size if self.table else 0
            if old_bytes + slots * ENTRY.size > self.budget:
                raise BudgetExceeded('Shared-table limit reached; raise --memory-mb')
        return slots

    def close(self):
        if self.table:
            self.table.close(unlink=True)
            self.table = None


@contextmanager
def read_snapshot(snapshot):
    # Accept old private run_chunk inputs for comparison with shared lookup.
    if isinstance(snapshot, bytes):
        yield pickle.loads(snapshot), NodeDelta()
        return
    name, slots, limit = snapshot
    table = SharedNodes(slots, name)
    try:
        yield table, NodeDelta(limit)
    finally:
        table.close()
