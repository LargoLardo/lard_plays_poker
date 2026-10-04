"""Memory-conscious training loops and backwards-compatible pickle stores."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import pickle
import random
import signal
import time
from multiprocessing import get_context, freeze_support

from tqdm import tqdm

from utils.bucketer import Bucketer
from utils.card_bucketer import configure_caches
from utils.shared_nodes import SharedSnapshot, BudgetExceeded, NodeDelta


class TrainingNode:
    __slots__ = ("regret_sum", "strategy_sum", "times_visited")

    def __init__(self):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.times_visited = 0

    def __setstate__(self, state):
        # Accept both the original __dict__ pickles and new slotted nodes.
        if isinstance(state, tuple):
            state = state[1]
        for name in self.__slots__ or TrainingNode.__slots__:
            setattr(self, name, state[name])

    def clone(self):
        node = type(self)()
        node.regret_sum.update(self.regret_sum)
        node.strategy_sum.update(self.strategy_sum)
        node.times_visited = self.times_visited
        return node


class NodeStore(dict):
    """Still a node dictionary for existing agents/exporters, with resume metadata."""

    iterations = 0
    rng_state = None
    trainer = None
    samples = 100
    algorithm = 1
    schema = 1
    workers = 1
    chunk_size = 64


class NodeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Node":
            return TrainingNode
        return super().find_class(module, name)


def worker_init(counter, limit):
    # The coordinator finishes/merges the batch before saving on Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    NodeDelta.counter, NodeDelta.total_limit = counter, limit


def load_nodes(path):
    # Pickles must come from trusted local training runs.
    with open(path, "rb") as source:
        nodes = NodeUnpickler(source).load()
    return nodes if isinstance(nodes, NodeStore) else NodeStore(nodes)


def save_nodes(nodes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as output:
        pickle.dump(nodes, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def train_loop(create_state, play_hand, run_chunk, merge_nodes, *, trainer,
               iters=100_000, n_workers=None, merge_every=None, output=None,
               resume=None, seed=1, samples=None, cache_size=10_000,
               max_nodes=200_000, checkpoint_every=60, reset_average=False,
               snapshot_every=0, memory_mb=256):
    nodes = load_nodes(resume) if resume else NodeStore()
    n_workers = nodes.workers if n_workers is None else n_workers
    if n_workers == 0:
        n_workers = min(256, (getattr(os, 'process_cpu_count', os.cpu_count)() or 1))
    merge_every = nodes.chunk_size if merge_every is None else merge_every
    samples = nodes.samples if samples is None else samples
    if iters < 0 or n_workers < 1 or merge_every < 1 or samples < 1:
        raise ValueError("iterations must be nonnegative; workers/chunk/samples must be positive")
    if max_nodes < 1 or cache_size < 0 or checkpoint_every < 0 or snapshot_every < 0:
        raise ValueError("max_nodes must be positive; cache/checkpoint/snapshot interval must be nonnegative")
    if n_workers > 256 or memory_mb < 4 or memory_mb > 1_048_576:
        raise ValueError("workers must be 0..256; memory budget must be 4..1048576 MiB")
    if resume and nodes.schema != 2:
        raise ValueError("Legacy checkpoint buckets merge legal actions; start a fresh run with a new --output")
    if nodes.trainer and (nodes.trainer != trainer or nodes.samples != samples):
        raise ValueError("Resume with the same trainer and sample count")
    if resume and nodes.algorithm != 2 and not reset_average:
        raise ValueError("Legacy checkpoint averages use a different algorithm; start fresh or pass --reset-average")
    if reset_average:
        if not resume:
            raise ValueError("--reset-average requires --resume")
        for node in nodes.values():
            node.strategy_sum.clear()
            node.times_visited = 0
    nodes.algorithm = 2
    nodes.schema = 2
    nodes.trainer, nodes.samples = trainer, samples
    nodes.workers, nodes.chunk_size = n_workers, merge_every
    random.seed(seed)
    if nodes.rng_state is not None:
        random.setstate(nodes.rng_state)
    configure_caches(cache_size if n_workers == 1 else 0)
    bucketer = Bucketer(samples)
    output = output or resume or f"nodesets/{trainer}.pkl"
    if not resume and Path(output).exists():
        raise FileExistsError("Output already exists; use --resume or a new --output")
    saved_at = time.monotonic()
    remaining = iters
    print(f"{trainer}: {n_workers} worker(s), {len(nodes):,}/{max_nodes:,} nodes; output {output}")
    if n_workers > 1:
        print(f"Workers share one packed regret table; {memory_mb} MiB working-buffer budget.")
    shared = SharedSnapshot(memory_mb * 1024 * 1024 // 4)
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    handlers = {}

    def checkpoint():
        nodes.rng_state = random.getstate()
        save_nodes(nodes, output)

    def save_snapshot():
        if not snapshot_every or nodes.iterations % snapshot_every:
            return
        target = Path(output)
        directory = target.parent / (target.stem + "-snapshots")
        path = directory / f"iter-{nodes.iterations}{target.suffix or '.pkl'}"
        if path.exists():
            print(f"Keeping existing snapshot {path}")
            return
        nodes.rng_state = random.getstate()
        save_nodes(nodes, path)
        print(f"Saved snapshot {path}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                old = signal.signal(signum, on_signal)
                handlers[signum] = old
            except ValueError:  # A caller may train from a non-main thread.
                break
        with tqdm(total=iters, desc="Hands", unit="hand") as progress:
            if n_workers == 1:
                # Update one store directly: no snapshot, delta store, or process queues.
                while remaining and len(nodes) < max_nodes and not interrupted:
                    play_hand(create_state(), nodes.iterations % 2, {}, nodes, bucketer)
                    nodes.iterations += 1
                    remaining -= 1
                    progress.update(1)
                    save_snapshot()
                    if time.monotonic() - saved_at >= checkpoint_every:
                        checkpoint()
                        saved_at = time.monotonic()
            elif remaining:
                # Spawn avoids inheriting the parent's Python model and works on Windows.
                context = get_context('spawn')
                counter = context.Value('Q', 0)
                # Allowance includes worker, returned, and staged Python updates.
                delta_limit = memory_mb * 1024 * 1024 * 3 // 4 // 4096
                with context.Pool(n_workers, initializer=worker_init, initargs=(counter, delta_limit)) as pool:
                    while remaining and len(nodes) < max_nodes and not interrupted:
                        previous_rng = random.getstate()
                        try:
                            jobs = n_workers * 4
                            counter.value = 0
                            snapshot = shared.publish(nodes, delta_limit)
                            args = []
                            offset = nodes.iterations
                            batch_remaining = min(remaining, n_workers * merge_every)
                            if snapshot_every:
                                batch_remaining = min(batch_remaining, snapshot_every - offset % snapshot_every)
                            for _ in range(jobs):
                                count = min((merge_every + 3) // 4, batch_remaining)
                                if not count:
                                    break
                                args.append((count, random.getrandbits(32), snapshot, offset, samples, cache_size // n_workers))
                                offset += count
                                batch_remaining -= count
                            results = pool.map(run_chunk, args, chunksize=1)
                            originals = {key: nodes.get(key) for delta, _ in results for key in delta}
                            projected = len(nodes) + sum(node is None for node in originals.values())
                            if projected > max_nodes:
                                raise BudgetExceeded('Node limit reached; raise --max-nodes')
                            shared.check_size(projected)
                            changes = {key: node.clone() for key, node in originals.items() if node is not None}
                            for delta, _ in results:
                                merge_nodes(changes, delta)
                            try:
                                nodes.update(changes)
                            except MemoryError:
                                for key, old in originals.items():
                                    if old is not None:
                                        nodes[key] = old
                                    elif key in nodes:
                                        del nodes[key]
                                raise
                        except (BudgetExceeded, MemoryError) as error:
                            random.setstate(previous_rng)
                            print(f"{error or 'Allocation refused'}; unfinished batch discarded and progress saved.")
                            break
                        count = offset - nodes.iterations
                        nodes.iterations += count
                        remaining -= count
                        progress.update(count)
                        del results, changes, originals, args, delta
                        save_snapshot()
                        if time.monotonic() - saved_at >= checkpoint_every:
                            checkpoint()
                            saved_at = time.monotonic()
    except KeyboardInterrupt:
        print("\nInterrupted; saving accumulated training.")
    finally:
        try:
            shared.close()
            checkpoint()
        finally:
            for signum, old in handlers.items():
                signal.signal(signum, old)
    if interrupted:
        print('Interrupted after the completed hand/batch; progress saved.')
    if len(nodes) >= max_nodes:
        print("Node limit reached; checkpoint saved. Raise --max-nodes to continue.")
    print(f"Saved {nodes.iterations:,} completed hands and {len(nodes):,} nodes to {output}")
    return nodes


def training_main(train):
    freeze_support()
    parser = argparse.ArgumentParser(description="Train the original Python MCCFR implementation")
    parser.add_argument("--iterations", type=int, default=100_000, help="Additional hands to train")
    parser.add_argument("--workers", type=int, help="Workers: 0 uses all cores; new runs default to 1, resumes restore settings")
    parser.add_argument("--chunk-size", type=int, help="Hands per worker between merges (new-run default 64)")
    parser.add_argument("--samples", type=int, help="Equity samples (new-run default 100; resumes restore settings)")
    parser.add_argument("--memory-mb", type=int, default=256, help="Shared table/update working-buffer budget; additional to master model/process RAM")
    parser.add_argument("--cache-size", type=int, default=10_000, help="Maximum joint equity/potential cache entries; 0 disables")
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=float, default=60, help="Seconds between atomic saves")
    parser.add_argument("--snapshot-every", type=int, default=0, help="Retain a separate checkpoint every N total hands; 0 disables")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--reset-average", action="store_true", help="Discard averages/visits on a compatible resume, retaining regrets")
    args = parser.parse_args()
    train(args.iterations, args.workers, args.chunk_size, output=args.output,
          resume=args.resume, seed=args.seed, samples=args.samples,
          cache_size=args.cache_size, max_nodes=args.max_nodes,
          checkpoint_every=args.checkpoint_every, reset_average=args.reset_average,
          snapshot_every=args.snapshot_every, memory_mb=args.memory_mb)
