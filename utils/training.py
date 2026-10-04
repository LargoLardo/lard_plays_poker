"""Memory-conscious training loops and backwards-compatible pickle stores."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import pickle
import random
import time
from multiprocessing import Pool

from tqdm import tqdm

from utils.bucketer import Bucketer
from utils.card_bucketer import configure_caches


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


class NodeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Node":
            return TrainingNode
        return super().find_class(module, name)


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
               iters=100_000, n_workers=1, merge_every=1000, output=None,
               resume=None, seed=1, samples=100, cache_size=10_000,
               max_nodes=200_000, checkpoint_every=60):
    n_workers = n_workers or 1
    if iters < 0 or n_workers < 1 or merge_every < 1 or samples < 1:
        raise ValueError("iterations must be nonnegative; workers/chunk/samples must be positive")
    if max_nodes < 1 or cache_size < 0 or checkpoint_every < 0:
        raise ValueError("max_nodes must be positive; cache/checkpoint interval must be nonnegative")
    nodes = load_nodes(resume) if resume else NodeStore()
    if nodes.trainer and (nodes.trainer != trainer or nodes.samples != samples):
        raise ValueError("Resume with the same trainer and sample count")
    nodes.trainer, nodes.samples = trainer, samples
    random.seed(seed)
    if nodes.rng_state is not None:
        random.setstate(nodes.rng_state)
    configure_caches(cache_size)
    bucketer = Bucketer(samples)
    output = output or f"nodesets/{trainer}.pkl"
    if not resume and Path(output).exists():
        raise FileExistsError("Output already exists; use --resume or a new --output")
    saved_at = time.monotonic()
    remaining = iters
    print(f"{trainer}: {n_workers} worker(s), {len(nodes):,}/{max_nodes:,} nodes; output {output}")
    if n_workers > 1:
        print("Parallel workers each hold a node snapshot; use --workers 1 for minimum RAM.")

    def checkpoint():
        nodes.rng_state = random.getstate()
        save_nodes(nodes, output)

    try:
        with tqdm(total=iters, desc="Hands", unit="hand") as progress:
            if n_workers == 1:
                # Update one store directly: no snapshot, delta store, or process queues.
                while remaining and len(nodes) < max_nodes:
                    play_hand(create_state(), nodes.iterations % 2, {}, nodes, bucketer)
                    nodes.iterations += 1
                    remaining -= 1
                    progress.update(1)
                    if time.monotonic() - saved_at >= checkpoint_every:
                        checkpoint()
                        saved_at = time.monotonic()
            else:
                with Pool(n_workers) as pool:
                    while remaining and len(nodes) < max_nodes:
                        # Freeze once so merging early results cannot change queued inputs.
                        snapshot = pickle.dumps(nodes, protocol=pickle.HIGHEST_PROTOCOL)
                        args = []
                        offset = nodes.iterations
                        for _ in range(n_workers):
                            count = min(merge_every, remaining)
                            if not count:
                                break
                            args.append((count, random.getrandbits(32), snapshot, offset, samples, cache_size))
                            offset += count
                            remaining -= count
                        for delta, count in pool.imap(run_chunk, args, chunksize=1):
                            merge_nodes(nodes, delta)
                            nodes.iterations += count
                            progress.update(count)
                        del snapshot, args, delta
                        if time.monotonic() - saved_at >= checkpoint_every:
                            checkpoint()
                            saved_at = time.monotonic()
    except KeyboardInterrupt:
        print("\nInterrupted; saving accumulated training.")
    finally:
        checkpoint()
    if len(nodes) >= max_nodes:
        print("Node limit reached; checkpoint saved. Raise --max-nodes to continue.")
    print(f"Saved {nodes.iterations:,} completed hands and {len(nodes):,} nodes to {output}")
    return nodes


def training_main(train):
    parser = argparse.ArgumentParser(description="Train the original Python MCCFR implementation")
    parser.add_argument("--iterations", type=int, default=100_000, help="Additional hands to train")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--cache-size", type=int, default=10_000, help="Maximum entries in each equity cache; 0 disables")
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=float, default=60, help="Seconds between atomic saves")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.iterations, args.workers, args.chunk_size, output=args.output,
          resume=args.resume, seed=args.seed, samples=args.samples,
          cache_size=args.cache_size, max_nodes=args.max_nodes,
          checkpoint_every=args.checkpoint_every)
