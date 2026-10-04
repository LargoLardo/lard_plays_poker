# Standalone C++ MCCFR trainer

This is a separate implementation of the original heads-up, 100bb Hold'em
trainer. The Python trainers, prototypes, and bundled 10M model remain in the
repository. It needs a C++17 compiler (Clang or GCC) and no Python packages or
third-party C++ libraries.

From the repository root:

```bash
./cpp/run.sh --iterations 1000000 --memory-mb 256
```

The script builds an optimized executable in `cpp/build/` on the first run and
when its sources change. Set `CXX` to choose a compiler. Training defaults to
full-game mode, 100 equity samples, seed 1, and `nodesets/cpp/full.bin`.
All paths supplied to the executable are relative to your current directory.

Resume, run preflop only, or choose a separate output:

```bash
./cpp/run.sh --resume nodesets/cpp/full.bin --iterations 1000000
./cpp/run.sh --mode preflop --iterations 1000000 --output nodesets/cpp/preflop.bin
./cpp/run.sh --iterations 100000 --samples 50 --memory-mb 128 --output nodesets/cpp/experiment.bin
./cpp/run.sh --help
```

`--iterations` is the number of additional hands. Resume restores the original
mode, sample count, completed-hand count, node weights, and random-generator
state. It reproduces an uninterrupted run on the same compiler/standard-library
build. An explicit conflicting mode or sample count is rejected. Starting a new
run refuses to overwrite an existing checkpoint. C++ checkpoints are versioned
binary files; Python pickle checkpoints stay separate and cannot be resumed by
the C++ executable. Version 3 (`LARDCPP3`) distinguishes exact legal-action sets
using two spare bits in the existing 32-bit key; node size is unchanged. It keeps
version 2's standard average update at sampled opponent nodes. Versions 1/2
remain readable for inspection/export, but cannot resume training, even with
`--reset-average`: their merged regrets cannot be split reliably. Start a fresh
run with a new output, or export an old checkpoint without rewriting it:

```bash
./cpp/run.sh --iterations 1000000 --output nodesets/cpp/full-v4.bin
./cpp/run.sh --resume nodesets/cpp/old.bin --iterations 0 --export nodesets/cpp/old-web-model
```

New saves use `LARDCPP4` to retain worker count and chunk size as well. V3
checkpoints have compatible keys and can resume; they upgrade to V4 on save.
The browser and Python agents prefer the matching action mask, with fallback
to legacy keys for old models. They never use a new node with a different mask.

## Using more CPU cores

`--workers 0` uses all detected CPU cores; choose `--workers 8` to leave more
CPU capacity available for other work. Serial training remains the default.
On the development Mac, all cores means 12 workers (8 performance and 4
efficiency cores):

```bash
caffeinate -i ./cpp/run.sh --workers 0 --iterations 100000000 --samples 500 \
  --memory-mb 256 --snapshot-every 1000000 --output nodesets/cpp/full-v4.bin
./cpp/run.sh --resume nodesets/cpp/full-v4.bin --iterations 100000000 --snapshot-every 1000000
```

Workers share one immutable node table during each batch and keep bounded local
updates. A persistent thread pool distributes smaller tasks across cores; the
coordinator merges results in a fixed order. The default `--chunk-size 64`
allows up to `workers × 64` hands between merges. Smaller values give fresher
shared regrets with more synchronization; larger values increase update delay
and local memory requirements. Batches stop exactly at snapshot milestones and
the requested hand count. Ctrl+C finishes the batch and saves. If allocation
or node limits fail, the entire batch and its RNG draws roll back.

V4 checkpoints restore workers/chunk size automatically. Explicit options can
override them. With the same settings and batch boundaries, repeated and resumed
runs reproduce node values and RNG state on the same compiler/library build.
Changing workers, chunk size, or boundaries changes the training trajectory.
Parallel training reads shared regrets at batch boundaries, so it is a batched
MCCFR variant; throughput gains do not establish equal convergence per hand.

## Memory and checkpointing

One process stores each node once, using packed integer bucket keys, six doubles,
and a visit count in a flat hash table. Betting states contain fixed-size arrays
and copy directly on the stack. A direct 5–7-card evaluator avoids PokerKit state
serialization and Python hand-evaluation calls. Card features are computed once
per player/street in each traversal; no global board cache accumulates over time.

`--memory-mb` bounds the node table, its temporary replacement during growth,
and a reserved per-hand update buffer. The default is 256 MiB. Process runtime,
executable/library pages, thread stacks, allocator overhead, and small I/O buffers are additional;
this is an allocation budget, not an operating-system RSS limit. Use a budget
comfortably below your available RAM. `--max-nodes N` can impose a lower cap.

If a whole hand would exceed the node/update-buffer limit, its updates and
random draws are discarded, and the last completed training state is saved.
Nodes are never evicted or silently frozen to squeeze more hands into memory.
Resume with a larger budget or cap to continue. A checkpoint too large for the
requested budget is rejected without overwriting it.

Checkpoints stream to a temporary file and replace the previous checkpoint
after a successful close. They save every 60 seconds (`--checkpoint-every N`),
at completion, at a limit, and after Ctrl+C/SIGTERM. Signals finish the current
hand (or parallel batch) before saving. A hard kill or power loss can lose work since the last
successful save; the temporary file is not a resume checkpoint.

## Keeping models at different training stages

Add `--snapshot-every N` to retain a separate nodeset every N completed hands,
alongside the main checkpoint. For example, a 100M-hand run with snapshots every
1M hands produces 100 snapshots plus the main checkpoint:

```bash
caffeinate -i ./cpp/run.sh --iterations 100000000 --samples 500 --memory-mb 256 \
  --snapshot-every 1000000 --output nodesets/cpp/full-100m-s500.bin
```

The main checkpoint is `nodesets/cpp/full-100m-s500.bin`. Retained checkpoints
are grouped in `nodesets/cpp/full-100m-s500-snapshots/`, named
`iter-1000000.bin`, `iter-2000000.bin`, and so on. Counts are cumulative across
resumes; snapshots contain the exact completed-hand count, node weights, and RNG
state. They stream to disk without duplicating the model in RAM. Existing
snapshot paths are kept and reported rather than replaced. `--snapshot-every 0`
(the default) disables retained snapshots. Repeat the option when resuming:

```bash
./cpp/run.sh --resume nodesets/cpp/full-100m-s500.bin --iterations 100000000 --snapshot-every 1000000
```

The main checkpoint still saves on its seconds-based interval, at the end of a
run, on interruption, and at memory limits, including between snapshot milestones.
There is no configured snapshot-count limit; available disk space and save time
are the constraints. With the current 100bb full-game action tree, an upper
bound of 269,466 action-aware keys gives roughly 16.2 MB per native snapshot, or about 1.6 GB
for 100 snapshots. Actual files are usually smaller. Different training stages
do not guarantee increasing playing strength; evaluate them before assigning
difficulty levels.

To continue from an earlier snapshot, use a separate `--output` so the original
snapshot stays intact. A zero-iteration export can read any snapshot as well:

```bash
./cpp/run.sh --resume nodesets/cpp/full-100m-s500-snapshots/iter-10000000.bin \
  --iterations 10000000 --snapshot-every 1000000 --output nodesets/cpp/fork.bin
./cpp/run.sh --resume nodesets/cpp/full-100m-s500-snapshots/iter-10000000.bin \
  --iterations 0 --output nodesets/cpp/export-source.bin --export nodesets/cpp/model-10m
```

## Using a trained model

Export the exact JSON key/weight format used by the existing browser app:

```bash
./cpp/run.sh --resume nodesets/cpp/full.bin --iterations 0 --export nodesets/cpp/web-model
```

This writes `preflop-model.json` and `postflop-model.json` without copying the
entire export into memory. Inspect these files first. To publish a chosen model,
copy them into `public/` and rebuild the web app. Training and the commands above
do not replace the bundled browser model.

The action abstraction follows the Python trainers: fold, check/call, and one
raise size. Full mode uses 3× the current bet preflop and half pot postflop;
preflop mode samples the original pot fractions and checks through later streets.
Both use the corrected raise-to cap (stack plus existing bet), position mapping,
and straight-draw rank order. Equity is sampled from all cards unknown to the
acting player, including possible opponent cards and burns. C++ reuses the same
Monte Carlo samples for equity/potential and the same sampled full deal across
branches, so its stochastic training path is not identical to Python's.
Lowering `--samples` changes abstraction accuracy as well as throughput; keep it
constant when resuming. This remains an abstracted MCCFR trainer, not a claim
that a particular number of hands produces an optimal strategy.

## Checks and measured performance

Run the native engine, rehash, checkpoint/resume, and memory-rollback checks:

```bash
./cpp/test.sh
```

For differential tests against PokerKit/Treys, including 12,005 sampled 5–7-card
hands, 250 betting sequences, browser exports, malformed checkpoints, and a
4 MiB budget stop:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements-training.txt
venv/bin/python -m unittest discover -s tests -v
```

The rank-multiplicity bitmask evaluator improved median serial throughput from
16,623 to 32,394 hands/s at 100 equity samples (+94.9%), and from 3,821 to 7,105
hands/s at 500 samples (+85.9%). These matched V3 runs use 100,000 hands, seed 7,
and three repetitions, including checkpoint saves; every node value and RNG
state matches before/after. Earlier shared-board/straight-table results are
recorded in [TRAINING_REVIEW.md](../TRAINING_REVIEW.md).

Parallel V4 measurements on the 12-core development Mac, median of three runs
of 100,000 hands, seed 7, chunk size 64, including the final checkpoint:

| Workers | 100 samples, hands/s | 500 samples, hands/s | Peak RSS at 500 samples |
|---|---:|---:|---:|
| 1 | 32,681 | 7,098 | 9.5 MiB |
| 4 | 98,606 | 25,646 | 11.0 MiB |
| 8 | 151,490 | 47,223 | 11.9 MiB |
| 12 | 167,704 | 57,137 | 13.1 MiB |

Twelve workers provided 5.1× throughput at 100 samples and 8.0× at 500 samples.
The latter averaged about 9.2 busy cores (920% process CPU); merging and uneven
task costs leave some idle time. Identical configurations produced identical
checkpoint hashes on all repetitions. Different worker counts use different
training trajectories, so these compare wall time per hand, not convergence.

These are local throughput checks. Node growth, compiler, hardware, and machine
load affect longer runs; the results do not establish convergence or playing
strength. See [the Python/C++ training review](../TRAINING_REVIEW.md) for the
saved-model coverage audit and remaining information/action abstraction limits.
