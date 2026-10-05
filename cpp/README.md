# C++ poker training

## V5 (recommended for new runs)

V5 is the new C++ model. The original C++ V4 and Python trainers remain available
as legacy implementations; their checkpoints are preserved. V5 needs a fresh
training run because its information sets and actions differ.

Build a frozen card abstraction once, then train:

```bash
python cpp/run.py --model v5 --build-abstraction nodesets/cpp/v5-cards.abs --workers 0
python cpp/run.py --model v5 --abstraction nodesets/cpp/v5-cards.abs \
  --iterations 10000000 --snapshot-every 1000000 --output nodesets/cpp/full-v5.bin
```

The local development checkout already has this asset and a 1M-hand V5 checkpoint.
Continue it with:

```bash
python cpp/run.py --model v5 --resume nodesets/cpp/full-v5.bin \
  --iterations 10000000 --snapshot-every 1000000
```

`--iterations` means additional hands. Use `venv/bin/python` on the development
Mac, or `py -3` in a Windows Developer Command Prompt. The same C++17 launcher
supports MSVC, Clang, and GCC. It defaults to V4 when `--model` is omitted, and
`run.sh` continues to launch V4. Always select `--model v5` for the new trainer.

V5 defaults to all detected CPUs (1–256), a **4 GiB working allocation budget**,
a 256 MiB bounded assignment cache, and a 5M-node cap. The table is shared across
threads. Node growth, rehash peaks, local updates, and assignment caches are
budgeted; runtime/thread stacks and allocator overhead are additional. Nodes
are never evicted or frozen. A failed hand/batch and its RNG draws roll back,
then the last completed state is saved. On a small budget, reduce `--cache-mb`
and `--chunk-size`; `--memory-mb 128 --cache-mb 4 --workers 4 --chunk-size 4`
is useful for small test abstractions. More workers do not multiply the model.

The asset defaults to 50,000 examples per street, 256/512/256 flop/turn/river
clusters, 512 samples, and six clustering rounds. Suit-equivalent visible cards
produce identical deterministic features and bucket IDs. Features combine a
16-bin CDF of future equity, strength against eight opponent hand classes, and
made-hand/blocker/board features. River equity enumerates all 990 possible
opponent hands. The CDF uses earth mover distance; other components use squared
distance. Preflop retains the exact 169 starting-hand classes. Opponent classes
are simple rank/suitedness groups, rather than the learned classes from the
OCHS paper. Flop/turn estimates remain sampled. The asset, feature seed, and
sample count are embedded in each checkpoint and frozen on resume/inference.

Each node retains the full public action sequence, acting seat, street, card
cluster, and legal-action mask. Five choices are fold, check/call, small raise,
pot raise, and all-in. The small opening raise is 2.5 BB; later preflop raises
are 3× the current bet. Postflop small raises are half pot. Pot sizes include
the call, amounts round to half-BB units, and duplicate/illegal choices vanish.
After two raises per street, only the jam raise remains. This produces **13,608
decision histories** at 100 BB. Card abstraction still has imperfect recall;
preserving public history does not supply a full-game convergence guarantee.

The default is **Linear external-sampling MCCFR**. After each 10M nodes touched,
completed batches discount regrets and average sums by the elapsed-period
ratio; raw visits remain unchanged. `--algorithm vanilla` starts a control run.
`--discount-every N` changes the period for a fresh run. Algorithm and period
are recorded and cannot change on resume. Serial resume is exact; parallel
resume matches an uninterrupted run with the same worker/chunk settings and
batch boundaries. Snapshots split batches, so keep the same snapshot schedule
when comparing trajectories. Ctrl+C finishes the current batch and saves.

Select a V5 checkpoint in the existing local browser via **••• → Nodeset**.
Play uses a persistent native inference process with the embedded abstraction;
it sends only the agent's cards, visible board, and public actions. Study shows
a preflop projection combining the raise choices, excluding each source node
below 1,000 visits before aggregation. Postflop policies stay native, so no
large JSON export or approximate JavaScript card bucketer is needed. A V5
selection requires `tools/serve.py`; static hosting keeps the bundled model.

Observed bets outside the trained action choices translate to the nearest
pot fraction in a virtual 100-BB game. Arena and Play apply chosen sizes to the
actual legal game. If translated history can no longer follow the real street
or actor, Play checks/calls; Arena uses the configured missing-node fallback.
Arena reports translated actions and off-tree decisions. Browser bankrolls
other than 100 BB remain outside the trained starting-stack game.

```bash
python agent_arena.py nodesets/cpp/full-v5.bin nodesets/cpp/full-v4.bin \
  --hands 100000 --seed 17 --output artifacts/arena/v5-v4.json
python agent_arena.py nodesets/cpp/full-v5.bin baseline:random --hands 100000
python tools/audit_model.py nodesets/cpp/full-v5.bin
python cpp/run.py --model v5 --test
```

Arena also offers `baseline:call` and `baseline:pot`. `--samples` changes legacy
arena estimates, while V5 always uses its frozen features. These match results
measure performance against the selected opponent; they are not exploitability.
See [the review and measured validation](../TRAINING_REVIEW.md).

## Legacy V4 trainer

This is a separate implementation of the original heads-up, 100bb Hold'em
trainer. The Python trainers, prototypes, and bundled 10M model remain in the
repository. It needs a C++17 compiler (Clang, GCC, or MSVC) and no Python packages or
third-party C++ libraries.

For a launcher that also works on Windows, use Python 3.10+ and a C++17 compiler:

```bash
python cpp/run.py --workers 0 --iterations 1000000 --memory-mb 256
python cpp/run.py --test
```

On Windows, run `py -3 cpp/run.py ...` from a Visual Studio Developer Command
Prompt (MSVC `cl`), or install GCC/Clang and set `CXX` if needed. The launcher
uses MSVC flags and `.exe` filenames on Windows; it also supports `clang-cl`.
The native code uses MSVC rank intrinsics and Windows checkpoint replacement
with [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw),
so repeated saves can replace an existing checkpoint. The shell launcher below
continues to work on macOS/Linux without requiring Python.

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

`--workers 0` uses all detected logical CPU cores (up to 256); explicit counts
1–256 are supported, with no requirement for exactly 12 cores. On the
development Mac, choose `--workers 8` to leave more
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

The training workflow runs Python/shared-memory, native engine, checkpoint/resume,
and policy checks on Windows/MSVC, Linux/GCC, and macOS/Clang. Platform results
are available once the workflow runs after a push; local validation here used
macOS/Clang and the Python `spawn` multiprocessing context.
