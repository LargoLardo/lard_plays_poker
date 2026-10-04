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
the C++ executable.

## Memory and checkpointing

One process stores each node once, using packed integer bucket keys, six doubles,
and a visit count in a flat hash table. Betting states contain fixed-size arrays
and copy directly on the stack. A direct 5–7-card evaluator avoids PokerKit state
serialization and Python hand-evaluation calls. Card features are computed once
per player/street in each traversal; no global board cache accumulates over time.

`--memory-mb` bounds the node table, its temporary replacement during growth,
and a reserved per-hand update buffer. The default is 256 MiB. Process runtime,
executable/library pages, allocator overhead, and small I/O buffers are additional;
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
hand before saving. A hard kill or power loss can lose work since the last
successful save; the temporary file is not a resume checkpoint.

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

On the development Apple Silicon machine with Clang 21, a single C++ full-game
run of 100,000 hands, 100 equity samples, and seed 7 averaged about **14,000
hands/second**, including its final checkpoint. It ended with 30,129 nodes in a
4 MiB allocated table. This is a local throughput check; stochastic trajectories,
node growth, disk speed, compiler, and machine load affect longer runs. It does
not establish convergence or playing strength.
