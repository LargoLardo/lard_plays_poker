# Python and C++ training review

## C++ V5 implementation and validation

V5 is now the recommended native model on `feat/cpp-model-v5`. Python training
and the original C++ trainer are legacy implementations kept separately. V5
uses a fresh checkpoint format, `LARDCPP5`; old models remain available for
Play, Study, Arena, export, and V4 training. No old checkpoint was migrated or
overwritten. Commits are dated October 5, 2026, 14:15 Toronto at the user's request.

V5 implements deterministic, suit-canonical visible-card features, a frozen
offline learned abstraction embedded in every checkpoint, full public action
history, five explicit actions with consistent legal masks, and Linear
external-sampling MCCFR. Native inference supplies the same card buckets to
the existing browser. The flat shared table, bounded assignment/update caches,
transactional batches, and worker pool retain controlled memory use.

The card vector combines a 16-bin future-equity CDF, eight opponent-class
strength estimates, and eight made-hand/blocker/board features. River equity
enumerates all 990 remaining opponent combinations. Earlier streets sample up
to 32 runouts and opponents within each runout. CDF distance is the 1-D earth
mover distance, and cluster updates use coordinate medians for those dimensions.
The defaults are 50k examples/street, 256/512/256 clusters, 512 samples, and six
rounds. Preflop retains all 169 starting-hand classes. Feature sampling uses a
separate state-derived seed and never consumes the training RNG.

Distributional and opponent-class features follow the ideas in
[Johanson et al., Evaluating State-Space Abstractions in Extensive-Form Games](https://webdocs.cs.ualberta.ca/~mbowling/papers/13aamas-abstraction.pdf).
This implementation uses simple rank/suitedness opponent classes rather than
their learned OCHS classes. Its final-strength distribution is also simpler
than the hierarchical transitions in
[Ganzfried and Sandholm's potential-aware abstraction](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf).
It should be described as a distribution-based abstraction with blocker
features, rather than a reproduction of that potential-aware algorithm.

The Linear update follows the sampling-compatible experiment in
[Brown and Sandholm, Discounted Monte Carlo CFR](https://arxiv.org/html/1809.04040):
every 10M nodes touched (including terminals), regrets and average weights are
discounted by `n/(n+1)`. V5 applies this at successful batch boundaries and
telescopes skipped periods. Visits are raw coverage counters. `--algorithm
vanilla` supplies a fresh-run control; feature and algorithm settings are
frozen on resume. This is Linear MCCFR, without clipping cumulative regrets
as CFR+ would. The paper's sampled case motivates that choice.

The finalized public tree contains 13,608 decision histories: 20 preflop,
252 flop, 2,156 turn, and 11,180 river. With the default clusters, at most
4,033,844 information sets fit this fixed 100-BB action tree. The node table
stores 104-byte entries with 16-byte keys and grows without eviction. Default
working budget is 4 GiB, including rehash peaks, cache and update allowances;
runtime stacks and allocator overhead are additional. Default 5M cap exceeds
this tree's upper bound. A maximal checkpoint is about 385 MiB, so retained
snapshots need substantially more disk space than V4 snapshots.

### Measured local results

On the 12-CPU, 48-GiB M4 Pro, the offline default asset used about 10 MiB peak
RSS. A fresh 100k-hand run at 12 workers/chunk 64 reached 1,611,326 nodes,
32,412 hands/s including two snapshots, and 862 MiB peak RSS. Continuing to
1M hands with snapshots every 250k reached 3,627,169 nodes, 47,059 additional
hands/s, and **1,440 MiB peak RSS**. The final file is about 346 MiB. These
short local measurements include saves and are not billion-hand projections.

Three fresh Linear and vanilla runs used the same asset, 12 workers, chunk 64,
and ten seconds of training before SIGINT finished the batch and saved. Each
run took about 11.5 seconds including its save. Each matchup below uses
100k hands with seat swaps and identical deals; positive values favor Linear.

| Training seed | Linear hands | Vanilla hands | Linear BB/100 | Approx. 95% interval |
|---|---:|---:|---:|---|
| 7 | 696,576 | 704,256 | +43.08 | +29.20 to +56.97 |
| 19 | 696,576 | 706,560 | +45.67 | +32.00 to +59.33 |
| 31 | 672,000 | 576,768 | +62.77 | +48.35 to +77.20 |

These early comparisons support retaining Linear as the default in this
implementation. They do not establish later convergence, and the third pair
had noticeably different throughput despite equal time budgets.

The saved 1M-hand V5 model also produced the following paired results:

| Opponent | Match hands | V5 BB/100 | Approx. 95% interval |
|---|---:|---:|---|
| V4 at 500M hands (seed 19, legacy features at 500 samples) | 100,000 | +135.03 | +120.49 to +149.56 |
| Always check/call (seed 17) | 50,000 | +589.18 | +559.43 to +618.94 |
| Random legal raise sizes (seed 17) | 50,000 | +349.14 | +321.53 to +376.75 |
| Random actions with pot raises (seed 17) | 50,000 | +273.90 | +250.18 to +297.61 |

V5's trained-average coverage against V4 was 99.37%; V4's was 86.21%.
That gap matters: V5 has more betting choices, and some V4 responses fall back
outside its original tree. V5 translated 13,299 observed actions and had 1,106
off-tree decisions in that match. Random sizing caused 4,069 off-tree decisions
out of 82,387 V5 decisions. Thus those match gains include action-abstraction
and fallback effects. They are not isolated evidence that the card clusters
alone improved play. Baselines are deliberately weak controls.

The 1M-node audit found no invalid weights and all 169 SB root hand classes.
Many deeper nodes still have few visits. V5 Study excludes individual nodes
under 1,000 visits before combining histories into its preflop projection.
Arena records missing/unaveraged coverage, translated actions, and off-tree
decisions. Exact selfplay produces zero paired gain and zero translation.

Local evidence stays under ignored `artifacts/`: `v5-training*.log`,
`v5-variant-benchmark.json`, `v5-linear-vanilla-seed*.json`,
`v5-v4-500samples.json`, baseline reports, and `audit-v5-1m.json`.
The full checkpoints live in `checkpoints/`; iteration snapshots remain outside Git.

### Checks and remaining limits

All 39 Python/native tests pass, including exact Treys river comparisons,
suit/hidden-card invariance, every public history, native/browser action-size
parity, serial and parallel checkpoint resume, asset freezing, malformed data,
memory rollback, cross-model Arena, and inference-process lifecycle.
V5 unit checks pass under AddressSanitizer, UndefinedBehaviorSanitizer, and
ThreadSanitizer. Browser checks cover native five-action play, arbitrary human
sizes, stale inference replies, fallback, existing streets/scene, Study, Arena,
mobile, reduced motion, and WebGL fallback. Local native inference is exercised
through `tools/serve.py`; Windows/MSVC and Linux execution await CI.

V5 retains full **public** history but forgets earlier private buckets. Its
card abstraction remains imperfect recall. Starting stacks are always 100 BB;
other browser bankrolls are outside that trained game. Observed arbitrary
raises translate to nearest abstract pot fractions. A virtual history that
cannot follow the actual actor/street uses Play's check/call fallback or
Arena's configured fallback. This is still an abstract game, not unrestricted
no-limit Hold'em or a measured GTO solver.

The next experiments are held-out card-abstraction validation, hierarchical
potential-aware transitions or learned opponent classes, starting-stack
coverage, and reduced-game best-response evaluation of the new native path.
Regret pruning or float compression should follow measurements of convergence
and numerical error. The existing Kuhn best-response check validates its own
reference solver; it does not measure V5 Hold'em exploitability.

## Earlier Python and V4 review

Reviewed October 4, 2026: `pf_mccfr.py`, `full_game_mccfr.py`, the standalone C++
trainer in both modes, the bundled `checkpoints/v1.pkl`, and local C++ v1/v2/v3/v4
checkpoints. Original models and the Python implementations are preserved.

External-sampling MCCFR is a sound foundation for heads-up, zero-sum poker.
Keeping regrets, accumulated strategy weights, and visits per information set
is also a sensible design. The C++ packed keys and flat table make that design
much cheaper to run. The main limitations are the information/action abstraction
and the absence of Hold'em strength measurements. These models should not yet
be described as validated GTO strategies.

The MCCFR convergence results assume perfect recall and consistent actions
within each information set. The legal-action inconsistency is now fixed.
The buckets still lack perfect recall. See the [original MCCFR paper](https://www.cs.cmu.edu/~waugh/publications/nips09b.pdf).

## Correctness changes applied to both implementations

Both Python trainers and the original C++ port accumulated an unweighted average
at traverser nodes. The standard two-player external-sampling update accumulates
the average at sampled opponent nodes, where sampling supplies the player's
reach weighting; regrets update at traverser nodes. Both implementations and
the Kuhn reference now follow that rule. [OpenSpiel's reference implementation](https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/python/algorithms/external_sampling_mccfr.py)
documents and implements the same separation.

New Python stores record algorithm version 2 and bucket schema 2; native
checkpoints use `LARDCPP4` (V3 keys plus worker/chunk metadata). Every training
key includes the exact available fold/call/raise mask, after checking the prescribed or sampled raise amount.
The C++ mask uses two spare key bits, preserving the 32-bit key and 64-byte
hash-table entry. Exporters, Python agents, and browser lookup use the mask;
approximate browser lookups never cross into another new action mask. Legacy
models remain readable for play, export, and inspection. Training requires a
fresh output: neither regrets nor averages can be separated reliably from old
merged buckets, including with `--reset-average`.

Exhaustively walking the 100bb full-game betting tree visits 2,314 decision
states. With each actor/street card feature fixed, the old 174 context keys
include 24 contexts with differing legal-action masks. Schema 2 produces 198
contexts with zero such collisions. A Python regression also recreates the same
river deal with two `SB/vs_4bet/small/short` situations: facing an all-in allows
fold/call, while facing a raise allows fold/call/raise. They now learn in separate
nodes. The action-aware upper bound is 269,466 nodes for this fixed full-game
tree; arbitrary browser sizing and preflop-only sampling are different trees.

Historical Python code also excluded hidden opponent/burn cards from equity
sampling, used the wrong rank order for straight-draw flags, and capped a raise
using remaining chips instead of chips plus the existing bet. Those issues were
fixed in the preceding training work. Changing code cannot retroactively fix the
bundled model; the strength impact of each historical issue is unmeasured.

## Measured efficiency changes

Native profiling placed about 95% of runtime in postflop feature generation.
Further hash-table tuning would therefore have little effect on full-game
throughput. The changes target equity computation:

- C++ tracks rank multiplicities with bitmasks and extracts high ranks without
  scanning all 13 ranks. It also shares board rank/suit counts between hero and
  villain, reuses hero's river score, and looks up straights in an 8 KiB table covering all rank masks.
- Python estimates equity and both potentials in one sample pass, reuses the
  fixed hero scores, and stores the three results in one bounded joint cache.
  Flop/turn bucketing still uses one pass with caching disabled.
- Python's cache cap remains 10,000 entries by default, now for one cache rather
  than two. Parallel Python workers divide that cap and read a shared packed
  regret table rather than separate unpickled models. Native card features stay local to a traversal. The existing native
  memory budget, transactional hand rollback, and streamed checkpoints remain.

Local measurements on Apple Silicon / Apple Clang 21, median of three runs:

| Full-game trainer | Before, hands/s | After, hands/s | Throughput increase | After peak RSS |
|---|---:|---:|---:|---:|
| Python, one worker | 25.65 | 29.07 | 13.3% | 44.9 MiB |
| C++ | 13,210.8 | 15,418.9 | 16.7% | 9.5 MiB |

Both sides use the corrected averaging algorithm, seed 7, and 100 equity
samples. Native runs each complete 100,000 hands and save a final checkpoint;
each ends with 33,992 nodes in a 4 MiB table. Python measurements each run 250
full-game traversals without saving. Python's joint samples change correlations,
RNG consumption, and paths: before/after runs ended with 5,471/5,346 nodes. These
are throughput measurements, not equal-convergence or strength comparisons.
No preflop-only speedup is claimed from removing postflop feature work.

The native optimization preserves results exactly: the 100,000-hand checkpoints
before and after have identical settings, RNG state, and every node's regrets,
average weights, and visits. File record order is irrelevant to this comparison.
Timing and RSS depend on hardware, compiler, workload, and machine load. Short
runs do not measure the maximum memory of a long training job. Python's node
limit can overshoot by a hand or a multiworker batch; native allocation budgets
are not operating-system RSS limits.

The action-aware V3 evaluator was additionally benchmarked before/after the
multiplicity-mask change: three 100,000-hand runs each, seed 7, with exact RNG,
regrets, average weights, and visit equality. At 100 equity samples, median
throughput rose from 16,623 to 32,394 hands/s (+94.9%); at 500 samples it rose
from 3,821 to 7,105 hands/s (+85.9%). All 12,005 Treys differential rankings
still agree. These speedups preserve the abstraction and sampled outcomes.

The C++ trainer now supports a persistent CPU thread pool with `--workers N`
(`0` detects all cores). Workers read one frozen shared model per batch, keep
bounded local deltas, and merge in deterministic task order. The coordinator
assigns independent RNG seeds before dispatch, so scheduling does not affect
results. Update memory is partitioned among the workers and merge buffer;
no complete model is copied per worker. Snapshot boundaries, batch rollback,
and worker/chunk checkpoint metadata are covered by regressions. V3 files
remain resumable; V1/V2 cannot resume because their keys lack action masks.
This is batched MCCFR: serial and parallel trajectories differ, and stronger
convergence per second still needs playing-strength evaluation.

V4 scaling on this 12-core Mac (8 performance/4 efficiency cores), medians of
three 100,000-hand runs, seed 7, chunk size 64, including final saves:

| Workers | 100 samples, hands/s | 500 samples, hands/s | Peak RSS at 500 samples |
|---|---:|---:|---:|
| 1 | 32,681 | 7,098 | 9.5 MiB |
| 4 | 98,606 | 25,646 | 11.0 MiB |
| 8 | 151,490 | 47,223 | 11.9 MiB |
| 12 | 167,704 | 57,137 | 13.1 MiB |

At 500 samples, 12 workers give 8.0× serial throughput and average 9.2 busy CPU
cores. At 100 samples the increase is 5.1×; synchronization/merging takes a
larger share of wall time. Repetitions with identical worker/chunk settings
produce identical checkpoint hashes. Full model copies are avoided, but these
short runs do not bound peak memory once every possible node has been reached.

Python parallel mode now publishes one read-only packed regret table with
32-byte slots using the same schema-2 integer keys. Spawned workers read it
directly, retain bounded local updates, and return deltas for a deterministic
merge. The coordinator stages changed nodes before applying a batch. Its
working-buffer budget covers the shared table and a common update-node allowance
across tasks, permitting uneven tasks to share free space. Master dictionary,
interpreter/library RAM, caches, and allocator overhead are additional; this is
not a total RSS cap. Node limits or buffer failures roll back the whole batch
and coordinator RNG. Worker/chunk/sample settings persist in Python checkpoints.

A read-only capacity test with the 56,143-node bundled model, four spawned
workers, and 1,003 queried keys measured peak worker RSS around 152 MiB with
copied dictionaries versus 50 MiB with shared regrets. The shared table was
4 MiB, and all queried regret sums matched. This isolates snapshot memory;
it is not a long training memory bound or a playing-strength measurement.
Both Python worker implementations also match dictionary-based fixed-seed
traversals exactly, including RNG state and all node updates. A 12-worker Python
full-game smoke run completed 768 hands at 100 equity samples and saved exact
384/768-hand snapshots. Python still uses PokerKit and its original evaluator,
so the C++ throughput gains do not apply to Python.

Live interruption checks on macOS sent SIGINT to serial Python training and
SIGTERM to a three-worker run. Both saved completed hands/batches, matched an
uninterrupted reference including RNG state and all updates, and resumed.

A portable native launcher, MSVC bit scans, and Windows checkpoint replacement
remove the shell/GCC-only assumptions. The training CI matrix includes Windows,
Linux, and macOS; it has not run locally on Windows. Workers can be configured
from 1 to 256, with `0` detecting available logical CPUs.

## Audit of the original saved model

`checkpoints/v1.pkl` contains **56,143 nodes**. Its name indicates 10M
iterations, but the legacy pickle records neither an iteration count nor the
sample count. All inspected numeric values are finite and strategy weights are
nonnegative. The table below counts recorded visits, not independent samples:

| Street | Nodes | No average weights | Visits below 100 | Visits below 500 | Median visits |
|---|---:|---:|---:|---:|---:|
| Preflop | 2,028 | 0 | 90 | 651 | 1,047.5 |
| Flop | 8,418 | 638 | 4,900 | 6,344 | 48.5 |
| Turn | 41,347 | 4,175 | 29,305 | 35,916 | 18 |
| River | 4,350 | 2 | 323 | 983 | 2,507.5 |

Turn uses 73.6% of stored nodes, yet 70.9% of its nodes have fewer than 100
visits and 86.9% have fewer than 500. Flop also has 58.2% below 100. This is strong
evidence that the abstraction spreads training too thinly. The browser blends
nodes below 100 visits with heuristics; the Python agent uses a 500-visit scale
and keeps a small prior even for mature nodes. Thus actual play depends on
fallback behavior as well as learned averages.

After explicitly swapping historical SB/BB labels, all 169 SB opening classes
are covered. Weighting hand classes by their number of physical combinations,
the stored opening policy is **11.60% fold, 40.94% limp, 47.46% raise**. It shows
learned behavior rather than a uniform policy. Mixed limping can be reasonable
in a restricted action tree; these frequencies alone establish neither good
poker nor a defect. A corrected exporter can swap labels, but cannot repair the
averages or historical features used during training.

## Audit of native checkpoints

The inspected v1 and corrected v2 checkpoints each trained 100,000 hands with
100 samples and seed 7. All inspected values are finite and strategy weights
are nonnegative. They are much shorter runs than the bundled Python model.

| Street | v1 nodes | v1 median visits | v2 nodes | v2 median visits | v2 without average weights |
|---|---:|---:|---:|---:|---:|
| Preflop | 1,690 | 60 | 1,690 | 100.5 | 156 |
| Flop | 4,571 | 14 | 5,054 | 10 | 572 |
| Turn | 19,570 | 5 | 22,918 | 4 | 4,090 |
| River | 4,298 | 65 | 4,330 | 81.5 | 323 |

V1 counted visits during regret updates; v2 counts average updates at opponent
nodes and also stores nodes reached only for regret updates. Visit counts and
zero-average counts therefore cannot be compared as equivalent confidence
measures. V2's zero-average nodes require fallback behavior and are omitted from
browser exports. In the corrected checkpoint, 93.1% of turn nodes have fewer
than 100 average updates. More training is needed before assessing its policy.
The corrected SB root frequencies are 31.24% fold, 29.66% limp, 39.10% raise;
all 169 opening classes are covered. Comparing those directly to the bundled
10M model would confound algorithm, training duration, and stochastic paths.

## Remaining limits shared by Python and C++

1. **The buckets forget information and action history.** Postflop keys discard
   original hole ranks/suit blockers, prior street card buckets, preflop pot
   type, and most past betting history. Turn/river keep only a boolean indicating
   whether the actor raised on the previous street. This is imperfect recall;
   ordinary perfect-recall CFR guarantees cannot be assumed for these buckets.

2. **There is only one raise action per state.** Full mode uses 3x the current
   bet preflop and the current bet plus half the current total pot postflop,
   subject to rounding/capping and a late-raise shove rule. There are no separate
   small, large, overbet, or shove choices to learn between. Preflop-only mode
   samples several fractions but puts them all into the same `raise` action;
   it cannot learn a sizing preference. Preflop-only payoffs assume check-through
   after preflop, so they do not represent optimal postflop continuation values.

3. **Card abstraction is noisy and misses range/blocker information.** Equity
   is measured against uniform unknown opponents, not a betting-conditioned
   range. Using this as a feature does not force the learned policy to assume a
   uniform opponent, but grouping hands by this feature loses strategic detail.
   At 100 samples, nearby bin assignments can vary between encounters/restarts.
   Turn has 3,072 possible card-feature combinations versus 768 on the flop and
   64 on the river, contributing to sparse coverage. Deterministic, suit-aware
   canonical features would also improve training/inference consistency.

4. **Fresh hands always start at 100bb each.** Midhand stack/SPR buckets do not
   substitute for training games starting at 20bb, 40bb, or 200bb. The browser
   carries bankroll between hands and accepts arbitrary human sizing, relying on
   approximate lookups and heuristics outside the trained game.

5. **Hold'em strength has not been measured.** Evaluator correctness, finite
   weights, coverage, and hands/second cannot establish exploitability. The
   added Kuhn check enumerates exact best responses: at 30,000 iterations its
   average-policy value is -0.05532734 (equilibrium -1/18), with exploitability
   0.00677301 chips/hand. This checks the small reference solver, not the Hold'em
   model. Even the old averaging rule can look plausible on a small toy game.

## Recommended next work

Next retain a coherent public betting history and balance turn
abstraction resolution against the available training budget and make card
features deterministic. Add a small set of explicit raise sizes with stack-aware
translation. Assess each change with fixed-deal, seat-swapped matches against
several baselines, confidence intervals, and exact best-response evaluation on a
tractable reduced poker game. Compare both equal-hand and equal-time budgets.
Use those measurements to judge batching, regret pruning, or other regret
variants: higher throughput alone does not establish convergence per second.

Start a separate corrected native run:

```bash
./cpp/run.sh --iterations 1000000 --samples 100 --memory-mb 256 --output checkpoints/v2.bin
```

Read-only audits support trusted Python pickles and native v1/v2/v3/v4 checkpoints:

```bash
venv/bin/python tools/audit_model.py checkpoints/v1.pkl --swap-legacy-positions --output artifacts/audit-bundled.json
venv/bin/python tools/audit_model.py checkpoints/v2.bin --output artifacts/audit-native.json
```

Validation includes 31 Python/native tests and browser policy checks, with
12,005 Treys evaluator comparisons, 250 PokerKit betting sequences, averaging-phase
checks, legacy resume guards, checkpoint/resume, memory rollback, and the joint
sample/cache regression. Native serial/parallel unit checks also pass under
AddressSanitizer, UndefinedBehaviorSanitizer, and ThreadSanitizer. Benchmark logs
and audit reports remain local
under ignored `artifacts/`. Temporary test/benchmark nodesets were subsequently
removed during cleanup; bundled models remain preserved.
