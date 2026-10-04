# Python and C++ training review

Reviewed October 4, 2026: `pf_mccfr.py`, `full_game_mccfr.py`, the standalone C++
trainer in both modes, the bundled `FULLGAME_10m_iters.pkl`, and local C++ v1/v2
checkpoints. Original models and the Python implementations are preserved.

External-sampling MCCFR is a sound foundation for heads-up, zero-sum poker.
Keeping regrets, accumulated strategy weights, and visits per information set
is also a sensible design. The C++ packed keys and flat table make that design
much cheaper to run. The main limitations are the information/action abstraction
and the absence of Hold'em strength measurements. These models should not yet
be described as validated GTO strategies.

The MCCFR convergence results assume perfect recall and consistent actions
within each information set. The current buckets do not satisfy those
conditions in general. See the [original MCCFR paper](https://www.cs.cmu.edu/~waugh/publications/nips09b.pdf).

## Correctness changes applied to both implementations

Both Python trainers and the original C++ port accumulated an unweighted average
at traverser nodes. The standard two-player external-sampling update accumulates
the average at sampled opponent nodes, where sampling supplies the player's
reach weighting; regrets update at traverser nodes. Both implementations and
the Kuhn reference now follow that rule. [OpenSpiel's reference implementation](https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/python/algorithms/external_sampling_mccfr.py)
documents and implements the same separation.

New Python stores record algorithm version 2; native checkpoints use `LARDCPP2`.
Legacy stores remain readable for play, export, and inspection. Resuming them
requires explicit `--reset-average`, which discards average weights and visits
but retains regrets. This is a warm start, not a repair or validation of legacy
regrets. Use a different output to preserve an old checkpoint; fresh training is
preferred for evaluating quality.

Historical Python code also excluded hidden opponent/burn cards from equity
sampling, used the wrong rank order for straight-draw flags, and capped a raise
using remaining chips instead of chips plus the existing bet. Those issues were
fixed in the preceding training work. Changing code cannot retroactively fix the
bundled model; the strength impact of each historical issue is unmeasured.

## Measured efficiency changes

Native profiling placed about 95% of runtime in postflop feature generation.
Further hash-table tuning would therefore have little effect on full-game
throughput. The changes target equity computation:

- C++ shares board rank/suit counts between hero and villain, reuses hero's river
  score, and looks up straights in an 8 KiB table covering all rank masks.
- Python estimates equity and both potentials in one sample pass, reuses the
  fixed hero scores, and stores the three results in one bounded joint cache.
  Flop/turn bucketing still uses one pass with caching disabled.
- Python's cache cap remains 10,000 entries by default, now for one cache rather
  than two. Native card features stay local to a traversal. The existing native
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

## Audit of the original saved model

`FULLGAME_10m_iters.pkl` contains **56,143 nodes**. Its name indicates 10M
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

1. **Some bucket keys merge incompatible legal actions.** Exhaustively walking
   the current full-game betting tree from 100bb visits 2,314 decision states.
   Holding each actor/street card feature fixed gives 174 context keys, of which
   24 encounter different legal-action masks. For example, two reachable river
   SB states have the same `vs_4bet`, `small`, `short`, prior-aggressor context:

   | State, BB/SB values in bb | Stacks | Current bets | Total pot | Available actions |
   |---|---|---|---:|---|
   | Facing BB's all-in | 0 / 55 | 93 / 38 | 145 | Fold, call |
   | Facing BB's raise | 55 / 75 | 38 / 18 | 70 | Fold, call, raise |

   Both can occur with the same deal and therefore the same card feature. Their
   common earlier actions are preflop call/call; flop BB check, SB bet 1, BB
   call; turn BB check, SB bet 2, BB raise to 5, SB call. The first river line is
   BB check, SB bet 7, BB raise to 18, SB raise to 38, BB shove to 93. The second
   is BB bet 7, SB raise to 18, BB raise to 38. C++ and Python use matching
   context rules. Filtering illegal actions at runtime keeps play legal, but
   does not separate the accumulated learning. A fix needs a versioned key
   change coordinated across trainers, exports, and inference.

2. **The buckets forget information and action history.** Postflop keys discard
   original hole ranks/suit blockers, prior street card buckets, preflop pot
   type, and most past betting history. Turn/river keep only a boolean indicating
   whether the actor raised on the previous street. This is imperfect recall;
   ordinary perfect-recall CFR guarantees cannot be assumed for these buckets.

3. **There is only one raise action per state.** Full mode uses 3x the current
   bet preflop and the current bet plus half the current total pot postflop,
   subject to rounding/capping and a late-raise shove rule. There are no separate
   small, large, overbet, or shove choices to learn between. Preflop-only mode
   samples several fractions but puts them all into the same `raise` action;
   it cannot learn a sizing preference. Preflop-only payoffs assume check-through
   after preflop, so they do not represent optimal postflop continuation values.

4. **Card abstraction is noisy and misses range/blocker information.** Equity
   is measured against uniform unknown opponents, not a betting-conditioned
   range. Using this as a feature does not force the learned policy to assume a
   uniform opponent, but grouping hands by this feature loses strategic detail.
   At 100 samples, nearby bin assignments can vary between encounters/restarts.
   Turn has 3,072 possible card-feature combinations versus 768 on the flop and
   64 on the river, contributing to sparse coverage. Deterministic, suit-aware
   canonical features would also improve training/inference consistency.

5. **Fresh hands always start at 100bb each.** Midhand stack/SPR buckets do not
   substitute for training games starting at 20bb, 40bb, or 200bb. The browser
   carries bankroll between hands and accepts arbitrary human sizing, relying on
   approximate lookups and heuristics outside the trained game.

6. **Hold'em strength has not been measured.** Evaluator correctness, finite
   weights, coverage, and hands/second cannot establish exploitability. The
   added Kuhn check enumerates exact best responses: at 30,000 iterations its
   average-policy value is -0.05532734 (equilibrium -1/18), with exploitability
   0.00677301 chips/hand. This checks the small reference solver, not the Hold'em
   model. Even the old averaging rule can look plausible on a small toy game.

## Recommended next work

First separate legal-action contexts and retain a coherent public betting
history, using a new schema for both training and play. Then balance turn
abstraction resolution against the available training budget and make card
features deterministic. Add a small set of explicit raise sizes with stack-aware
translation. Assess each change with fixed-deal, seat-swapped matches against
several baselines, confidence intervals, and exact best-response evaluation on a
tractable reduced poker game. Compare both equal-hand and equal-time budgets.
Do that before adding more workers, pruning regrets, or switching regret
variants: those changes need evidence that they improve convergence per second.

Start a separate corrected native run:

```bash
./cpp/run.sh --iterations 1000000 --samples 100 --memory-mb 256 --output nodesets/cpp/full-v2.bin
```

Read-only audits support trusted Python pickles and native v1/v2 checkpoints:

```bash
venv/bin/python tools/audit_model.py FULLGAME_10m_iters.pkl --swap-legacy-positions --output artifacts/audit-bundled.json
venv/bin/python tools/audit_model.py nodesets/cpp/full-v2.bin --output artifacts/audit-native.json
```

Validation: all 18 Python/native differential tests pass, including 12,005
Treys evaluator comparisons, 250 PokerKit betting sequences, averaging-phase
checks, legacy resume guards, checkpoint/resume, memory rollback, and the joint
sample/cache regression. Native unit checks also pass under AddressSanitizer
and UndefinedBehaviorSanitizer. Benchmarks and inspected native checkpoints
remain local under ignored `artifacts/` and `nodesets/`; bundled models were not
modified or replaced.
