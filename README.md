# lard plays poker!

## Play in the browser

The browser table puts you in a first-person poker room, holding your cards
above green felt and a dark wood rail. A single overhead light picks out
the table and your opponent's hands; the rest of the room fades into darkness.
Rigged white-gloved hands hold your cards and gesture when checking or betting.
Beveled chips slide into bets, collect between streets, and move to the winner.
Cards leave a visible deck in sequence, and showdown uses staggered, full card
turnovers. Deals keep moving through quick actions; pots wait for the reveal.
The scene respects reduced-motion preferences. The hand meshes are bundled locally under the MIT license; see
[asset provenance](public/assets/hands/SOURCE.md).

Cards, bets, and chip stacks follow the selected strategy, starting with the
bundled 10-million-iteration full-game model. The filterable 13x13 strategy explorer lives in Study, and the
player's stack persists between completed hands in browser storage.
Browsers without WebGL use a playable flat table automatically.

For the quickest local development server, run:

```bash
npm install
npm run dev
```

Then open `http://localhost:8000`. The build bundles Three.js locally, so no
CDN is needed to load the scene. After `npm run build`, you can also serve
`public/` with any static server. To reproduce Vercel's development routing,
run `npx vercel dev` instead. You can also import the repository into Vercel and
deploy with the default settings. No environment variables or database are
required. Run `npm run test:web` for the strategy checks. For the browser
checks, run `npx playwright install chromium` once, then `npm run test:ui`.
The browser checks cover betting, every street, showdown, mobile layout,
the strategy explorer, and the fallback table. Screenshots go to `artifacts/`.

Open **••• → Nodeset** to choose a local `.bin` or `.pkl` checkpoint, including
iteration snapshots under `nodesets/`. Study updates immediately; Play uses the
selection from the next hand. The browser remembers your choice. Refresh the
page to discover newly saved checkpoints. The local server exports only the
chosen checkpoint and caches its JSON under `artifacts/web-models/`; it reads
the training files without changing them.

New **V5 C++** selections use native inference with their frozen card abstraction.
Study shows a compact preflop projection; Play retains all five action choices
and uses native postflop buckets. Arena also offers check/call, random legal
sizing, and pot-bet baselines. Translated bets and off-tree decisions appear in
its coverage results. See [V5 training and model details](cpp/README.md#v5-recommended-for-new-runs).

Study greys out nodes with fewer than **1,000 visits**. A mixed-context hand cell
includes only nodes meeting that threshold, and the frequency summary excludes
sparse nodes too. This affects Study display only.

The **Arena** tab compares two checkpoints using duplicate deals: each deal is
played twice with the agents swapping seats, starting with 100 BB each. It shows
net BB, BB/100, an approximate 95% interval based on duplicate pairs, wins/losses/
ties, and trained-node coverage. It uses the saved average strategies directly.
Missing or unaveraged nodes use uniform legal actions; preflop-only agents
check/call later streets. The bundled legacy model gets its seat correction
automatically. Arena runs through the same local server.

For a standalone match and a saved report, use the new `agent_arena.py`:

```bash
python agent_arena.py nodesets/cpp/full-v4.bin nodesets/cpp/full-v4-snapshots/iter-100000000.bin --hands 100000 --seed 1 --output artifacts/arena/latest-vs-100m.json
```

Both `.bin` and `.pkl` checkpoints are supported. The native engine runs the
match; Python policies are converted in a temporary directory. Inputs stay
unchanged. Hand counts must be even. `--samples 0` (default) uses the larger
recorded sample count, with 100 assumed for legacy Python files lacking metadata.
Use `--fallback call` for check/call fallback instead. To compare the original
`FULLGAME_10m_iters.pkl`, add `--swap-a-legacy-positions` or
`--swap-b-legacy-positions` according to its argument position. Ctrl+C finishes
the current duplicate pair and returns a partial report. These results measure
the selected matchup and evaluation settings.

If the scene is already built, start the same frontend directly with
`venv/bin/python tools/serve.py` (Windows: `py -3 tools/serve.py`). C++ selections
need the same C++17 compiler as training; Python selections need the training
dependencies. `npm run dev` now uses this server. A plain static server or
the deployed static site offers the bundled model; local checkpoint selection
uses the `/api/nodesets` routes supplied by `tools/serve.py`.

Open the browser developer console while playing to inspect the agent trace.
Each hand logs an ISO timestamp and Lard's cards; each decision logs its
interpreted model bucket, node visits, strategy source, normalized
fold/call/raise frequencies, and sampled action. Mature nodes (100+ visits) use
the model unchanged; sparse nodes blend with heuristics in proportion to their
missing visits. Agent actions use a 1.5-second delay so they are readable in the
table UI.

`vercel.json` explicitly sets the Framework Preset to **Other** so Vercel does
not mistake the repository's offline Python training scripts for a Python web
application. It builds and publishes only the static `public/` directory.

The compact browser model can be regenerated after training with:

```bash
python tools/export_web_model.py nodesets/your_model.pkl
```

For node sets trained before the heads-up position mapping fix, add
`--swap-legacy-positions`. The bundled 10M model has already been exported
with this correction.

No-limit Texas Hold’em bots trained with **external-sampling Monte Carlo CFR**
(MCCFR). **C++ V5 is the recommended trainer for new models.** The Python
trainers built on [pokerkit](https://github.com/uoft-cs/pokerkit) and native V4
remain available as legacy implementations with their original models intact.

## Setup

V5 training needs Python 3.10+ for the launcher and a C++17 compiler; it has no
Python package dependencies. The packages below support legacy trainers and tools.

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `pokerkit`, `numpy`, `torch`, `tqdm`, `networkx`.

## Running

For the new C++ model, build a frozen abstraction once and then train:

```bash
python cpp/run.py --model v5 --build-abstraction nodesets/cpp/v5-cards.abs
python cpp/run.py --model v5 --abstraction nodesets/cpp/v5-cards.abs \
  --iterations 10000000 --snapshot-every 1000000 --output nodesets/cpp/full-v5.bin
```

V5 defaults to all CPUs and a 4 GiB working budget. It uses deterministic suit
canonicalization, learned card clusters, full public action histories, several
raise sizes, and Linear MCCFR. Existing V4/Python checkpoints cannot be migrated
into these new information sets; start fresh. The development checkout already
has a 1M-hand V5 model; continue with `python cpp/run.py --model v5 --resume
nodesets/cpp/full-v5.bin --iterations 10000000 --snapshot-every 1000000`.
See [cpp/README.md](cpp/README.md) for compiler setup, memory settings, and resume
details. `cpp/run.sh` and the Python launcher without `--model v5` retain V4.

| Script | Role |
|--------|------|
| `cpp/train_v5.cpp` | Recommended full-game C++ Linear MCCFR trainer. |
| `cpp/trainer.cpp` | Legacy native V4 trainer, kept separately. |
| `pf_mccfr.py` | Legacy Python preflop MCCFR training (check-through payoffs). |
| `full_game_mccfr.py` | Legacy Python full-game MCCFR training. |
| `utils/play_hand.py` | Interactive / scripted play against a loaded strategy |
| `utils/agent_test.py` | Local agent testing harness (paths/iterations are edited in-file). |
| `agent_arena.py` | C++/Python checkpoint matches with duplicate deals and JSON results. |
| `visualizers/*.py` | Preflop range visualization helpers. |
| `kuhn/*.py` | Small Kuhn poker CFR / MCCFR reference implementations. |
| `protos/*.py` | Earlier or alternate prototypes (Hold’em setup, random sims, CFR variants). |
| `FULLGAME_10m_iters.pkl` | Example pickled nodeset trained on 10m iterations |

The original Python trainers stay unchanged as legacy. Install only their dependencies
with `pip install -r requirements-training.txt`, then run:

```bash
python full_game_mccfr.py --iterations 100000 --output nodesets/full-game.pkl
python full_game_mccfr.py --iterations 100000 --resume nodesets/full-game.pkl
python pf_mccfr.py --iterations 100000 --output nodesets/preflop.pkl
```

`--iterations` means additional hands. Training defaults to one worker.
Both Python trainers accept `--workers 0` to detect available logical CPUs,
or `--workers N` for an explicit count (1–256). Parallel workers use one packed
shared regret table instead of unpickling a full model each. Processes start
with `spawn` on every platform. The original PokerKit training logic remains;
this is pure Python, and C++ is still much faster.

The default `--chunk-size 64` allows up to `workers × 64` hands per batch;
smaller tasks balance work across cores. Results merge in a fixed order, so
scheduling does not change a successful run. Worker count, chunk size, samples,
nodes, and RNG state are restored on resume. Changing batching changes the
training trajectory; parallel throughput alone does not establish strength.

Equity and positive/negative potential share one sampling pass and a joint cache
capped at 10,000 entries (`--cache-size 0` disables caching). That cap is divided
among parallel workers. `--max-nodes 200000` stops serial growth after the current
hand, which can overshoot by a hand. Parallel mode checks the projected node
count before applying a whole batch; a failed batch and its RNG draws roll back.
Ctrl+C/SIGTERM finishes the current hand/batch and saves.

Python's `--memory-mb 256` budgets **additional working buffers**, including the
shared table and conservative update-node allowances across all tasks. The
master Python dictionary, interpreter/library RAM per process, caches, and
allocator overhead are additional. It is not a total RSS cap or the same budget
as C++'s packed model table. `--workers 1` minimizes Python process overhead;
`--workers 8` leaves more CPU capacity available on the 12-core development Mac.

Python 3.10+ training runs on Windows, macOS, and Linux. For example, from
Windows Command Prompt or PowerShell:

```powershell
py -3 -m pip install -r requirements-training.txt
py -3 full_game_mccfr.py --workers 0 --iterations 1000000 --samples 100 --memory-mb 256 --output nodesets/full-game-shared.pkl
```

The shared table uses the standard library's [cross-process shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html).
Use the same flags with `pf_mccfr.py` for preflop training. Tests use the Windows
`spawn` process model locally; the training CI also includes a Windows runner.

Checkpoints save atomically every 60 seconds and at completion, interruption, or
the node limit. Set the interval with `--checkpoint-every`; use `--resume` to
continue. New runs refuse to overwrite existing outputs. New checkpoints remain
pickle node dictionaries usable by the agents and browser exporter; old pickles
can also be read. New training averages strategies at sampled opponent nodes
and records schema 2: each bucket includes its exact legal-action mask.
Legacy checkpoints remain usable for play/export, but training needs a fresh
output: merged regrets cannot be separated with `--reset-average`.
Python resumes preserve
accumulated nodes and RNG state,
but bounded equity caches are recomputed, so full-game resumes need not be
bit-for-bit identical to uninterrupted runs. Only load trusted pickle files.

Both Python trainers also accept `--snapshot-every N` to retain a separate
nodeset every N completed hands. For example:

```bash
python full_game_mccfr.py --iterations 1000000 --snapshot-every 100000 --output nodesets/full-game.pkl
```

This keeps the main checkpoint and writes `nodesets/full-game-snapshots/iter-100000.pkl`,
`iter-200000.pkl`, and so on. The default `0` disables snapshots. Counts are
cumulative; repeat the flag when resuming. Scheduled snapshots preserve existing
files. To train from an earlier snapshot, give a separate `--output` to preserve
that snapshot. Multiworker batches stop at exact snapshot boundaries, which can
change batching and the training trajectory. There is no fixed number of
snapshots; disk space and write time are the constraints. These are training
stages, and playing strength needs separate evaluation. The C++ equivalent is
documented in [cpp/README.md](cpp/README.md#keeping-models-at-different-training-stages).

New training also samples equity from cards unknown to the acting player,
uses deuce-through-ace rank order for straight-draw flags, freezes regret
matching before exploring a node's children, and caps raises using the remaining
stack plus chips already bet. Existing bundled models remain intact. Large node
sets and logs are kept **out of Git**.

Directory `nodesets/` is created locally for trained `.pkl` files referenced by the agents.

See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the Python/C++ algorithm review,
model coverage audit, measured optimizations, and remaining abstraction limits.
The native trainer supports `--workers 0` to use all CPU cores while sharing one
node table; [CPU options and resume details](cpp/README.md#using-more-cpu-cores).
`python cpp/run.py` builds/runs C++ on Windows, macOS, and Linux; Windows needs
MSVC in a Developer Command Prompt or an installed GCC/Clang compiler.
Audit a trusted local checkpoint without changing it:

```bash
venv/bin/python tools/audit_model.py FULLGAME_10m_iters.pkl --swap-legacy-positions
venv/bin/python tools/audit_model.py nodesets/cpp/full.bin
```

GTO solver solution for Preflop Open:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/95acaf64-fece-4b42-a582-ac05a70b3279" />

My solver solution for Preflop Open after 10 million iterations and approximately 8 hours of training:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/cc56bab4-1349-4b92-acd2-f55b8a789f14" />
