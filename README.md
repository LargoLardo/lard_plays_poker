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

Cards, bets, and chip stacks follow the bundled 10-million-iteration full-game
strategy. The filterable 13x13 strategy explorer lives in Study, and the
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

No-limit Texas Hold’em bots trained with **external-sampling Monte Carlo CFR** (MCCFR), built on [pokerkit](https://github.com/uoft-cs/pokerkit). The main line trains **preflop-only** (`pf_mccfr.py`) and **full-street** (`full_game_mccfr.py`) abstractions using card bucketing and pickled node stores.

## Setup

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `pokerkit`, `numpy`, `torch`, `tqdm`, `networkx`.

## Running

For a standalone, faster C++ trainer with a configurable memory budget, see
[cpp/README.md](cpp/README.md). Build and train with one command:

```bash
./cpp/run.sh --iterations 1000000 --memory-mb 256
```

| Script | Role |
|--------|------|
| `pf_mccfr.py` | Preflop MCCFR training (stops after preflop; payoffs from check-through). |
| `full_game_mccfr.py` | Full-game MCCFR training through showdown. |
| `utils/play_hand.py` | Interactive / scripted play against a loaded strategy |
| `utils/agent_test.py` | Local agent testing harness (paths/iterations are edited in-file). |
| `visualizers/*.py` | Preflop range visualization helpers. |
| `kuhn/*.py` | Small Kuhn poker CFR / MCCFR reference implementations. |
| `protos/*.py` | Earlier or alternate prototypes (Hold’em setup, random sims, CFR variants). |
| `FULLGAME_10m_iters.pkl` | Example pickled nodeset trained on 10m iterations |

The original Python trainers remain available. Install only their dependencies
with `pip install -r requirements-training.txt`, then run:

```bash
python full_game_mccfr.py --iterations 100000 --output nodesets/full-game.pkl
python full_game_mccfr.py --iterations 100000 --resume nodesets/full-game.pkl
python pf_mccfr.py --iterations 100000 --output nodesets/preflop.pkl
```

`--iterations` means additional hands. Training defaults to one worker, updating
one node store instead of copying it to every CPU. Equity and positive/negative
potential share one sampling pass and a joint cache capped at 10,000 entries
(`--cache-size 0` disables caching). `--max-nodes 200000`
stops growth after the current hand; an explicit multiworker run checks this
limit after each batch and can overshoot by a batch's new nodes. Use
`--workers 1` for minimum RAM, or explicitly choose `--workers N --chunk-size N`.
This node limit controls growth, not exact process RSS.

Checkpoints save atomically every 60 seconds and at completion, interruption, or
the node limit. Set the interval with `--checkpoint-every`; use `--resume` to
continue. New runs refuse to overwrite existing outputs. New checkpoints remain
pickle node dictionaries usable by the agents and browser exporter; old pickles
can also be read. New training averages strategies at sampled opponent nodes.
Legacy checkpoints require `--reset-average` to discard old averages and visit
counts while retaining regrets; use a different `--output` to preserve the old
file. Prefer fresh training for model comparisons. Python resumes preserve
accumulated nodes and RNG state,
but bounded equity caches are recomputed, so full-game resumes need not be
bit-for-bit identical to uninterrupted runs. Only load trusted pickle files.

New training also samples equity from cards unknown to the acting player,
uses deuce-through-ace rank order for straight-draw flags, freezes regret
matching before exploring a node's children, and caps raises using the remaining
stack plus chips already bet. Existing bundled models remain intact. Large node
sets and logs are kept **out of Git**.

Directory `nodesets/` is created locally for trained `.pkl` files referenced by the agents.

See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the Python/C++ algorithm review,
model coverage audit, measured optimizations, and remaining abstraction limits.
Audit a trusted local checkpoint without changing it:

```bash
venv/bin/python tools/audit_model.py FULLGAME_10m_iters.pkl --swap-legacy-positions
venv/bin/python tools/audit_model.py nodesets/cpp/full.bin
```

GTO solver solution for Preflop Open:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/95acaf64-fece-4b42-a582-ac05a70b3279" />

My solver solution for Preflop Open after 10 million iterations and approximately 8 hours of training:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/cc56bab4-1349-4b92-acd2-f55b8a789f14" />
