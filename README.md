# lard plays poker!

## Play in the browser

The browser table puts you in a first-person poker room, holding your cards
above green felt and a dark wood rail. A single overhead light picks out
the table and your opponent's hands; the rest of the room fades into darkness.
Rigged white-gloved hands hold your cards and gesture when checking or betting.
Beveled chips slide into bets, collect between streets, and move to the winner;
folds and showdown reveals are animated too. The scene respects reduced-motion
preferences. The hand meshes are bundled locally under the MIT license; see
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

Training scripts load and save pickle node dictionaries; default paths are configured inside each script. Large node sets and logs are kept **out of Git** (see below).

Directory `nodesets/` is created locally for trained `.pkl` files referenced by the agents.

GTO solver solution for Preflop Open:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/95acaf64-fece-4b42-a582-ac05a70b3279" />

My solver solution for Preflop Open after 10 million iterations and approximately 8 hours of training:


<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/cc56bab4-1349-4b92-acd2-f55b8a789f14" />
