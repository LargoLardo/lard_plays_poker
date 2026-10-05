import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import pickle

from pokerkit import Hand
from pf_mccfr import Node

ACTION = '4bet'
PATH = _ROOT / 'checkpoints' / 'v1.pkl'
RANKS = "AKQJT98765432"
rank_to_idx = {r: i for i, r in enumerate(RANKS)}

hands = {}

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

for bucket, node in nodes.items():
    if ACTION in bucket[3]:
        hand = bucket[0]
        node_sum = sum(node.strategy_sum.values())

        if node_sum <= 0:
            continue

        hands[hand] = {
            "raise": node.strategy_sum.get("raise", 0) / node_sum,
            "check/call": node.strategy_sum.get("check/call", 0) / node_sum,
            "fold": node.strategy_sum.get("fold", 0) / node_sum,
            "visited": node.times_visited
        }

def parse_hand(hand):
    """
    Returns (row, col) indices for a 13x13 poker grid.
    Pairs go on diagonal.
    Suited go above diagonal.
    Offsuit go below diagonal.
    """
    r1, r2 = hand[0], hand[1]
    i, j = rank_to_idx[r1], rank_to_idx[r2]

    if r1 == r2:
        return i, i
    elif hand[2] == "s":
        return min(i, j), max(i, j)
    else:
        return max(i, j), min(i, j)

# Colors for each action
ACTION_COLORS = {
    "raise": "#d62728",       # red
    "check/call": "#2ca02c",  # green
    "fold": "#1f77b4",        # blue
}

fig = go.Figure()

# Keep axes in a 13x13 grid
fig.update_xaxes(
    range=[0, 13],
    tickmode="array",
    tickvals=[i + 0.5 for i in range(13)],
    ticktext=list(RANKS),
    side="top",
    showgrid=False,
    zeroline=False,
)

fig.update_yaxes(
    range=[13, 0],  # reversed
    tickmode="array",
    tickvals=[i + 0.5 for i in range(13)],
    ticktext=list(RANKS),
    showgrid=False,
    zeroline=False,
    scaleanchor="x",
    scaleratio=1,
)

# Add cell backgrounds / borders
for i in range(13):
    for j in range(13):
        fig.add_shape(
            type="rect",
            x0=j, x1=j + 1,
            y0=i, y1=i + 1,
            line=dict(color="black", width=1),
            fillcolor="white",
            layer="below",
        )

# Add action-frequency rectangles inside each cell
for hand, freqs in hands.items():
    try:
        r, c = parse_hand(hand)

        x0, x1 = c, c + 1
        y0, y1 = r, r + 1

        # stack actions vertically inside the cell
        # top: raise, middle: call, bottom: fold
        parts = [
            ("raise", freqs.get("raise", 0)),
            ("check/call", freqs.get("check/call", 0)),
            ("fold", freqs.get("fold", 0)),
        ]

        current_y = y0
        for action, frac in parts:
            if frac <= 0:
                continue

            height = frac
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=current_y,
                y1=current_y + height,
                line=dict(width=0),
                fillcolor=ACTION_COLORS[action],
            )
            current_y += height

        # Add hover + label
        fig.add_trace(go.Scatter(
            x=[c + 0.5],
            y=[r + 0.5],
            mode="text",
            text=[hand],
            textfont=dict(color="black", size=12),
            hovertemplate=(
                f"<b>{hand}</b><br>"
                f"Raise: {freqs.get('raise', 0):.1%}<br>"
                f"Call: {freqs.get('check/call', 0):.1%}<br>"
                f"Fold: {freqs.get('fold', 0):.1%}"
                f"\nTimes Visited: {freqs.get('visited', 0)}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    except Exception as e:
        print(f"Skipping hand {hand}: {e}")

fig.update_layout(
    title=f"Poker Hand Frequencies Preflop: {ACTION}",
    width=800,
    height=800,
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.show()