"""
MCCFR – External Sampling – Kuhn Poker
=======================================
External sampling variant (Lanctot et al. 2009):
  - Chance node   : sample one deal per iteration
  - Traversing player : explore ALL actions (update regrets)
  - Opponent          : SAMPLE one action from current strategy

This is proper MCCFR — the game tree is never fully traversed.
Two passes per iteration (one per player as traverser).

Action encoding  (c/b, meaning is context-dependent):
  history ''   P1: c=check,  b=bet
  history 'c'  P2: c=check,  b=bet
  history 'b'  P2: c=call,   b=fold
  history 'cb' P1: c=call,   b=fold

Terminals (P1 payoff):
  'bc'  -> showdown  ±2   (P1 bet, P2 called)
  'bb'  -> P1 wins   +1   (P1 bet, P2 folded)
  'cc'  -> showdown  ±1   (both checked)
  'cbc' -> showdown  ±2   (check-bet-call)
  'cbb' -> P2 wins   -1   (check-bet-fold)
"""

import random
from collections import defaultdict

CARDS = ['J', 'Q', 'K']

VALID_ACTIONS = {
    '':   ['c', 'b'],
    'c':  ['c', 'b'],
    'b':  ['c', 'b'],
    'cb': ['c', 'b'],
}

TERMINALS = {'bc', 'bb', 'cc', 'cbc', 'cbb'}

def is_terminal(h):
    return h in TERMINALS

def payoff_p0(h, cards):
    p0_wins = CARDS.index(cards[0]) > CARDS.index(cards[1])
    if h == 'bb':  return  1
    if h == 'cbb': return -1
    pot = {'bc': 2, 'cc': 1, 'cbc': 2}[h]
    return pot if p0_wins else -pot

def current_player(h):
    return len(h) % 2   # 0 = P0, 1 = P1


# ── Info-set node ──────────────────────────────────────────────────────────────

class Node:
    def __init__(self):
        self.regret_sum   = defaultdict(float)
        self.strategy_sum = defaultdict(float)

    def get_strategy(self, actions):
        """Regret-matching (no reach weighting needed for external sampling)."""
        pos = sum(max(self.regret_sum[a], 0.0) for a in actions)
        if pos > 0:
            return {a: max(self.regret_sum[a], 0.0) / pos for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def accumulate_strategy(self, strat, actions):
        """Update the two-player external-sampling average at opponent nodes."""
        for a in actions:
            self.strategy_sum[a] += strat[a]

    def avg_strategy(self, actions):
        total = sum(self.strategy_sum[a] for a in actions)
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}


nodes = {}

# ── External sampling MCCFR ────────────────────────────────────────────────────

def mccfr(cards, h, traverser):
    """
    Returns the estimated utility for `traverser` from history h.

    Traverser  -> explore all actions, compute regrets
    Opponent   -> SAMPLE one action (the key Monte Carlo step)
    """
    if is_terminal(h):
        v = payoff_p0(h, cards)
        return v if traverser == 0 else -v

    p       = current_player(h)
    key     = cards[p] + ':' + h
    actions = VALID_ACTIONS[h]

    if key not in nodes:
        nodes[key] = Node()
    node  = nodes[key]
    strat = node.get_strategy(actions)

    if p == traverser:
        # ── Traversing player: explore every action ──────────────────────────

        vals = {a: mccfr(cards, h + a, traverser) for a in actions}
        v    = sum(strat[a] * vals[a] for a in actions)

        # Instantaneous counterfactual regret (no reach prob needed here)
        for a in actions:
            node.regret_sum[a] += vals[a] - v

        return v

    else:
        # ── Opponent: SAMPLE a single action ─────────────────────────────────
        node.accumulate_strategy(strat, actions)
        probs = [strat[a] for a in actions]
        sampled_a = random.choices(actions, weights=probs)[0]
        return mccfr(cards, h + sampled_a, traverser)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000):
    """Two traversals per iteration (alternate which player is traverser)."""
    p0_total = 0.0
    for _ in range(iters):
        # Sample chance node: deal two cards
        deck = CARDS[:]
        random.shuffle(deck)
        cards = (deck[0], deck[1])

        # Traverse as P0, then as P1
        v0 = mccfr(cards, '', traverser=0)
        v1 = mccfr(cards, '', traverser=1)
        p0_total += v0   # v0 is already P0's payoff

    print(f"\nTraining complete ({iters:,} iterations, {2*iters:,} traversals)")
    print(f"  Avg P0 value (from P0 traversals) : {p0_total/iters:+.5f}")
    print(f"  Nash equilibrium value            : {-1/18:+.5f}\n")


# ── Display ────────────────────────────────────────────────────────────────────

ACTION_LABEL = {
    '':   ('Check', 'Bet'),
    'c':  ('Check', 'Bet'),
    'b':  ('Call',  'Fold'),
    'cb': ('Call',  'Fold'),
}

def display():
    print("=" * 56)
    print(f"{'Info Set':<10} {'c-action':>10} {'p(c)':>8}  {'b-action':>10} {'p(b)':>8}")
    print("=" * 56)
    for key in sorted(nodes):
        h       = key.split(':')[1]
        actions = VALID_ACTIONS[h]
        avg     = nodes[key].avg_strategy(actions)
        lab     = ACTION_LABEL[h]
        pc, pb  = avg['c'], avg['b']
        print(f"{key:<10} {lab[0]:>10} {pc:>8.3f}  {lab[1]:>10} {pb:>8.3f}")
    print("=" * 56)
    print("""
Nash equilibrium for Kuhn Poker:
  J:''   bluff-bet with prob alpha in [0, 1/3]
  J:cb   always fold
  Q:''   always check
  Q:b    call with prob ~1/3
  Q:cb   call with prob alpha + 1/3
  K:''   bet with prob 3 * alpha
  K:b    always call
  K:c    always bet (after checking)
  K:cb   always call
  Game value for P0 = -1/18 ≈ -0.0556
""")


if __name__ == '__main__':
    random.seed() #Initialize with a certain seed
    train(100_000)
    display()
