import random
from collections import Counter, OrderedDict
from pokerkit import Card, State, StandardHighHand, Rank
from treys import Card as TreysCard, Evaluator

# ── Equity (EHS) and Potential ──────────────────────────────────────────────────────────────

_evaluator = Evaluator()
_ehs_cache = OrderedDict()
_pot_cache = OrderedDict()
_cache_limit = 10_000
_DECK = tuple(TreysCard.new(r + suit) for r in "23456789TJQKA" for suit in "shdc")


def configure_caches(max_entries=10_000):
    global _cache_limit
    if max_entries < 0:
        raise ValueError("cache size must be nonnegative")
    _cache_limit = max_entries
    # Avoid retaining a previous training run's cached samples.
    _ehs_cache.clear()
    _pot_cache.clear()


def _store(cache, key, value):
    if _cache_limit:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > _cache_limit:
            cache.popitem(last=False)


def _equity_inputs(state, n_samples):
    if n_samples < 1:
        raise ValueError("sample count must be positive")
    hero = tuple(sorted(_to_treys(repr(c)) for c in state.hole_cards[state.actor_index] if c is not None))
    board = tuple(sorted(_to_treys(repr(c[0])) for c in state.board_cards))
    # Unknown opponent cards and burns remain possible: do not leak the deal.
    known = set(hero + board)
    deck = tuple(c for c in _DECK if c not in known)
    return hero, board, deck


def _to_treys(card_str: str):
    """Convert pokerkit repr like 'Ah' to treys int. e.g. 'Ah' -> treys card int"""
    return TreysCard.new(card_str[0] + card_str[1].lower())

def compute_ehs(state: State, n_samples: int = 100) -> float:
    hero_cards, board_cards, deck = _equity_inputs(state, n_samples)

    # Cache in case of repeat board
    key = (hero_cards, board_cards, n_samples)
    if key in _ehs_cache:
        _ehs_cache.move_to_end(key)
        return _ehs_cache[key]

    cards_to_deal = 5 - len(board_cards)
    wins = ties = 0

    for _ in range(n_samples):
        sample     = random.sample(deck, 2 + cards_to_deal)
        vil_cards  = tuple(sample[:2])
        runout     = board_cards + tuple(sample[2:])

        hero_score = _evaluator.evaluate(runout, hero_cards)
        vil_score  = _evaluator.evaluate(runout, vil_cards)

        if hero_score < vil_score:    # lower = better in treys
            wins += 1
        elif hero_score == vil_score:
            ties += 1

    result = (wins + 0.5 * ties) / n_samples

    _store(_ehs_cache, key, result)
    return result


def compute_potential(state: State, n_samples: int = 100) -> tuple[float, float]:
    hero_cards, board_cards, deck = _equity_inputs(state, n_samples)

    # Cache in case of repeat board
    key = (hero_cards, board_cards, n_samples)
    if key in _pot_cache:
        _pot_cache.move_to_end(key)
        return _pot_cache[key]

    cards_to_deal = 5 - len(board_cards)

    ahead_now_behind_later = 0
    behind_now_ahead_later = 0
    ahead_now_total  = 0
    behind_now_total = 0

    hero_now = _evaluator.evaluate(board_cards, hero_cards)
    for _ in range(n_samples):
        sample    = random.sample(deck, 2 + cards_to_deal)
        vil_cards = tuple(sample[:2])
        runout    = board_cards + tuple(sample[2:])

        vil_now  = _evaluator.evaluate(board_cards, vil_cards)

        hero_final = _evaluator.evaluate(runout, hero_cards)
        vil_final  = _evaluator.evaluate(runout, vil_cards)

        currently_ahead  = hero_now < vil_now   # lower = better
        currently_behind = hero_now > vil_now

        if currently_ahead:
            ahead_now_total += 1
            if hero_final > vil_final:          # was ahead, now behind
                ahead_now_behind_later += 1

        if currently_behind:
            behind_now_total += 1
            if hero_final < vil_final:          # was behind, now ahead
                behind_now_ahead_later += 1

    ppot = behind_now_ahead_later / behind_now_total if behind_now_total > 0 else 0.0
    npot = ahead_now_behind_later / ahead_now_total  if ahead_now_total  > 0 else 0.0
    result = ppot, npot
    
    _store(_pot_cache, key, result)
    return result


def compute_ehs2(state: State, n_samples=1000) -> float: 
    """Combined EHS + potential metric. Better single feature than EHS alone."""
    ehs = compute_ehs(state, n_samples)
    ppot, npot = compute_potential(state, n_samples)
    return ehs + (1 - ehs) * ppot - ehs * npot


# ── Board texture ──────────────────────────────────────────────────────────────

def board_flush_texture(state: State) -> str:
    """
    Returns 'monotone', 'two_tone', or 'rainbow'.
    Valid for flop (3 cards) and turn (4 cards).
    On river this is less meaningful — use draw_completed instead.
    """
    suit_counts = Counter(c[0].suit for c in state.board_cards)
    max_suit = max(suit_counts.values())
    if max_suit >= 3:
        return 'monotone'
    if max_suit == 2:
        return 'two_tone'
    return 'rainbow'


def board_is_paired(state: State) -> bool:
    """True if any rank appears more than once on the board."""
    ranks = [c[0].rank for c in state.board_cards]
    return len(ranks) != len(set(ranks))


# ── Draw completed (turn and river) ───────────────────────────────────────────

def flush_draw_completed(prev_board: list[Card], new_board: list[Card]) -> bool:
    """
    Did a flush become possible (or complete) with the new card?
    prev_board = board before this street's card
    new_board = board including this street's card
    """
    def max_suited(b: list[Card]):
        if not b:
            return 0
        return max(Counter(c.suit for c in b).values())
    return max_suited(new_board) >= 3 and max_suited(prev_board) < 3


# Ordered once at module level — deuce=0, trey=1, ..., ace=12
_RANK_ORDER = [Rank(r) for r in "23456789TJQKA"]

def _rank_index(r: Rank) -> int:
    return _RANK_ORDER.index(r)

def _max_straight_draw(board: list[Card]) -> int:
    """Return the largest number of board ranks that fit in any 5-rank window."""
    if not board:
        return 0

    indices = {_rank_index(c.rank) for c in board}
    best = 0

    # Windows A-high through T-high (low card index 0..8)
    for low in range(9):
        hits = sum(1 for r in range(low, low + 5) if r in indices)
        best = max(best, hits)

    # Wheel (A-2-3-4-5): ace index=12 treated as -1
    if 12 in indices:
        wheel = sum(1 for r in range(0, 4) if r in indices) + 1  # +1 for the ace
        best = max(best, wheel)

    return best

def straight_draw_completed(prev_board: list[Card], new_board: list[Card]) -> bool:
    """
    Did a straight draw become possible (3+ ranks fitting a 5-rank window)
    with the addition of the new card(s)?
    """
    return _max_straight_draw(new_board) >= 3 and _max_straight_draw(prev_board) < 3


# ── Bucketing functions ────────────────────────────────────────────────────────
# These convert raw floats to discrete bucket indices.

def equity_bucket(ehs: float, n_bins: int = 8) -> int:
    """0 to n_bins-1"""
    return min(int(ehs * n_bins), n_bins - 1)

def potential_bucket(ppot: float, n_bins: int = 4) -> int:
    return min(int(ppot * n_bins), n_bins - 1)


# ── Street-level bucket builders ──────────────────────────────────────────────

def preflop_card_bucket(
    state: State
) -> str:
    actor = state.actor_index
    hole_cards = str(state.hole_cards[actor]).strip('[]').split(', ')

    # ---------- card parsing ----------
    def rank_value(card):
        r = card[0]
        mapping = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
            "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12,
            "K": 13, "A": 14
        }
        return mapping[r]

    def suit(card):
        return card[1]

    r1 = rank_value(hole_cards[0])
    r2 = rank_value(hole_cards[1])
    s1 = suit(hole_cards[0])
    s2 = suit(hole_cards[1])

    high = max(r1, r2)
    low = min(r1, r2)
    suited = (s1 == s2)
    pair = (r1 == r2)
    gap = high - low

    # ---------- hand bucket ----------
    if pair:
        if high >= 11:  # JJ+
            hand_bucket = "premium_pairs"
        elif 8 <= high <= 10:  # 88-TT
            hand_bucket = "medium_pairs"
        else:  # 22-77
            hand_bucket = "small_pairs"

    elif suited and high == 14 and 10 <= low <= 13:
        # ATs, AJs, AQs, AKs
        hand_bucket = "premium_suited_aces"

    elif high == 14 and suited and 2 <= low <= 9:
        # A2s-A9s
        hand_bucket = "weak_suited_aces"

    elif high == 14 and not suited and low >= 12:
        # AKo, AQo
        hand_bucket = "premium_offsuit_aces"

    elif suited and high >= 11 and low >= 10:
        # KQs, KJs, QJs, JTs, AQs, AKs etc.
        # Since suited aces already handled above, this is mostly KQs-JTs type hands
        hand_bucket = "premium_suited_broadways"

    elif not suited and high >= 11 and low >= 10:
        # KQo / KJo / QJo / JTo / AQo / AJo / etc.
        if (high == 13 and low == 12):
            hand_bucket = "premium_offsuit_broadways"  # KQo
        elif (high == 14 and low >= 11):
            # AKo/AQo handled above, AJo falls here if you want it weaker
            hand_bucket = "weak_offsuit_broadways"
        elif high == 13 and low == 11:
            hand_bucket = "weak_offsuit_broadways"  # KJo
        elif high == 12 and low == 11:
            hand_bucket = "weak_offsuit_broadways"  # QJo
        elif high == 11 and low == 10:
            hand_bucket = "weak_offsuit_broadways"  # JTo
        else:
            hand_bucket = "trash_offsuit_hands"

    elif suited and gap == 1 and high <= 13:
        # suited connectors like 98s, T9s, 76s, KQs technically connector but already caught above
        hand_bucket = "suited_connectors"

    elif suited and gap == 2 and high <= 13:
        # suited one-gappers like 97s, T8s, J9s
        hand_bucket = "suited_gappers"

    else:
        hand_bucket = "trash_offsuit_hands"
    
    return hand_bucket


def flop_card_bucket(
    state: State, # 3 cards
    n_samples: int = 500
) -> tuple:
    assert len(state.board_cards) == 3

    ehs = compute_ehs(state, n_samples)
    ppot, npot = compute_potential(state, n_samples)

    # ehs2_bucket = equity_bucket(compute_ehs2(state, 200), 16)

    ehs_bucket = equity_bucket(ehs)
    ppot_bucket = potential_bucket(ppot)
    npot_bucket = potential_bucket(npot)

    flush_texture = board_flush_texture(state)
    board_paired = board_is_paired(state)

    result = (
        ehs_bucket,
        ppot_bucket,
        npot_bucket,
        # ehs2_bucket,
        flush_texture,
        board_paired
    )

    return result


def turn_card_bucket(
    state: State,# first 3 cards (to detect draw completion)
    n_samples: int = 500
) -> tuple:
    assert len(state.board_cards) == 4
    ehs = compute_ehs(state, n_samples)
    ppot, npot = compute_potential(state, n_samples)
    board = [card[0] for card in state.board_cards]
    flop = [state.board_cards[i][0] for i in range(3)]

    ehs_bucket = equity_bucket(ehs)
    ppot_bucket = potential_bucket(ppot)
    npot_bucket = potential_bucket(npot)

    # ehs2_bucket = equity_bucket(compute_ehs2(state, 200), 16)

    flush_texture = board_flush_texture(state)
    board_paired = board_is_paired(state)
    flush_draw_completed_bucket = flush_draw_completed(flop, board)
    straight_draw_completed_bucket = straight_draw_completed(flop, board)
    
    result = (
        ehs_bucket,
        ppot_bucket,
        npot_bucket,
        # ehs2_bucket,
        flush_texture,
        board_paired,
        flush_draw_completed_bucket,
        straight_draw_completed_bucket
    )

    return result


def river_card_bucket(
    state: State,
    n_samples: int = 500
) -> tuple:
    assert len(state.board_cards) == 5

    board = [card[0] for card in state.board_cards]
    turn_board = [state.board_cards[i][0] for i in range(4)]
    ehs = compute_ehs(state, n_samples) # pure equity, no potential
    
    ehs_bucket = equity_bucket(ehs, n_bins=8)
    flush_draw_completed_bucket = flush_draw_completed(turn_board, board)
    straight_draw_completed_bucket = straight_draw_completed(turn_board, board)
    board_paired = board_is_paired(state)

    result = (
        ehs_bucket,
        flush_draw_completed_bucket,
        straight_draw_completed_bucket,
        board_paired
    )

    return result

def exact_preflop_card_bucket(
    state: State
) -> str:
    actor = state.actor_index
    hole_cards = str(state.hole_cards[actor]).strip('[]').split(', ')
    suited = 's' if Card.are_suited(''.join(hole_cards)) else 'o'
    ranks = ''.join(sorted(hole_cards[0][0] + hole_cards[1][0]))
    
    hand_bucket = ranks + suited

    return hand_bucket
