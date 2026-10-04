from pokerkit import State
from utils.card_bucketer import exact_preflop_card_bucket, preflop_card_bucket, flop_card_bucket, turn_card_bucket, river_card_bucket

class Bucketer:
    def __init__(self, samples=100) -> None:
        self.samples = samples

    def preflop_bucket(self, state: State, history: list) -> tuple:

        actor = state.actor_index
        
        hand_bucket = preflop_card_bucket(state)

        # ---------- position bucket ----------
        # PokerKit heads-up seats: actor 0 = BB, actor 1 = SB.
        position_bucket = "BB" if actor == 0 else "SB"

        # ---------- effective stack bucket ----------
        # Assumes state.stacks exists and blinds are normalized to bb units,
        # or at least consistent with 100bb+ interpretation.
        eff_stack = min(state.stacks[0], state.stacks[1])

        if eff_stack < 20:
            stack_bucket = "short"
        elif eff_stack < 50:
            stack_bucket = "medium"
        elif eff_stack < 100:
            stack_bucket = "deep"
        else:
            stack_bucket = "very_deep"

        # ---------- betting history bucket ----------
        if not history:
            history_bucket = "root"
        elif len(history) == 1:
            first = history[0]
            if first == 'check/call':
                history_bucket = "limped"
            elif first == 'raise':
                history_bucket = "vs_open"
            else:
                history_bucket = "root"
        elif len(history) == 2:
            if (history[0] == 'raise') and (history[1] == 'raise'):
                history_bucket = "vs_3bet"
            else:
                history_bucket = "vs_open"
        elif len(history) == 3:
            if (history[0] == 'raise') and (history[1] == 'raise') and (history[2] == 'raise'):
                history_bucket = "vs_4bet"
            else:
                history_bucket = "vs_3bet"
        else:
            history_bucket = "vs_4bet"

        # ---------- betting size bucket ----------
        # Assumes there is some amount currently being faced.
        # Best case: state.to_call and blind size available.
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        bb = max(state.blinds_or_straddles)

        # Size in big blinds
        size_bb = to_call / bb if bb > 0 else to_call

        if history_bucket == "limped":
            size_bucket = "Limp"
        elif size_bb <= 2.0 and to_call > 0:
            size_bucket = "~2.0bb raise"
        elif 2.0 < size_bb <= 2.75:
            size_bucket = "~2.75bb raise"
        elif 2.75 < size_bb < 6.0:
            size_bucket = "~6.0bb raise"
        elif 6.0 <= size_bb < 10.0:
            size_bucket = "~10.0bb raise"
        elif 10.0 <= size_bb < 25.0:
            size_bucket = "~25.0bb raise"
        elif size_bb >= 25.0:
            size_bucket = "Jam (>25.00bb) raise"
        else:
            size_bucket = "Limp"

        return (
            hand_bucket,
            position_bucket,
            stack_bucket,
            history_bucket,
            size_bucket,
        )
        
    def flop_bucket(self, state: State, history: list) -> tuple:

        actor = state.actor_index
        
        hand_bucket = flop_card_bucket(state, n_samples=self.samples)

        position_bucket = "BB" if actor == 0 else "SB"
        
        raise_count = history.count("raise")
        raises_this_street_bucket = raise_count if raise_count < 3 else 3  # capped at 3
        
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        pot_size = state.total_pot_amount

        ratio = to_call / pot_size
        if ratio < 0.4:     size_bucket = 'small'    # ~1/3 pot
        elif ratio < 0.75:  size_bucket = 'medium'   # ~1/2-2/3 pot
        elif ratio < 1.1:   size_bucket = 'large'    # ~pot
        else: size_bucket = 'overbet'
        
        spr = state.stacks[actor] / state.total_pot_amount
        if spr > 10:  spr_bucket = 'deep'
        elif spr > 4:   spr_bucket = 'mid_deep'  
        elif spr > 1.5: spr_bucket = 'mid'
        else: spr_bucket = 'short'        # near-shove territory

        return (
            hand_bucket,
            position_bucket,
            raises_this_street_bucket,
            size_bucket,
            spr_bucket,
        )

    def turn_bucket(self, state: State, turn_history: list, flop_history: list) -> tuple:

        actor = state.actor_index
        
        hand_bucket = turn_card_bucket(state, n_samples=self.samples)

        position_bucket = "BB" if actor == 0 else "SB"
        
        # raise_count = turn_history.count("raise")
        # raises_this_street_bucket = raise_count if raise_count < 3 else 3  # capped at 3
        if not turn_history:
            history_bucket = "root"
        elif len(turn_history) == 1:
            first = turn_history[0]
            if first == 'check/call':
                history_bucket = "limped"
            elif first == 'raise':
                history_bucket = "vs_open"
            else:
                history_bucket = "root"
        elif len(turn_history) == 2:
            if (turn_history[0] == 'raise') and (turn_history[1] == 'raise'):
                history_bucket = "vs_3bet"
            else:
                history_bucket = "vs_open"
        elif len(turn_history) == 3:
            if (turn_history[0] == 'raise') and (turn_history[1] == 'raise') and (turn_history[2] == 'raise'):
                history_bucket = "vs_4bet"
            else:
                history_bucket = "vs_3bet"
        else:
            history_bucket = "vs_4bet"
        
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        pot_size = state.total_pot_amount

        ratio = to_call / pot_size
        if ratio < 0.4:     size_bucket = 'small'    # ~1/3 pot
        elif ratio < 0.75:  size_bucket = 'medium'   # ~1/2-2/3 pot
        elif ratio < 1.1:   size_bucket = 'large'    # ~pot
        else: size_bucket = 'overbet'
        
        spr = state.stacks[actor] / state.total_pot_amount
        if spr > 10:  spr_bucket = 'deep'
        elif spr > 4:   spr_bucket = 'mid_deep'  
        elif spr > 1.5: spr_bucket = 'mid'
        else: spr_bucket = 'short'        # near-shove territory

        prev_street_raise_bucket = False
        if actor == 0:
            for idx, action in enumerate(flop_history):
                if idx % 2 == 0 and action == 'raise':
                    prev_street_raise_bucket = True
        else:
            for idx, action in enumerate(flop_history):
                if idx % 2 == 1 and action == 'raise':
                    prev_street_raise_bucket = True

        return (
            hand_bucket,
            position_bucket,
            history_bucket,
            size_bucket,
            spr_bucket,
            prev_street_raise_bucket
        )

    def river_bucket(self, state: State, river_history: list, turn_history: list) -> tuple:

        actor = state.actor_index
        
        hand_bucket = river_card_bucket(state, n_samples=self.samples)

        position_bucket = "BB" if actor == 0 else "SB"
        
        # raise_count = river_history.count("raise")
        # raises_this_street_bucket = raise_count if raise_count < 3 else 3  # capped at 3
        if not river_history:
            history_bucket = "root"
        elif len(river_history) == 1:
            first = river_history[0]
            if first == 'check/call':
                history_bucket = "limped"
            elif first == 'raise':
                history_bucket = "vs_open"
            else:
                history_bucket = "root"
        elif len(river_history) == 2:
            if (river_history[0] == 'raise') and (river_history[1] == 'raise'):
                history_bucket = "vs_3bet"
            else:
                history_bucket = "vs_open"
        elif len(river_history) == 3:
            if (river_history[0] == 'raise') and (river_history[1] == 'raise') and (river_history[2] == 'raise'):
                history_bucket = "vs_4bet"
            else:
                history_bucket = "vs_3bet"
        else:
            history_bucket = "vs_4bet"
        
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        pot_size = state.total_pot_amount

        ratio = to_call / pot_size
        if ratio < 0.4:     size_bucket = 'small'    # ~1/3 pot
        elif ratio < 0.75:  size_bucket = 'medium'   # ~1/2-2/3 pot
        elif ratio < 1.1:   size_bucket = 'large'    # ~pot
        else: size_bucket = 'overbet'
        
        spr = state.stacks[actor] / state.total_pot_amount
        if spr > 10:  spr_bucket = 'deep'
        elif spr > 4:   spr_bucket = 'mid_deep'  
        elif spr > 1.5: spr_bucket = 'mid'
        else: spr_bucket = 'short'        # near-shove territory

        prev_street_raise_bucket = False
        if actor == 0:
            for idx, action in enumerate(turn_history):
                if idx % 2 == 0 and action == 'raise':
                    prev_street_raise_bucket = True
        else:
            for idx, action in enumerate(turn_history):
                if idx % 2 == 1 and action == 'raise':
                    prev_street_raise_bucket = True

        return (
            hand_bucket,
            position_bucket,
            history_bucket,
            size_bucket,
            spr_bucket,
            prev_street_raise_bucket
        )

    def exact_preflop_bucket(self, state: State, history: list) -> tuple:

        actor = state.actor_index
        
        hand_bucket = exact_preflop_card_bucket(state)

        # ---------- position bucket ----------
        # PokerKit heads-up seats: actor 0 = BB, actor 1 = SB.
        position_bucket = "BB" if actor == 0 else "SB"

        # ---------- effective stack bucket ----------
        # Assumes state.stacks exists and blinds are normalized to bb units,
        # or at least consistent with 100bb+ interpretation.
        eff_stack = min(state.stacks[0], state.stacks[1])

        if eff_stack < 20:
            stack_bucket = "short"
        elif eff_stack < 50:
            stack_bucket = "medium"
        else:
            stack_bucket = "deep"

        # ---------- betting history bucket ----------
        if not history:
            history_bucket = "root"
        elif len(history) == 1:
            first = history[0]
            if first == 'check/call':
                history_bucket = "limped"
            elif first == 'raise':
                history_bucket = "vs_open"
            else:
                history_bucket = "root"
        elif len(history) == 2:
            if (history[0] == 'raise') and (history[1] == 'raise'):
                history_bucket = "vs_3bet"
            else:
                history_bucket = "vs_open"
        elif len(history) == 3:
            if (history[0] == 'raise') and (history[1] == 'raise') and (history[2] == 'raise'):
                history_bucket = "vs_4bet"
            else:
                history_bucket = "vs_3bet"
        else:
            history_bucket = "vs_4bet"

        # ---------- betting size bucket ----------
        # Assumes there is some amount currently being faced.
        # Best case: state.to_call and blind size available.
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        bb = max(state.blinds_or_straddles)

        # Size in big blinds
        size_bb = to_call / bb if bb > 0 else to_call

        if history_bucket == "limped":
            size_bucket = "Limp"
        elif size_bb <= 2.0 and to_call > 0:
            size_bucket = "~2.0bb raise"
        elif 2.0 < size_bb <= 2.75:
            size_bucket = "~2.75bb raise"
        elif 2.75 < size_bb < 6.0:
            size_bucket = "~6.0bb raise"
        elif 6.0 <= size_bb < 10.0:
            size_bucket = "~10.0bb raise"
        elif 10.0 <= size_bb < 25.0:
            size_bucket = "~25.0bb raise"
        elif size_bb >= 25.0:
            size_bucket = "Jam (>25.00bb) raise"
        else:
            size_bucket = "Limp"

        return (
            hand_bucket,
            position_bucket,
            stack_bucket,
            history_bucket,
            size_bucket,
        )

"""
---------------- Preflop
hand bucket
    premium pairs (A-J)
    medium pairs (10-8)
    small pairs (7-2)
    premium suited broadways (KQs - J10s)
    premium offsuit broadways (KQo)
    premium suited aces (AKs-A10s)
    weak suited aces (A9s-A2s)
    premium offsuit aces (AKo-AQo)
    suited connectors 
    suited gappers 
    weak offsuit broadways (KJo-J10o)
    trash offsuit hands (everything else)

position bucket
    SB
    BB
    BTN and other pos (for multiway, probably not using for prototype)

effective stack size (constrain to 100bb+ scenarios)
    0-20bb (short)
    20-50bb (medium)
    50-100bb (deep)
    100bb+ (very deep)

betting history (action / facing 3 bet / facing open)
    root
    open
    vs_open
    vs_3bet
    vs_4bet
    limped
    limp_raise

betting size
    limp
    min open
    2.5x open
    large open
    small 3-bet
    large 3-bet
    jam

----------------- Flop

HS
EHS
EHS²
maybe one or two draw flags

Turn

Same as flop, but maybe lighter-weight if speed is an issue

River

Usually HS alone is much more informative, since there are no future streets left
and EHS basically collapses toward current hand strength.
"""
