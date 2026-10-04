from pokerkit import Automation, State, NoLimitTexasHoldem, Mode
from utils.bucketer import Bucketer
from utils.logger import Logger
from utils.agent_policy import choose_action, legal_actions, node_for_actions
from full_game_mccfr import get_pf_raise_size, get_halfp_raise_size
import random

# ----------- HELPER FUNCTIONS ---------------------
def random_action(state: State) -> tuple:

    actions = []
    probs = [0.2, 0.4, 0.4]

    if state.bets[state.actor_index] < max(state.bets) and state.can_fold():
        actions.append(("fold", lambda: state.fold()))

    if state.can_check_or_call():
        actions.append(("check/call", lambda: state.check_or_call()))

    amount = 0
    if state.can_complete_bet_or_raise_to():
        amount = max(state.bets) + state.total_pot_amount * 1/2 #Raises half pot by default
        if amount > state.stacks[state.actor_index]:
            all_in_amt = state.stacks[state.actor_index]
            min_bet = state.min_completion_betting_or_raising_to_amount
            if min_bet is None:
                min_bet = 0
            amount = all_in_amt if all_in_amt >= min_bet else None
        if state.can_complete_bet_or_raise_to(amount):
            actions.append(("raise", lambda: state.complete_bet_or_raise_to(amount)))
        else:
            probs = [0.2, 0.8]
    else:
        probs = [0.2, 0.8]

    if not actions:
        raise Exception("No actions avaliable???")

    action_name, action_fn = random.choices(actions, weights=probs)[0]
    action_fn()
    return action_name, amount

# ----------- TEST PLAYING HANDS -------------------
def random_vs_random(logger: Logger) -> State:
    """
    Simulate a hand where the random plays against random (20% fold, 40% call, 40% raise)
    """
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (1, 2),                # blinds
        2,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        match state.street_index:
            case 0: # IF PREFLOP: 
                bucket = bucketer.preflop_bucket(state, pf_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)
            case _: # STOPS AFTER PREFLOP
                while state.actor_index is not None:
                    state.check_or_call()
                break

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        action_name, amount = random_action(state)
            
        if action_name == 'raise':
            logger.log(f"Player {state.actor_index}: raises to {amount}")
        else:
            logger.log(f"Player {state.actor_index}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state.stacks[1] - state.starting_stacks[1]

def agent_vs_player() -> State:
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (1, 2),                # blinds
        2,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    print(tuple(state.pot_amounts))
    print("boom")

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        # Log once whenever a new street begins
        match state.street_index:
            case 0: # IF PREFLOP: 
                bucket = bucketer.preflop_bucket(state, pf_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == 0:
            action_name, amount = random_action(state)
        else:
            while (True):
                try:
                    action = int(input("Action? 0: fold, 1: check/call, 2: raise\n"))
                    match action:
                        case 0:
                            state.fold()
                            action_name = "fold"
                        case 1:
                            state.check_or_call()
                            action_name = "check/call"
                        case 2:
                            amount = int(input("Input raise size: "))
                            if amount >= state.stacks[state.actor_index]:
                                amount = state.stacks[state.actor_index]
                            state.complete_bet_or_raise_to(amount)
                            action_name = "raise"
                    break
                except Exception:
                    print("Error. Try reinputting your action.\n")
        if action_name == 'raise':
            logger.log(f"Player {state.actor_index}: raises to {amount}")
        else:
            logger.log(f"Player {state.actor_index}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state

def agent_vs_random(agent: dict, agent_pos: int, logger: Logger) -> State:
    """
    Simulate a hand where the agent plays against an opponent that plays randomly (20% fold, 40% call, 40% raise)
    \nagent: dict\n
    - The dictionary containing all the buckets and corresponding nodes that the agent plays from.
    \nagent_pos: int\n
    - What position the agent should start in, 0 for BB, 1 for SB.
    """
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (0.5, 1),                # blinds
        1,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        match state.street_index:
            case 0: # IF PREFLOP: 
                bucket = bucketer.exact_preflop_bucket(state, pf_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)
            case _: # STOPS AFTER PREFLOP
                while state.actor_index is not None:
                    state.check_or_call()
                break

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == agent_pos:
            amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
            actions = legal_actions(state, amount)
            node = node_for_actions(agent, bucket, actions)
            action_name = choose_action(state, bucket, node, raise_to=amount)
            if action_name == 'raise':
                state.complete_bet_or_raise_to(amount)
            elif action_name == 'check/call':
                state.check_or_call()
            elif action_name == 'fold':
                state.fold()
            else:
                raise Exception
        else:
            action_name, amount = random_action(state)
            
        if action_name == 'raise':
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'random'}: raises to {amount}")
        else:
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'random'}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state.stacks[agent_pos] - state.starting_stacks[agent_pos]

def full_agent_vs_random(agent: dict, agent_pos: int, logger: Logger) -> State:
    """
    Simulate a hand where the agent plays against an opponent that plays randomly (20% fold, 40% call, 40% raise)
    \nagent: dict\n
    - The dictionary containing all the buckets and corresponding nodes that the agent plays from.
    \nagent_pos: int\n
    - What position the agent should start in, 0 for BB, 1 for SB.
    """
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (0.5, 1),                # blinds
        1,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        match state.street_index:
            case 0: # IF PREFLOP: 
                state.check_or_call()
                # bucket = bucketer.preflop_bucket(state, pf_history)
                # logger.log(state.hole_cards[state.actor_index])
                # logger.log(bucket)
            case 1: 
                bucket = bucketer.flop_bucket(state, flop_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)
            case 2: 
                bucket = bucketer.turn_bucket(state, turn_history, flop_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)
            case 3:
                bucket = bucketer.river_bucket(state, river_history, turn_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == agent_pos:
            amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
            actions = legal_actions(state, amount)
            node = node_for_actions(agent, bucket, actions)
            action_name = choose_action(state, bucket, node, raise_to=amount)
            if action_name == 'raise':
                state.complete_bet_or_raise_to(amount)
            elif action_name == 'check/call':
                state.check_or_call()
            elif action_name == 'fold':
                state.fold()
            else:
                raise Exception
        else:
            action_name, amount = random_action(state)
            
        if action_name == 'raise':
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'random'}: raises to {amount}")
        else:
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'random'}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state.stacks[agent_pos] - state.starting_stacks[agent_pos]

def agent_vs_agent(agent: dict, agent_2: dict, agent_pos: int, logger: Logger) -> State:
    """
    Simulate a hand where the agent plays against an opponent that plays randomly (20% fold, 40% call, 40% raise)
    \nagent: dict\n
    - The dictionary containing all the buckets and corresponding nodes that the agent plays from.
    \nagent_pos: int\n
    - What position the agent should start in, 0 for BB, 1 for SB.
    """
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (0.5, 1),                # blinds
        1,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        match state.street_index:
            case 0: # IF PREFLOP: 
                if state.actor_index == agent_pos:
                    bucket = bucketer.exact_preflop_bucket(state, pf_history)
                else:
                    bucket = bucketer.preflop_bucket(state, pf_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)
            case _: # STOPS AFTER PREFLOP
                while state.actor_index is not None:
                    state.check_or_call()
                break

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == agent_pos:
            amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
            actions = legal_actions(state, amount)
            node = node_for_actions(agent, bucket, actions)
            action_name = choose_action(state, bucket, node, raise_to=amount)
            if action_name == 'raise':
                state.complete_bet_or_raise_to(amount)
            elif action_name == 'check/call':
                state.check_or_call()
            elif action_name == 'fold':
                state.fold()
            else:
                raise Exception
        else:
            amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
            actions = legal_actions(state, amount)
            node = node_for_actions(agent_2, bucket, actions)
            action_name = choose_action(state, bucket, node, raise_to=amount)
            if action_name == 'raise':
                state.complete_bet_or_raise_to(amount)
            elif action_name == 'check/call':
                state.check_or_call()
            elif action_name == 'fold':
                state.fold()
            else:
                raise Exception
            
        if action_name == 'raise':
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'agent 2'}: raises to {amount}")
        else:
            logger.log(f"Player {'agent' if state.actor_index == agent_pos else 'agent 2'}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state.stacks[agent_pos] - state.starting_stacks[agent_pos]

def full_agent_vs_player(agent: dict, agent_pos: int, logger: Logger) -> State:
    """
    Simulate a hand where the agent plays against an opponent that plays randomly (20% fold, 40% call, 40% raise)
    \nagent: dict\n
    - The dictionary containing all the buckets and corresponding nodes that the agent plays from.
    \nagent_pos: int\n
    - What position the agent should start in, 0 for BB, 1 for SB.
    """
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (0.5, 1),                # blinds
        1,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        match state.street_index:
            case 0: # IF PREFLOP: 
                bucket = bucketer.exact_preflop_bucket(state, pf_history)
                # logger.log(state.hole_cards[state.actor_index])
                # logger.log(bucket)
            case 1: 
                bucket = bucketer.flop_bucket(state, flop_history)
                # logger.log(state.hole_cards[state.actor_index])
                # logger.log(bucket)
            case 2: 
                bucket = bucketer.turn_bucket(state, turn_history, flop_history)
                # logger.log(state.hole_cards[state.actor_index])
                # logger.log(bucket)
            case 3:
                bucket = bucketer.river_bucket(state, river_history, turn_history)
                # logger.log(state.hole_cards[state.actor_index])
                # logger.log(bucket)

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == agent_pos:
            amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
            actions = legal_actions(state, amount)
            node = node_for_actions(agent, bucket, actions)
            action_name = choose_action(state, bucket, node, raise_to=amount)
            if action_name == 'raise':
                state.complete_bet_or_raise_to(amount)
            elif action_name == 'check/call':
                state.check_or_call()
            elif action_name == 'fold':
                state.fold()
            else:
                raise Exception
        else:
            logger.log(f"Your hole cards: {state.hole_cards[state.actor_index]}")
            while (True):
                try:
                    action = int(input("Action? 0: fold, 1: check/call, 2: raise\n"))
                    match action:
                        case 0:
                            state.fold()
                            action_name = "fold"
                        case 1:
                            state.check_or_call()
                            action_name = "check/call"
                        case 2:
                            amount = int(input("Input raise size: "))
                            if amount >= state.stacks[state.actor_index]:
                                amount = state.stacks[state.actor_index]
                            state.complete_bet_or_raise_to(amount)
                            action_name = "raise"
                    break
                except Exception:
                    print("Error. Try reinputting your action.\n")
            
        if action_name == 'raise':  # if agent acted, the index is already off of them and on the player
            logger.log(f"Player{' agent' if state.actor_index != agent_pos else ''}: raises to {amount}")
        else:
            logger.log(f"Player{' agent' if state.actor_index != agent_pos else ''}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state.stacks[agent_pos] - state.starting_stacks[agent_pos]

if __name__ == "__main__":
    logger = Logger(output_path="game_log.txt")
    logger.clear_logs() # clears logs so new hand can be logged
