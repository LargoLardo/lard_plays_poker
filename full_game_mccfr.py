import random
import pickle
from pokerkit import Automation, Mode, NoLimitTexasHoldem, State
from utils.training import TrainingNode, train_loop, training_main
from utils.bucketer import Bucketer
from utils.agent_policy import legal_actions, bucket_with_actions



def is_terminal(state: State) -> bool:
    return state.actor_index is None

def payoff_p0(state: State):
    return state.stacks[0] - state.starting_stacks[0]

# ── Info-set node ──────────────────────────────────────────────────────────────

class Node(TrainingNode):
    __slots__ = ()

# ── External sampling MCCFR ────────────────────────────────────────────────────

def mccfr(state: State, traverser: int, histories: list[list[str]], base_nodes: dict, delta_nodes: dict, bucketer: Bucketer):
    if is_terminal(state):
        payoff = payoff_p0(state)
        return payoff if traverser == 0 else -payoff
    
    match state.street_index:
        case 0: #pf
            bucket = bucketer.exact_preflop_bucket(state, histories[0])
        case 1: #flop
            bucket = bucketer.flop_bucket(state, histories[1])
        case 2: #turn
            bucket = bucketer.turn_bucket(state, histories[2], histories[1])
        case 3: #river
            bucket = bucketer.river_bucket(state, histories[3], histories[2])

    cur_actor = state.actor_index
    amount = get_pf_raise_size(state, bucket) if state.street_index == 0 else get_halfp_raise_size(state, bucket)
    actions = legal_actions(state, amount)
    bucket = bucket_with_actions(bucket, actions)

    base_node = base_nodes.get(bucket)

    if bucket not in delta_nodes:
        delta_nodes[bucket] = Node()
    delta_node = delta_nodes.get(bucket)

    def current_regret(action):
        base_val = base_node.regret_sum.get(action, 0.0) if base_node else 0.0
        delta_val = delta_node.regret_sum.get(action, 0.0) if delta_node else 0.0
        return base_val + delta_val

    def get_current_strategy(actions):
        pos = sum(max(current_regret(a), 0.0) for a in actions)
        if pos > 0:
            return {a: max(current_regret(a), 0.0) / pos for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    # Freeze regret matching before exploring children of this information set.
    strat = get_current_strategy(actions)

    def next_position(action):
        next_state = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        next_history = [h.copy() for h in histories]
        if action == 'fold':
            next_state.fold()
        elif action == 'check/call':
            next_state.check_or_call()
        else:
            next_state.complete_bet_or_raise_to(amount)
        next_history[state.street_index].append(action)
        return next_state, next_history

    if cur_actor == traverser:
        utils = {}
        for action in actions:
            next_state, next_history = next_position(action)
            utils[action] = mccfr(next_state, traverser, next_history, base_nodes, delta_nodes, bucketer)
        node_util = sum(strat[a] * utils[a] for a in actions)
        for action in actions:
            delta_node.regret_sum[action] += utils[action] - node_util
        return node_util

    # Two-player external sampling: opponent reach is supplied by sampling.
    delta_node.times_visited += 1
    for action in actions:
        delta_node.strategy_sum[action] += strat[action]
    action = random.choices(actions, weights=[strat[a] for a in actions])[0]
    next_state, next_history = next_position(action)
    return mccfr(next_state, traverser, next_history, base_nodes, delta_nodes, bucketer)

# -- Helper functions -------------------------------

def clone_nodes(nodes: dict) -> dict:
    return {k: v.clone() for k, v in nodes.items()}

def get_halfp_raise_size(state: State, bucket: tuple) -> float:
    amount = max(state.bets) + state.total_pot_amount * 1/2 #Raises half pot by default
    amount = round(amount)
    maximum = state.stacks[state.actor_index] + state.bets[state.actor_index]
    amount = maximum if 'vs_4bet' in bucket else min(amount, maximum)
    return amount

def get_pf_raise_size(state: State, bucket: tuple) -> float:
    amount = max(state.bets) * 3
    amount = round(amount)
    maximum = state.stacks[state.actor_index] + state.bets[state.actor_index]
    amount = maximum if 'vs_4bet' in bucket else min(amount, maximum)
    return amount

def get_rand_raise_size(state: State, bucket: tuple) -> float:
    amount = max(state.bets) + state.total_pot_amount * random.choice((1/3, 1/2, 2/3, 1))
    amount = round(amount)
    maximum = state.stacks[state.actor_index] + state.bets[state.actor_index]
    amount = maximum if 'vs_4bet' in bucket else min(amount, maximum)
    return amount

# -- Multiprocessing / Worker Managers -------------------------------------------------------

def run_chunk(args):
    chunk_size, seed, snapshot, start, samples, cache_size = args
    random.seed(seed)

    from utils.card_bucketer import configure_caches
    configure_caches(cache_size)
    base_nodes = pickle.loads(snapshot)
    delta_nodes = {}
    local_bucketer = Bucketer(samples)

    for count in range(chunk_size):
        state = create_state()
        play_hand(
            state,
            traverser=(start + count) % 2,
            base_nodes=base_nodes,
            delta_nodes=delta_nodes,
            bucketer=local_bucketer,
        )

    return delta_nodes, chunk_size

def merge_nodes(master: dict, delta: dict):
    for key, delta_node in delta.items():
        if key not in master:
            master[key] = Node()

        m = master[key]

        for action, value in delta_node.regret_sum.items():
            m.regret_sum[action] += value

        for action, value in delta_node.strategy_sum.items():
            m.strategy_sum[action] += value

        m.times_visited += delta_node.times_visited

# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000, n_workers=1, merge_every=1000, **options):
    return train_loop(create_state, play_hand, run_chunk, merge_nodes,
                      trainer="full-game", iters=iters, n_workers=n_workers,
                      merge_every=merge_every, **options)

def play_hand(state, traverser, base_nodes, delta_nodes, bucketer):
    histories = list()
    for _ in range(4):
        histories.append(list())
    return mccfr(state, traverser, histories, base_nodes, delta_nodes, bucketer)
    
def create_state() -> State:
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.RUNOUT_COUNT_SELECTION,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING, #commented for now to show all hole cards at showdown
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
    return state


if __name__ == '__main__':
    training_main(train)
