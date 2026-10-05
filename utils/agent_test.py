import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pf_mccfr import Node
from utils.logger import Logger
from datetime import datetime
import utils.play_hand as play_hand
from tqdm import tqdm
import pickle
import os
import shutil

PATH = _ROOT / 'checkpoints' / 'v1.pkl'
PATH_2 = r''

def safe_name(s):
    return str(s).replace("/", "_").replace("\\", "_").replace(":", "_")

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs('logs', exist_ok=True)
os.makedirs('debug_logs', exist_ok=True)

logger = Logger(output_path=f"logs/{timestamp}.txt")
debug_logger = Logger(output_path=f"debug_logs/{timestamp}.txt")
game_logger = Logger(output_path="game_log.txt")
game_logger.clear_logs()

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

# with open(PATH_2, "rb") as f:
#     nodes_2 = pickle.load(f)

do_load = str(input("Would you like to load the nodes into a folder? ('yes' or 'no')\n"))

if do_load.lower() == 'yes':
    # optional: fully clear previous output
    if os.path.exists('loaded_nodes'):
        shutil.rmtree('loaded_nodes')
    os.makedirs('loaded_nodes', exist_ok=True)

    # create one file per hand label
    seen_hands = set()
    for bucket, node in nodes.items():
        hand = safe_name(bucket[0])
        if hand not in seen_hands:
            with open(f'loaded_nodes/{hand}.txt', 'w', encoding='utf-8') as f:
                pass
            seen_hands.add(hand)

    for bucket, node in nodes.items():
        hand = safe_name(bucket[0])
        with open(f'loaded_nodes/{hand}.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------\n')
            f.write(f"{bucket} - Loaded {node.times_visited} times\n")

            node_sum = sum(node.strategy_sum.values())
            if node_sum > 0:
                for k, v in node.strategy_sum.items():
                    f.write(f"{k}: {round(v / node_sum * 100, 1)}%\n")
            else:
                f.write("No accumulated strategy yet (node_sum = 0)\n")

            for k, v in node.regret_sum.items():
                f.write(f"{k} regret: {round(v)}\n")

game_sum = 0
for _ in range(10):
    reward = play_hand.full_agent_vs_player(nodes, 1, game_logger)
    game_sum += reward
print("Average reward for agent:", game_sum / 10)

# iters = 10_000

# game_sum = 0
# for _ in tqdm(range(iters)):
#     # reward = play_hand.agent_vs_random(nodes, 1, game_logger)
#     reward = play_hand.agent_vs_agent(nodes, nodes_2, 1, game_logger)
#     # reward = play_hand.full_agent_vs_random(nodes, 1, game_logger)
#     game_sum += reward

# print("Average reward for agent:", game_sum / iters)