#!/usr/bin/env python3
"""
Compute rewards for BC turn logs (jsonl) using PPO RewardTracker logic.

Usage examples:
  python bc/compute_rewards.py --jsonl bc_data/episode_1_turns.jsonl
  python bc/compute_rewards.py --dir bc_data
  python bc/compute_rewards.py --dir bc_data --winner p1
  python bc/compute_rewards.py --jsonl bc_data/episode_1_turns.jsonl --engine-log engine_output.txt

Winner resolution priority:
  1) --winner {p1,p2,draw}
  2) --engine-log file containing: "Winner (p1 perspective, 1 = p1 2 = p2): X"
  3) Auto from last hp vs enemy_hp in the jsonl
"""

import os
import sys
import argparse
import glob
import json
from typing import List, Dict, Optional, Tuple


def add_rl_path():
    """Ensure my-first-algo-ppo/rl is importable."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rl_root = os.path.join(repo_root, 'my-first-algo-ppo')
    if rl_root not in sys.path:
        sys.path.insert(0, rl_root)


add_rl_path()

try:
    from rl.reward_tracker import RewardTracker
except Exception as e:
    print(f"Error: could not import RewardTracker from my-first-algo-ppo/rl: {e}")
    sys.exit(1)


class MockGameState:
    def __init__(self, my_health: float, enemy_health: float):
        self.my_health = my_health
        self.enemy_health = enemy_health


def parse_engine_winner(engine_log_path: str) -> Optional[str]:
    """Parse engine log file to determine winner ('p1', 'p2', 'draw') or None."""
    try:
        with open(engine_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if "Winner (p1 perspective, 1 = p1 2 = p2): 1" in content:
            return 'p1'
        if "Winner (p1 perspective, 1 = p1 2 = p2): 2" in content:
            return 'p2'
        return 'draw' if "Winner" in content else None
    except Exception:
        return None


def determine_winner(turns: List[Dict], cli_winner: Optional[str], engine_log: Optional[str]) -> str:
    """Resolve winner using CLI, engine log, or HP auto detection."""
    # 1) CLI override
    if cli_winner in {'p1', 'p2', 'draw'}:
        return cli_winner

    # 2) Engine log
    if engine_log:
        parsed = parse_engine_winner(engine_log)
        if parsed in {'p1', 'p2', 'draw'}:
            return parsed

    # 3) Auto from last HP
    if not turns:
        return 'draw'
    last = turns[-1]
    my_hp = float(last.get('hp', 0.0))
    enemy_hp = float(last.get('enemy_hp', 0.0))
    if my_hp > enemy_hp:
        return 'p1'
    if my_hp < enemy_hp:
        return 'p2'
    return 'draw'


def compute_rewards_for_turns(turns: List[Dict], winner: str) -> List[Dict]:
    """Compute rewards in-place for a list of turn dicts and return updated list."""
    if not turns:
        return turns

    tracker = RewardTracker()

    # Initialize tracker with first turn HPs
    first = turns[0]
    tracker.reset(MockGameState(first['hp'], first['enemy_hp']))

    updated: List[Dict] = []
    for i, turn in enumerate(turns):
        turn = dict(turn)  # copy
        # Default
        turn.setdefault('reward', 0.0)
        turn.setdefault('terminal', False)

        if i < len(turns) - 1:
            # Non-terminal: use next state's HP for delta-based reward
            nxt = turns[i + 1]
            next_state = MockGameState(nxt['hp'], nxt['enemy_hp'])
            reward = tracker.compute_reward(next_state, turn.get('action'), turn)
            turn['reward'] = float(reward)
        else:
            # Terminal
            if winner == 'p1':
                # Ensure terminal bonus reflects a win
                turn['reward'] = float(100.0)
            elif winner == 'p2':
                turn['reward'] = float(-100.0)
            else:
                turn['reward'] = float(0.0)
            turn['terminal'] = True
        updated.append(turn)

    return updated


def read_jsonl(path: str) -> List[Dict]:
    turns: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    return turns


def write_jsonl(path: str, turns: List[Dict]) -> None:
    with open(path, 'w') as f:
        for t in turns:
            f.write(json.dumps(t) + '\n')


def process_file(jsonl_path: str, winner: Optional[str], engine_log: Optional[str]) -> Tuple[str, int]:
    turns = read_jsonl(jsonl_path)
    resolved_winner = determine_winner(turns, winner, engine_log)
    updated = compute_rewards_for_turns(turns, resolved_winner)
    write_jsonl(jsonl_path, updated)
    return resolved_winner, len(updated)


def main():
    parser = argparse.ArgumentParser(description='Compute rewards for BC jsonl turn logs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--jsonl', type=str, help='Path to a single episode_*_turns.jsonl file')
    group.add_argument('--dir', type=str, help='Directory containing jsonl files to process')
    parser.add_argument('--winner', type=str, choices=['p1', 'p2', 'draw'], help='Override winner')
    parser.add_argument('--engine-log', type=str, help='Path to engine stdout log to parse winner')

    args = parser.parse_args()

    processed = 0
    if args.jsonl:
        w, n = process_file(args.jsonl, args.winner, args.engine_log)
        print(f"Processed {os.path.basename(args.jsonl)}: winner={w}, turns={n}")
        processed += 1
    else:
        jsonls = sorted(glob.glob(os.path.join(args.dir, 'episode_*_turns.jsonl')))
        if not jsonls:
            print(f"No jsonl files found in {args.dir}")
            sys.exit(1)
        for path in jsonls:
            w, n = process_file(path, args.winner, args.engine_log)
            print(f"Processed {os.path.basename(path)}: winner={w}, turns={n}")
            processed += 1

    print(f"Done. Files processed: {processed}")


if __name__ == '__main__':
    main()


