#!/usr/bin/env python3
import os
import sys
import json
import glob
import random
from typing import List


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(repo_root, 'bc_data')
    os.makedirs(data_dir, exist_ok=True)

    # Find all episode_*_turns.jsonl
    files = sorted(glob.glob(os.path.join(data_dir, 'episode_*_turns.jsonl')))
    if not files:
        print(f"No episode_*_turns.jsonl found in {data_dir}")
        sys.exit(1)

    # Split by episode: 80% train, 20% val (round down to ensure val >=1 if possible)
    random.seed(42)
    files_shuffled = files[:]
    random.shuffle(files_shuffled)

    n = len(files_shuffled)
    n_train = max(1, int(n * 0.8)) if n > 1 else 1
    if n_train == n:
        n_train = n - 1 if n > 1 else 1

    train_files = files_shuffled[:n_train]
    val_files = files_shuffled[n_train:]
    if not val_files:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    # Write combined train.jsonl and val.jsonl for convenience
    def concat_jsonl(sources: List[str], target: str):
        count = 0
        with open(target, 'w') as out:
            for src in sources:
                with open(src, 'r') as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            count += 1
        return count

    train_out = os.path.join(data_dir, 'train.jsonl')
    val_out = os.path.join(data_dir, 'val.jsonl')
    train_count = concat_jsonl(train_files, train_out)
    val_count = concat_jsonl(val_files, val_out)

    # Also write file lists
    with open(os.path.join(data_dir, 'train_files.txt'), 'w') as f:
        f.write('\n'.join(train_files) + '\n')
    with open(os.path.join(data_dir, 'val_files.txt'), 'w') as f:
        f.write('\n'.join(val_files) + '\n')

    print(f"Prepared dataset in {data_dir}")
    print(f"Episodes: total={n}, train={len(train_files)}, val={len(val_files)}")
    print(f"Samples: train={train_count}, val={val_count}")


if __name__ == '__main__':
    main()


