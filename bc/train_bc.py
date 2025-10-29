#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def add_rl_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rl_root = os.path.join(repo_root, 'my-first-algo-ppo')
    if rl_root not in sys.path:
        sys.path.insert(0, rl_root)


add_rl_path()
from rl.network import CNNPPONetwork


class JsonlBCDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                state = d.get('state', {})
                board = np.asarray(state['board'], dtype=np.float32)
                scalar = np.asarray(state['scalar'], dtype=np.float32)
                action = int(d['action'])
                # Ensure shapes (5,28,28) and (7,)
                assert board.shape == (5, 28, 28), f"board shape {board.shape}"
                assert scalar.shape == (7,), f"scalar shape {scalar.shape}"
                self.samples.append((board, scalar, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        board, scalar, action = self.samples[idx]
        # To tensors
        board_t = torch.from_numpy(board)  # (5,28,28)
        scalar_t = torch.from_numpy(scalar)  # (7,)
        action_t = torch.tensor(action, dtype=torch.long)
        return board_t, scalar_t, action_t


def collate_fn(batch):
    boards = torch.stack([b for b, s, a in batch], dim=0)  # (B,5,28,28)
    scalars = torch.stack([s for b, s, a in batch], dim=0)  # (B,7)
    actions = torch.stack([a for b, s, a in batch], dim=0)  # (B,)
    return boards, scalars, actions


def train_one_epoch(model: CNNPPONetwork, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for boards, scalars, actions in loader:
        boards = boards.to(device)
        scalars = scalars.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        logits, _ = model(boards, scalars)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * actions.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == actions).sum().item()
        total += actions.size(0)

    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def evaluate(model: CNNPPONetwork, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for boards, scalars, actions in loader:
        boards = boards.to(device)
        scalars = scalars.to(device)
        actions = actions.to(device)

        logits, _ = model(boards, scalars)
        loss = criterion(logits, actions)

        total_loss += loss.item() * actions.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == actions).sum().item()
        total += actions.size(0)

    return total_loss / max(1, total), total_correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description='Behavior Cloning pretraining for CNNPPONetwork')
    parser.add_argument('--data-dir', type=str, default=None, help='Directory containing train.jsonl and val.jsonl (default: ../bc_data)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience on val loss')
    parser.add_argument('--logdir', type=str, default=None, help='TensorBoard log directory (default: ../my-first-algo-ppo/logs/bc)')
    parser.add_argument('--savedir', type=str, default=None, help='Directory to save best model (default: bc/checkpoints)')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = args.data_dir or os.path.join(repo_root, 'bc_data')
    logdir = args.logdir or os.path.join(repo_root, 'my-first-algo-ppo', 'logs', 'bc')
    savedir = args.savedir or os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    train_jsonl = os.path.join(data_dir, 'train.jsonl')
    val_jsonl = os.path.join(data_dir, 'val.jsonl')
    if not (os.path.exists(train_jsonl) and os.path.exists(val_jsonl)):
        print(f"Missing train/val jsonl. Please run: python bc/prepare_bc_dataset.py")
        sys.exit(1)

    # Datasets and loaders
    train_ds = JsonlBCDataset(train_jsonl)
    val_ds = JsonlBCDataset(val_jsonl)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNPPONetwork(board_channels=5, scalar_dim=7, action_dim=8, hidden_dim=128).to(device)

    # Loss only on actor logits (critic ignored in BC)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(logdir)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_path = os.path.join(savedir, 'bc_pretrain.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('acc/train', train_acc, epoch)
        writer.add_scalar('acc/val', val_acc, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        # Early stopping on val loss
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict()}, best_path)
            print(f"Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print("BC pretraining finished.")


if __name__ == '__main__':
    main()


