import os
import sys
import glob
import pickle
import json
from typing import Tuple, Optional, List, Dict

import numpy as np

# Try to import torch - if not available, we'll do basic validation only
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def add_repo_paths():
    """Ensure we can import the PPO network from my-first-algo-ppo."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    algo_pkg = os.path.join(repo_root, "my-first-algo-ppo")
    rl_pkg = os.path.join(algo_pkg, "rl")
    # Prepend so local version wins
    for p in [repo_root, algo_pkg, rl_pkg]:
        if p not in sys.path:
            sys.path.insert(0, p)


add_repo_paths()

# Try to load the network model if torch is available
PPONet = None
if TORCH_AVAILABLE:
    try:
        # Prefer the new CNN class if present
        from network import CNNPPONetwork as PPONet  # type: ignore
    except Exception:
        try:
            # Fallback to legacy name
            from network import PPONetwork as PPONet  # type: ignore
        except Exception:
            PPONet = None


def parse_sample(sample: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (boards, scalars, actions) from the expert demo format.
    
    Expected format (from algo_strategy.py):
    {
        'state': {
            'board': numpy array of shape (5, 28, 28)
            'scalar': numpy array of shape (7,)
        },
        'action': int (action index 0-7)
    }
    
    Returns numpy arrays with shapes (1, 5, 28, 28), (1, 7), (1,)
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Sample must be a dict, got {type(sample)}")
    
    # Allow two formats:
    # A) Legacy flat dict: {'board': ..., 'scalar': ..., 'action': ...}
    # B) New jsonl turn dict: {'state': {'board': ..., 'scalar': ...}, 'action': ..., ...}
    if 'state' in sample and isinstance(sample['state'], dict):
        state = sample['state']
        if 'board' not in state or 'scalar' not in state:
            raise KeyError("State must contain 'board' and 'scalar'")
        board = np.asarray(state['board'])
        scalar = np.asarray(state['scalar'])
        if 'action' not in sample:
            raise KeyError("Missing 'action' in sample")
        action = sample['action']
    else:
        if 'board' not in sample or 'scalar' not in sample or 'action' not in sample:
            raise KeyError("Sample must contain 'board', 'scalar', and 'action'")
        board = np.asarray(sample['board'])
        scalar = np.asarray(sample['scalar'])
        action = sample['action']
    
    # Convert to proper shapes for batching
    if board.ndim == 3:
        # (5, 28, 28) -> (1, 5, 28, 28)
        board = board[np.newaxis, :]
    elif board.ndim == 4:
        # Already batched (1, 5, 28, 28)
        pass
    else:
        raise ValueError(f"Board must be 3D or 4D, got shape {board.shape}")
    
    if scalar.ndim == 1:
        # (7,) -> (1, 7)
        scalar = scalar[np.newaxis, :]
    elif scalar.ndim == 2:
        # Already batched (1, 7)
        pass
    else:
        raise ValueError(f"Scalar must be 1D or 2D, got shape {scalar.shape}")
    
    # Convert action to int array
    if isinstance(action, int):
        action = np.array([action])
    else:
        action = np.asarray(action)
    
    return board.astype(np.float32), scalar.astype(np.float32), action.astype(np.int64)


def run_forward(model: Optional[any], boards: np.ndarray, scalars: np.ndarray):
    """Run forward pass through the network if available."""
    if not TORCH_AVAILABLE or model is None:
        return None, None
    
    with torch.no_grad():
        board_t = torch.from_numpy(boards).float()
        scalar_t = torch.from_numpy(scalars).float()
        logits, value = model(board_t, scalar_t)
        return logits, value


def validate_episode_file(path: str, model: Optional[any]) -> None:
    """Validate that the data file is compatible with the network."""
    with open(path, "rb") as f:
        sample = pickle.load(f)

    # Parse the sample into tensors
    boards, scalars, actions = parse_sample(sample)

    # Validate shapes
    if boards.ndim != 4 or boards.shape[2:] != (28, 28):
        raise AssertionError(f"Boards must be (B,C,28,28), got {boards.shape}")
    
    if boards.shape[1] != 5:
        raise AssertionError(f"Board must have 5 channels, got {boards.shape[1]}")
    
    if scalars.ndim != 2:
        raise AssertionError(f"Scalars must be (B,S), got {scalars.shape}")
    
    if scalars.shape[1] != 7:
        raise AssertionError(f"Scalars must have 7 dimensions, got {scalars.shape[1]}")
    
    if actions.ndim != 1:
        raise AssertionError(f"Actions must be (B,), got {actions.shape}")
    
    # Check batch consistency
    batch_size = boards.shape[0]
    if scalars.shape[0] != batch_size:
        raise AssertionError(f"Batch size mismatch: boards={batch_size}, scalars={scalars.shape[0]}")
    
    if actions.shape[0] != batch_size:
        raise AssertionError(f"Batch size mismatch: boards={batch_size}, actions={actions.shape[0]}")

    # Run forward pass through the network if available
    logits, value = run_forward(model, boards, scalars)
    
    if logits is not None and value is not None:
        # Validate output shapes
        if logits.shape != (batch_size, 8):
            raise AssertionError(f"Action logits shape mismatch: {logits.shape}, expected ({batch_size}, 8)")
        
        if value.shape != (batch_size, 1):
            raise AssertionError(f"Value shape mismatch: {value.shape}, expected ({batch_size}, 1)")

        # Print success message with details
        print(
            f"✓ {os.path.basename(path)}: "
            f"Board{tuple(boards.shape)}, Scalar{tuple(scalars.shape)}, Action{tuple(actions.shape)}, "
            f"Logits{tuple(logits.shape)}, Value{tuple(value.shape)}"
        )
    else:
        # Basic validation only (no torch/model)
        print(
            f"✓ {os.path.basename(path)}: "
            f"Board{tuple(boards.shape)}, Scalar{tuple(scalars.shape)}, Action{tuple(actions.shape)} "
            f"[Basic validation only - torch not available]"
        )


def read_jsonl_turns(path: str) -> List[Dict]:
    turns: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    return turns


def validate_jsonl_file(path: str, model: Optional[any]) -> None:
    """Validate a jsonl episode composed of per-turn dicts."""
    turns = read_jsonl_turns(path)
    if not turns:
        raise AssertionError(f"No turns found in {os.path.basename(path)}")

    failures = 0
    ran_forward = False
    for idx, turn in enumerate(turns):
        try:
            boards, scalars, actions = parse_sample(turn)
            # Reuse core validation (shapes and optional forward)
            # We validate one turn at a time to keep memory small
            validate_episode_file_inner(boards, scalars, actions, model, file_hint=f"{os.path.basename(path)}#t{idx}")

            # Reward/terminal optional checks
            if 'reward' in turn and not isinstance(turn['reward'], (int, float)):
                raise AssertionError(f"Turn {idx}: reward must be numeric if present")
            if 'terminal' in turn and not isinstance(turn['terminal'], bool):
                raise AssertionError(f"Turn {idx}: terminal must be boolean if present")
            ran_forward = True
        except Exception as e:
            failures += 1
            print(f"FAIL {os.path.basename(path)} turn {idx}: {e}")

    # If rewards are present, last turn should be terminal
    if any('reward' in t for t in turns):
        if not turns[-1].get('terminal', False):
            raise AssertionError("Last turn is not marked terminal while rewards are present")

    if failures:
        raise AssertionError(f"{failures} turn(s) failed validation in {os.path.basename(path)}")

    print(f"✓ {os.path.basename(path)}: {len(turns)} turns validated{' with forward' if ran_forward else ''}.")


def validate_episode_file_inner(boards: np.ndarray, scalars: np.ndarray, actions: np.ndarray, model: Optional[any], file_hint: str = "") -> None:
    # Validate shapes
    if boards.ndim != 4 or boards.shape[2:] != (28, 28):
        raise AssertionError(f"Boards must be (B,C,28,28), got {boards.shape}")
    if boards.shape[1] != 5:
        raise AssertionError(f"Board must have 5 channels, got {boards.shape[1]}")
    if scalars.ndim != 2:
        raise AssertionError(f"Scalars must be (B,S), got {scalars.shape}")
    if scalars.shape[1] != 7:
        raise AssertionError(f"Scalars must have 7 dimensions, got {scalars.shape[1]}")
    if actions.ndim != 1:
        raise AssertionError(f"Actions must be (B,), got {actions.shape}")

    batch_size = boards.shape[0]
    if scalars.shape[0] != batch_size or actions.shape[0] != batch_size:
        raise AssertionError("Batch size mismatch among boards/scalars/actions")

    # Optional forward
    logits, value = run_forward(model, boards, scalars)
    if logits is not None and value is not None:
        if logits.shape != (batch_size, 8):
            raise AssertionError(f"Action logits shape mismatch: {logits.shape}, expected ({batch_size}, 8)")
        if value.shape != (batch_size, 1):
            raise AssertionError(f"Value shape mismatch: {value.shape}, expected ({batch_size}, 1)")


def main():
    """Main validation entry point."""
    print("=" * 80)
    print("BC Data Validation Script")
    print("Validates expert demonstration data compatibility with CNNPPONetwork")
    print("=" * 80)
    print()
    
    # Find data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "bc_data"))
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    files = [("pkl", f) for f in pkl_files] + [("jsonl", f) for f in jsonl_files]
    
    if not files:
        print(f"❌ No .pkl or .jsonl files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(files)} file(s) in {data_dir} ({len(pkl_files)} pkl, {len(jsonl_files)} jsonl)")
    print()
    
    # Instantiate model if available
    model = None
    if TORCH_AVAILABLE and PPONet is not None:
        try:
            model = PPONet()
            model.eval()
            print(f"✓ Loaded model: {model.__class__.__name__}")
            print(f"  Board channels: {model.board_channels}, Scalar dim: {model.scalar_dim}, Action dim: {model.action_dim}")
            print()
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("  Continuing with basic validation only...")
            print()
    else:
        print("⚠ Torch not available - performing basic data validation only")
        print("  To test full compatibility, install torch: pip install torch")
        print()

    # Validate each file
    failures = 0
    for kind, fp in files:
        try:
            if kind == "pkl":
                validate_episode_file(fp, model)
            else:
                validate_jsonl_file(fp, model)
        except Exception as e:
            failures += 1
            print(f"❌ {os.path.basename(fp)}: {type(e).__name__}: {e}")
    
    print()
    print("=" * 80)
    if failures:
        print(f"❌ Validation complete: {failures} file(s) failed out of {len(files)}")
        sys.exit(2)
    else:
        print(f"✓ Validation complete: All {len(files)} file(s) passed")
        if model is not None:
            print("✓ Data is compatible with CNNPPONetwork for behavior cloning")
        else:
            print("✓ Data format looks correct (full compatibility test requires torch)")
    print("=" * 80)


if __name__ == "__main__":
    main()


