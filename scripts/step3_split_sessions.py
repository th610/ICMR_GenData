"""
Step 3: Split sessions into train/val/test sets (session-level split).

Input: sessions_original_50.json + sessions_synth_50.json
Output: sessions_train.json, sessions_val.json, sessions_test.json
"""
import argparse
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, save_json, print_stats, load_yaml


def split_sessions(sessions, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split sessions into train/val/test.
    
    Args:
        sessions: List of session dicts
        train_ratio: Train split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
    
    Returns:
        (train_sessions, val_sessions, test_sessions)
    """
    random.seed(seed)
    
    # Shuffle
    shuffled = sessions.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def compute_split_stats(sessions, split_name):
    """Compute statistics for a split."""
    total_sessions = len(sessions)
    
    # Count original vs synthetic
    original_count = sum(1 for s in sessions if 'orig_' in s.get('session_id', ''))
    synthetic_count = sum(1 for s in sessions if 'synth_' in s.get('session_id', ''))
    
    # Count violations in synthetic
    violation_counts = {}
    for s in sessions:
        if 'injected_violation' in s:
            v_type = s['injected_violation']['type']
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    stats = {
        f"{split_name}_total": total_sessions,
        f"{split_name}_original": original_count,
        f"{split_name}_synthetic": synthetic_count,
    }
    
    if violation_counts:
        stats[f"{split_name}_violations"] = violation_counts
    
    return stats


def main(args):
    print(f"\n{'='*60}")
    print("STEP 3: Split Sessions (Train/Val/Test)")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml(args.config)
    
    # Load original and synthetic sessions
    print(f"Loading original sessions from: {args.original}")
    original_sessions = load_json(args.original)
    print(f"  Original sessions: {len(original_sessions)}")
    
    print(f"Loading synthetic sessions from: {args.synthetic}")
    synthetic_sessions = load_json(args.synthetic)
    print(f"  Synthetic sessions: {len(synthetic_sessions)}")
    
    # Combine
    all_sessions = original_sessions + synthetic_sessions
    print(f"\nTotal sessions: {len(all_sessions)}")
    
    # Split
    train_ratio = config['split']['train']
    val_ratio = config['split']['val']
    test_ratio = config['split']['test']
    
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    train, val, test = split_sessions(
        all_sessions, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "sessions_train.json"
    val_path = output_dir / "sessions_val.json"
    test_path = output_dir / "sessions_test.json"
    
    save_json(train, str(train_path))
    save_json(val, str(val_path))
    save_json(test, str(test_path))
    
    print(f"\nSaved splits to: {output_dir}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    
    # Compute and print stats
    train_stats = compute_split_stats(train, "train")
    val_stats = compute_split_stats(val, "val")
    test_stats = compute_split_stats(test, "test")
    
    print_stats("Train Split", train_stats)
    print_stats("Val Split", val_stats)
    print_stats("Test Split", test_stats)
    
    print("\n✅ Step 3 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, default="data/sessions_original_50.json",
                        help="Path to original sessions")
    parser.add_argument("--synthetic", type=str, default="data/sessions_synth_50.json",
                        help="Path to synthetic sessions")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--config", type=str, default="configs/poc.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)
