"""
Step 2: Generate 200 synthetic sessions by rewriting one supporter turn per session.

Input: sessions_original_200.json
Output: sessions_synth_200.json + violation statistics
"""
import argparse
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, save_json, print_stats, load_yaml
from src.llm.openai_client import OpenAIClient
from src.synth.rewrite_turn import rewrite_session_with_violation


def create_violation_distribution(total: int, ratios: dict) -> list:
    """
    Create a list of violation types based on distribution ratios.
    
    Args:
        total: Total number of synthetic sessions
        ratios: Dict of violation -> ratio (e.g., {'V1': 0.25, ...})
    
    Returns:
        List of violation types (e.g., ['V1', 'V1', ..., 'V2', ...])
    """
    violations = []
    for v_type, ratio in ratios.items():
        count = int(total * ratio)
        violations.extend([v_type] * count)
    
    # Handle rounding errors
    while len(violations) < total:
        violations.append(random.choice(list(ratios.keys())))
    
    violations = violations[:total]  # Trim if over
    random.shuffle(violations)
    
    return violations


def main(args):
    print(f"\n{'='*60}")
    print("STEP 2: Generate Synthetic Sessions")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml(args.config)
    
    # Load original sessions
    print(f"Loading original sessions from: {args.input}")
    original_sessions = load_json(args.input)
    print(f"  Total original sessions: {len(original_sessions)}")
    
    # Setup
    random.seed(args.seed)
    num_synth = args.num_sessions or config['sampling']['synthetic_sessions']
    
    if len(original_sessions) < num_synth:
        print(f"  Warning: Need {num_synth} sessions but only {len(original_sessions)} available.")
        num_synth = len(original_sessions)
    
    # Create violation distribution
    violation_ratios = {k: v['ratio'] for k, v in config['violations'].items()}
    turn_ranges = {k: v['turn_range'] for k, v in config['violations'].items()}
    
    violations = create_violation_distribution(num_synth, violation_ratios)
    
    print(f"\nViolation distribution (target):")
    for v in sorted(set(violations)):
        print(f"  {v}: {violations.count(v)}")
    
    # Sample sessions to use as base
    base_sessions = random.sample(original_sessions, num_synth)
    
    # Initialize LLM client
    print(f"\nInitializing LLM client (model: {config['llm']['model']})...")
    llm_client = OpenAIClient(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens_rewrite'],
        timeout=config['llm']['timeout'],
        max_retries=config['llm']['max_retries']
    )
    
    # Generate synthetic sessions
    print(f"\nGenerating {num_synth} synthetic sessions...")
    print("(This will call OpenAI API - may take several minutes)")
    
    synthetic_sessions = []
    failed_sessions = []
    
    for i, (base_session, violation_type) in enumerate(zip(base_sessions, violations)):
        if (i + 1) % 10 == 0 or i == 0:  # Every 10 items + first item
            print(f"  Progress: {i + 1}/{num_synth} (failed: {len(failed_sessions)}) - Current: {violation_type}")
        
        synth_session = rewrite_session_with_violation(
            session=base_session,
            violation_type=violation_type,
            turn_ranges=turn_ranges,
            llm_client=llm_client,
            verbose=False
        )
        
        if synth_session:
            synthetic_sessions.append(synth_session)
        else:
            failed_sessions.append(i)
    
    print(f"  Progress: {num_synth}/{num_synth} (failed: {len(failed_sessions)})")
    
    # Compute statistics
    actual_violations = {}
    for sess in synthetic_sessions:
        v_type = sess.get('injected_violation', {}).get('type', 'unknown')
        actual_violations[v_type] = actual_violations.get(v_type, 0) + 1
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(synthetic_sessions, args.output)
    print(f"\nSaved to: {args.output}")
    
    # Print stats
    print_stats("Synthetic Sessions Statistics", {
        "Total synthetic sessions": len(synthetic_sessions),
        "Failed sessions": len(failed_sessions),
        "Success rate": f"{len(synthetic_sessions) / num_synth * 100:.1f}%",
    })
    
    print("\nActual violation distribution:")
    for v in sorted(actual_violations.keys()):
        print(f"  {v}: {actual_violations[v]}")
    
    if failed_sessions:
        print(f"\n⚠️  Warning: {len(failed_sessions)} sessions failed. You may want to retry.")
    
    print("\n✅ Step 2 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sessions_original_200.json",
                        help="Path to original sessions")
    parser.add_argument("--output", type=str, default="data/sessions_synth_200.json",
                        help="Output path for synthetic sessions")
    parser.add_argument("--config", type=str, default="configs/poc.yaml",
                        help="Path to config file")
    parser.add_argument("--num_sessions", type=int, default=None,
                        help="Number of synthetic sessions (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)
