"""
Step 1: Sample 200 original sessions from ESConv.json

Input: ESConv.json (raw data)
Output: sessions_original_200.json + statistics
"""
import argparse
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, save_json, print_stats


def assign_session_ids(sessions):
    """Assign unique session IDs to sessions (original format has no ID)."""
    for i, session in enumerate(sessions):
        session['session_id'] = f"orig_{i:04d}"
    return sessions


def compute_session_stats(sessions):
    """Compute statistics about sessions."""
    stats = {
        "total_sessions": len(sessions),
        "total_turns": sum(len(s['dialog']) for s in sessions),
        "avg_turns_per_session": round(sum(len(s['dialog']) for s in sessions) / len(sessions), 2),
        "supporter_turns": sum(sum(1 for turn in s['dialog'] if turn.get('speaker') == 'supporter') 
                               for s in sessions),
        "seeker_turns": sum(sum(1 for turn in s['dialog'] if turn.get('speaker') == 'seeker') 
                            for s in sessions),
    }
    
    # Emotion/problem type distribution
    emotion_types = {}
    problem_types = {}
    for s in sessions:
        et = s.get('emotion_type', 'unknown')
        pt = s.get('problem_type', 'unknown')
        emotion_types[et] = emotion_types.get(et, 0) + 1
        problem_types[pt] = problem_types.get(pt, 0) + 1
    
    stats['emotion_types'] = emotion_types
    stats['problem_types'] = problem_types
    
    return stats


def main(args):
    print(f"\n{'='*60}")
    print("STEP 1: Sample Original Sessions")
    print(f"{'='*60}\n")
    
    # Load ESConv.json
    print(f"Loading ESConv data from: {args.input}")
    all_sessions = load_json(args.input)
    print(f"  Total sessions in ESConv: {len(all_sessions)}")
    
    # Sample N sessions
    random.seed(args.seed)
    if len(all_sessions) < args.num_sessions:
        print(f"  Warning: Requested {args.num_sessions} but only {len(all_sessions)} available. Using all.")
        sampled = all_sessions
    else:
        sampled = random.sample(all_sessions, args.num_sessions)
    
    print(f"  Sampled: {len(sampled)} sessions (seed={args.seed})")
    
    # Assign session IDs
    sampled = assign_session_ids(sampled)
    
    # Compute stats
    stats = compute_session_stats(sampled)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(sampled, args.output)
    print(f"\nSaved to: {args.output}")
    
    # Print stats
    print_stats("Original Sessions Statistics", {
        "Total sessions": stats['total_sessions'],
        "Total turns": stats['total_turns'],
        "Avg turns/session": stats['avg_turns_per_session'],
        "Supporter turns": stats['supporter_turns'],
        "Seeker turns": stats['seeker_turns'],
    })
    
    print("\nEmotion types:")
    for k, v in sorted(stats['emotion_types'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v}")
    
    print("\nProblem types:")
    for k, v in sorted(stats['problem_types'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v}")
    
    print("\n✅ Step 1 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="ESConv.json", 
                        help="Path to ESConv.json")
    parser.add_argument("--output", type=str, default="data/sessions_original_200.json",
                        help="Output path for sampled sessions")
    parser.add_argument("--num_sessions", type=int, default=200,
                        help="Number of sessions to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)
