"""
Step 4: Create turn-level samples from sessions.

Input: sessions_{train/val/test}.json
Output: turn_samples_{train/val/test}.jsonl
"""
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_json, save_jsonl, print_stats, load_yaml
from src.data.make_turn_samples import process_session_to_turn_samples
from src.llm.openai_client import OpenAIClient


def process_split(sessions, split_name, config, llm_client=None, seed=42):
    """
    Process one split (train/val/test) to create turn samples.
    
    Args:
        sessions: List of session dicts
        split_name: 'train', 'val', or 'test'
        config: Config dict
        llm_client: OpenAI client (optional)
        seed: Random seed
    
    Returns:
        List of turn samples
    """
    print(f"\nProcessing {split_name} split ({len(sessions)} sessions)...")
    
    all_samples = []
    
    for i, session in enumerate(sessions):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(sessions)}")
        
        samples = process_session_to_turn_samples(
            session,
            config,
            llm_client,
            seed=seed + i  # Different seed per session
        )
        
        all_samples.extend(samples)
    
    print(f"  Progress: {len(sessions)}/{len(sessions)}")
    print(f"  Generated {len(all_samples)} turn samples")
    
    return all_samples


def compute_sample_stats(samples):
    """Compute statistics about turn samples."""
    total = len(samples)
    
    # Count by session type
    session_ids = [s['session_id'] for s in samples]
    original_count = sum(1 for sid in session_ids if 'orig_' in sid)
    synthetic_count = sum(1 for sid in session_ids if 'synth_' in sid)
    
    # Average context length
    avg_context_len = sum(len(s['context_turns']) for s in samples) / total if total > 0 else 0
    
    # Average summary length
    avg_summary_len = sum(len(s['summary']) for s in samples) / total if total > 0 else 0
    
    stats = {
        "total_samples": total,
        "from_original_sessions": original_count,
        "from_synthetic_sessions": synthetic_count,
        "avg_context_turns": round(avg_context_len, 2),
        "avg_summary_bullets": round(avg_summary_len, 2),
    }
    
    return stats


def main(args):
    print(f"\n{'='*60}")
    print("STEP 4: Create Turn-Level Samples")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml(args.config)
    
    # Initialize LLM client if needed
    use_llm_summary = config['sampling'].get('use_llm_summary', False)
    llm_client = None
    
    if use_llm_summary:
        print("Initializing LLM client for summaries...")
        llm_client = OpenAIClient(
            model=config['llm']['model'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens_summary'],
            timeout=config['llm']['timeout'],
            max_retries=config['llm']['max_retries']
        )
    else:
        print("Using rule-based summaries (faster)")
    
    # Process each split
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        input_path = input_dir / f"sessions_{split_name}.json"
        
        if not input_path.exists():
            print(f"\nWarning: {input_path} not found, skipping {split_name}")
            continue
        
        # Load sessions
        sessions = load_json(str(input_path))
        
        # Process to turn samples
        samples = process_split(sessions, split_name, config, llm_client, seed=args.seed)
        
        # Save
        output_path = output_dir / f"turn_samples_{split_name}.jsonl"
        save_jsonl(samples, str(output_path))
        print(f"  Saved to: {output_path}")
        
        # Stats
        stats = compute_sample_stats(samples)
        print_stats(f"{split_name.upper()} Turn Samples", stats)
    
    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/splits",
                        help="Directory with split session files")
    parser.add_argument("--output_dir", type=str, default="data/turn_samples",
                        help="Output directory for turn sample JSONL files")
    parser.add_argument("--config", type=str, default="configs/poc.yaml",
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)
