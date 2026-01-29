"""
Step 5: Label turn samples using LLM-judge.

Input: turn_samples_{train/val/test}.jsonl
Output: labeled_turns_{train/val/test}.jsonl
"""
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_jsonl, save_jsonl, print_stats, load_yaml
from src.llm.openai_client import OpenAIClient
from src.llm.judge import label_turn_sample


def compute_label_stats(samples):
    """Compute statistics about labeled samples."""
    total = len(samples)
    
    # Count labels
    label_counts = {'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'V5': 0}
    top_violation_counts = {}
    
    for sample in samples:
        labels = sample.get('labels', {})
        top_v = sample.get('top_violation', 'None')
        
        # Count individual labels
        for v in ['V1', 'V2', 'V3', 'V4', 'V5']:
            if labels.get(v, 0) == 1:
                label_counts[v] += 1
        
        # Count top violations
        top_violation_counts[top_v] = top_violation_counts.get(top_v, 0) + 1
    
    stats = {
        "total_samples": total,
        "label_distribution": label_counts,
        "top_violation_distribution": top_violation_counts,
    }
    
    # Compute multi-label stats
    multi_label_count = 0
    for sample in samples:
        labels = sample.get('labels', {})
        label_sum = sum(labels.get(v, 0) for v in ['V1', 'V2', 'V3', 'V4', 'V5'])
        if label_sum > 1:
            multi_label_count += 1
    
    stats["samples_with_multiple_labels"] = multi_label_count
    
    return stats


def process_split(samples, split_name, llm_client, verbose=True):
    """
    Label one split using LLM-judge.
    
    Args:
        samples: List of turn sample dicts
        split_name: 'train', 'val', or 'test'
        llm_client: OpenAI client
        verbose: Print progress
    
    Returns:
        List of labeled samples
    """
    print(f"\nLabeling {split_name} samples ({len(samples)} samples)...")
    print("(This will call OpenAI API - may take several minutes)")
    
    labeled_samples = []
    failed_count = 0
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(samples)} (failed: {failed_count})")
        
        labeled = label_turn_sample(sample, llm_client, verbose=False)
        
        if labeled:
            labeled_samples.append(labeled)
        else:
            failed_count += 1
            if verbose:
                print(f"  Warning: Failed to label sample {i + 1}")
    
    print(f"  Progress: {len(samples)}/{len(samples)} (failed: {failed_count})")
    print(f"  Successfully labeled: {len(labeled_samples)}/{len(samples)}")
    
    return labeled_samples


def main(args):
    print(f"\n{'='*60}")
    print("STEP 5: Label Turn Samples with LLM-Judge")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml(args.config)
    
    # Initialize LLM client
    print(f"Initializing LLM client (model: {config['llm']['model']})...")
    llm_client = OpenAIClient(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens_judge'],
        timeout=config['llm']['timeout'],
        max_retries=config['llm']['max_retries']
    )
    
    # Process each split
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        input_path = input_dir / f"turn_samples_{split_name}.jsonl"
        
        if not input_path.exists():
            print(f"\nWarning: {input_path} not found, skipping {split_name}")
            continue
        
        # Load samples
        samples = load_jsonl(str(input_path))
        
        # Label samples
        labeled_samples = process_split(samples, split_name, llm_client, verbose=True)
        
        # Save
        output_path = output_dir / f"labeled_turns_{split_name}.jsonl"
        save_jsonl(labeled_samples, str(output_path))
        print(f"  Saved to: {output_path}")
        
        # Stats
        stats = compute_label_stats(labeled_samples)
        print_stats(f"{split_name.upper()} Labeled Samples", {
            "Total samples": stats['total_samples'],
            "Multi-label samples": stats['samples_with_multiple_labels'],
        })
        
        print(f"\nLabel distribution ({split_name}):")
        for v in sorted(stats['label_distribution'].keys()):
            print(f"  {v}: {stats['label_distribution'][v]}")
        
        print(f"\nTop violation distribution ({split_name}):")
        for v in sorted(stats['top_violation_distribution'].keys(), 
                       key=lambda x: stats['top_violation_distribution'][x], 
                       reverse=True)[:8]:
            print(f"  {v}: {stats['top_violation_distribution'][v]}")
    
    print("\n✅ Step 5 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/turn_samples",
                        help="Directory with turn sample JSONL files")
    parser.add_argument("--output_dir", type=str, default="data/labeled",
                        help="Output directory for labeled JSONL files")
    parser.add_argument("--config", type=str, default="configs/poc.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    main(args)
