"""
Step 5: Split Dataset into Train/Valid/Test
============================================
평가 결과를 기반으로 올바르게 판단된 샘플만 선택하여 split합니다.

- Train: 800 samples
- Valid: 200 samples  
- Test (Gold): 300 samples (correctly classified only)

Output: data/final/train.json, valid.json, test.json
"""
import json
import random
from pathlib import Path
from collections import defaultdict

def load_evaluation_results():
    """Load and filter correctly classified samples"""
    eval_file = Path("generation/outputs/evaluated/evaluation_results.json")
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # Filter only correct predictions
    correct_samples = [s for s in all_samples if s.get("is_correct", False)]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Correctly classified: {len(correct_samples)}")
    print(f"Filtered out: {len(all_samples) - len(correct_samples)}")
    
    return correct_samples

def split_dataset(samples, train_size=800, valid_size=200):
    """Split samples into train/valid/test"""
    # Group by label
    by_label = defaultdict(list)
    for sample in samples:
        label = sample["ground_truth_label"]
        by_label[label].append(sample)
    
    print("\nSamples per label:")
    for label, items in by_label.items():
        print(f"  {label}: {len(items)}")
    
    # Shuffle each label
    random.seed(42)
    for label in by_label:
        random.shuffle(by_label[label])
    
    # Split proportionally
    train_samples = []
    valid_samples = []
    test_samples = []
    
    for label, items in by_label.items():
        total = len(items)
        
        # Calculate splits (proportional to original distribution)
        train_ratio = train_size / (train_size + valid_size + 300)
        valid_ratio = valid_size / (train_size + valid_size + 300)
        
        train_count = int(total * train_ratio)
        valid_count = int(total * valid_ratio)
        
        # Ensure we have enough for all splits
        if train_count + valid_count >= total:
            # Adjust to leave some for test
            test_count = max(1, total // 10)  # At least 10% for test
            train_count = int((total - test_count) * train_ratio / (train_ratio + valid_ratio))
            valid_count = total - train_count - test_count
        else:
            test_count = total - train_count - valid_count
        
        train_samples.extend(items[:train_count])
        valid_samples.extend(items[train_count:train_count+valid_count])
        test_samples.extend(items[train_count+valid_count:])
        
        print(f"  {label}: train={train_count}, valid={valid_count}, test={test_count}")
    
    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(test_samples)
    
    return train_samples, valid_samples, test_samples

def clean_sample(sample):
    """Remove evaluation metadata from sample"""
    keys_to_remove = ["predicted_label", "is_correct", "judge_response"]
    return {k: v for k, v in sample.items() if k not in keys_to_remove}

def main():
    # Load correct samples
    correct_samples = load_evaluation_results()
    
    # Split dataset
    print("\n" + "="*70)
    print("Splitting dataset...")
    print("="*70)
    
    train, valid, test = split_dataset(correct_samples)
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train)}")
    print(f"  Valid: {len(valid)}")
    print(f"  Test:  {len(test)}")
    
    # Clean samples (remove eval metadata)
    train = [clean_sample(s) for s in train]
    valid = [clean_sample(s) for s in valid]
    test = [clean_sample(s) for s in test]
    
    # Save to data/final/
    output_dir = Path("data/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "valid.json", 'w', encoding='utf-8') as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    
    # Also split test by label for detailed evaluation
    test_by_label = defaultdict(list)
    for sample in test:
        test_by_label[sample["ground_truth_label"]].append(sample)
    
    for label, samples in test_by_label.items():
        filename = output_dir / f"test_{label}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filename} ({len(samples)} samples)")
    
    # Copy metadata
    eval_file = Path("generation/outputs/evaluated/evaluation_results.json")
    metadata_file = output_dir / "metadata.json"
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    metadata = {
        "total_generated": len(eval_data),
        "correctly_classified": len(correct_samples),
        "accuracy": len(correct_samples) / len(eval_data) * 100,
        "train_size": len(train),
        "valid_size": len(valid),
        "test_size": len(test)
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Saved to: {output_dir}/")

if __name__ == "__main__":
    main()
