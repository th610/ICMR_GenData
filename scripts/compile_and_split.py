"""
Compile all generated data and split into train/valid/test

Final dataset compilation:
- Normal: 400
- V1: 240
- V2: 160
- V3: 200
- V4: 150
- V5: 150
Total: 1300

Split: 70/15/15 (train/valid/test) with stratification
"""
import json
import random
from pathlib import Path
from collections import Counter
from src.utils import load_json, save_json


def compile_dataset():
    """모든 데이터 파일을 하나로 합치기"""
    
    print("="*80)
    print("Step 1: Compiling Dataset")
    print("="*80)
    
    data_dir = Path("data/generated")
    pilot_dir = Path("data/pilot")
    
    all_sessions = []
    
    # Load each class
    files = [
        (data_dir / "normal_400.json", "normal", 400),
        (data_dir / "v1_240.json", "v1", 240),
        (data_dir / "v2_160.json", "v2", 160),
        (data_dir / "v3_200.json", "v3", 200),
        (pilot_dir / "v4_full_150.json", "v4", 150),
        (pilot_dir / "v5_full_150.json", "v5", 150)
    ]
    
    label_counts = {}
    
    for filepath, label, expected in files:
        print(f"\n📂 Loading {filepath.name}")
        data = load_json(str(filepath))
        
        # Add label to each session
        for session in data:
            session['label'] = label
            all_sessions.append(session)
        
        label_counts[label] = len(data)
        print(f"   Loaded: {len(data)} sessions (expected: {expected})")
        
        if len(data) != expected:
            print(f"   ⚠️  Warning: Count mismatch!")
    
    print(f"\n{'='*80}")
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Distribution: {label_counts}")
    print(f"{'='*80}\n")
    
    return all_sessions, label_counts


def stratified_split(sessions, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified split by label"""
    
    print("="*80)
    print("Step 2: Stratified Split")
    print("="*80)
    print(f"Ratios: Train={train_ratio}, Valid={valid_ratio}, Test={test_ratio}")
    print()
    
    random.seed(seed)
    
    # Group by label
    by_label = {}
    for session in sessions:
        label = session['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(session)
    
    # Shuffle each label group
    for label in by_label:
        random.shuffle(by_label[label])
    
    train_set = []
    valid_set = []
    test_set = []
    
    split_info = {}
    
    for label, label_sessions in by_label.items():
        total = len(label_sessions)
        
        train_size = int(total * train_ratio)
        valid_size = int(total * valid_ratio)
        # test gets remainder to ensure total matches
        
        train_portion = label_sessions[:train_size]
        valid_portion = label_sessions[train_size:train_size + valid_size]
        test_portion = label_sessions[train_size + valid_size:]
        
        train_set.extend(train_portion)
        valid_set.extend(valid_portion)
        test_set.extend(test_portion)
        
        split_info[label] = {
            'total': total,
            'train': len(train_portion),
            'valid': len(valid_portion),
            'test': len(test_portion)
        }
        
        print(f"{label.upper():8s}: {len(train_portion):3d} / {len(valid_portion):3d} / {len(test_portion):3d}  (total: {total})")
    
    # Shuffle final sets
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)
    
    print()
    print(f"{'='*80}")
    print(f"Train: {len(train_set)} sessions")
    print(f"Valid: {len(valid_set)} sessions")
    print(f"Test:  {len(test_set)} sessions")
    print(f"Total: {len(train_set) + len(valid_set) + len(test_set)} sessions")
    print(f"{'='*80}\n")
    
    return train_set, valid_set, test_set, split_info


def save_datasets(train, valid, test, split_info, label_counts):
    """Save compiled and split datasets"""
    
    print("="*80)
    print("Step 3: Saving Datasets")
    print("="*80)
    
    output_dir = Path("data/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "train.json"
    valid_path = output_dir / "valid.json"
    test_path = output_dir / "test.json"
    
    save_json(train, str(train_path))
    print(f"✅ Train: {train_path}")
    
    save_json(valid, str(valid_path))
    print(f"✅ Valid: {valid_path}")
    
    save_json(test, str(test_path))
    print(f"✅ Test:  {test_path}")
    
    # Save metadata
    metadata = {
        "created": "2026-01-31",
        "total_sessions": len(train) + len(valid) + len(test),
        "distribution": label_counts,
        "split": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test)
        },
        "split_detail": split_info
    }
    
    metadata_path = output_dir / "metadata.json"
    save_json(metadata, str(metadata_path))
    print(f"✅ Metadata: {metadata_path}")
    
    print(f"\n{'='*80}")
    print("All datasets saved successfully!")
    print(f"{'='*80}\n")


def main():
    print("\n" + "="*80)
    print("Dataset Compilation and Split")
    print("="*80 + "\n")
    
    # Step 1: Compile
    all_sessions, label_counts = compile_dataset()
    
    # Step 2: Split
    train, valid, test, split_info = stratified_split(all_sessions)
    
    # Step 3: Save
    save_datasets(train, valid, test, split_info, label_counts)
    
    print("\n✅ Done! Ready for training.\n")


if __name__ == "__main__":
    main()
