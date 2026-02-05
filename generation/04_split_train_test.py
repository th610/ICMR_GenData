"""
Split 1300 samples into 1000 train + 300 test (stratified by label)
"""
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# Load merged data
with open('data/final/all_labels_1300.json', 'r') as f:
    all_samples = json.load(f)

print("="*70)
print(f"Splitting {len(all_samples)} samples")
print("="*70)

# Get labels
labels = [s['primary_label'] for s in all_samples]
print("\nOriginal distribution:")
for label, count in Counter(labels).items():
    print(f"  {label}: {count}")

# Stratified split: 1000 train, 300 test
train_samples, test_samples = train_test_split(
    all_samples,
    test_size=300,
    random_state=42,
    stratify=labels
)

print(f"\n✓ Split complete:")
print(f"  Train: {len(train_samples)}")
print(f"  Test: {len(test_samples)}")

# Verify distributions
train_labels = [s['primary_label'] for s in train_samples]
test_labels = [s['primary_label'] for s in test_samples]

print("\nTrain distribution:")
for label, count in Counter(train_labels).items():
    print(f"  {label}: {count}")

print("\nTest distribution:")
for label, count in Counter(test_labels).items():
    print(f"  {label}: {count}")

# Save files
train_path = Path('data/final/train_gold_1000.json')
test_path = Path('data/final/test_gold_300.json')

with open(train_path, 'w') as f:
    json.dump(train_samples, f, indent=2, ensure_ascii=False)

with open(test_path, 'w') as f:
    json.dump(test_samples, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved:")
print(f"  {train_path}")
print(f"  {test_path}")
