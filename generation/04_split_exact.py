"""
Split 1300 samples with exact distribution specified by user
Train 1000: Normal 445, V1 150, V3 150, V2 140, V4 70, V5 45
Test(gold) 300: Normal 80, V2 60, V1 50, V3 50, V4 30, V5 30
"""
import json
from pathlib import Path
from collections import defaultdict
import random

# Load merged data
with open('data/final/all_labels_1300.json', 'r') as f:
    all_samples = json.load(f)

# Target distributions
train_dist = {
    'Normal': 445,
    'V1': 150,
    'V3': 150,
    'V2': 140,
    'V4': 70,
    'V5': 45
}

test_dist = {
    'Normal': 80,
    'V2': 60,
    'V1': 50,
    'V3': 50,
    'V4': 30,
    'V5': 30
}

print("="*70)
print("Splitting with user-specified distribution")
print("="*70)

# Group by label
samples_by_label = defaultdict(list)
for sample in all_samples:
    samples_by_label[sample['primary_label']].append(sample)

print("\nAvailable samples:")
for label in sorted(samples_by_label.keys()):
    print(f"  {label}: {len(samples_by_label[label])}")

# Set random seed for reproducibility
random.seed(42)

# Shuffle each label group
for label in samples_by_label:
    random.shuffle(samples_by_label[label])

# Split samples
train_samples = []
test_samples = []

for label in train_dist.keys():
    train_count = train_dist[label]
    test_count = test_dist[label]
    
    available = samples_by_label[label]
    if len(available) < train_count + test_count:
        print(f"⚠️  Warning: Not enough {label} samples ({len(available)} < {train_count + test_count})")
    
    train_samples.extend(available[:train_count])
    test_samples.extend(available[train_count:train_count + test_count])

print(f"\n✓ Split complete:")
print(f"  Train: {len(train_samples)}")
print(f"  Test (gold): {len(test_samples)}")

# Verify distributions
from collections import Counter
train_labels = Counter([s['primary_label'] for s in train_samples])
test_labels = Counter([s['primary_label'] for s in test_samples])

print("\nTrain distribution:")
for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
    print(f"  {label}: {train_labels[label]}")

print("\nTest (gold) distribution:")
for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
    print(f"  {label}: {test_labels[label]}")

# Create output with metadata
train_output = {
    "metadata": {
        "total_samples": len(train_samples),
        "split": "train",
        "distribution": dict(train_labels)
    },
    "samples": train_samples
}

test_output = {
    "metadata": {
        "total_samples": len(test_samples),
        "split": "test_gold",
        "distribution": dict(test_labels)
    },
    "samples": test_samples
}

# Save files
train_path = Path('data/final/train_1000.json')
test_path = Path('data/final/test_gold_300.json')

with open(train_path, 'w') as f:
    json.dump(train_output, f, indent=2, ensure_ascii=False)

with open(test_path, 'w') as f:
    json.dump(test_output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved:")
print(f"  {train_path}")
print(f"  {test_path}")
