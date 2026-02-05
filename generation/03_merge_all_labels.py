"""
Merge all 6 labels: Normal + V1 + V2 + V3 + V4 + V5_new
Total: 1300 samples
"""
import json
from pathlib import Path

# Load all label files
files = {
    'Normal': 'generation/outputs/generated/generated_Normal.json',
    'V1': 'generation/outputs/generated/generated_V1.json',
    'V2': 'generation/outputs/generated/generated_V2.json',
    'V3': 'generation/outputs/generated/generated_V3.json',
    'V4': 'generation/outputs/generated/generated_V4.json',
    'V5': 'generation/outputs/generated/generated_V5_new.json',  # New V5
}

all_samples = []
counts = {}

print("="*70)
print("Merging all labels")
print("="*70)

for label, filepath in files.items():
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    counts[label] = len(samples)
    all_samples.extend(samples)
    print(f"  {label}: {len(samples)} samples")

print(f"\n✓ Total merged: {len(all_samples)} samples")

# Verify no duplicates
session_ids = [s['esconv_session_id'] for s in all_samples]
if len(session_ids) != len(set(session_ids)):
    print("⚠️  Warning: Duplicate session IDs found!")
else:
    print("✓ No duplicate session IDs")

# Save to data/final/
output_path = Path('data/final/all_labels_1300.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved: {output_path}")
print(f"\nLabel distribution:")
for label, count in counts.items():
    print(f"  {label}: {count}")
