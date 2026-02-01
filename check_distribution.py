import json
from collections import Counter

print("=" * 70)
print("데이터셋 분포")
print("=" * 70)

# 1. 전처리 데이터 분포
print("\n📊 전처리 데이터 (data/processed/)")
print("-" * 70)
files = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
total = 0
for f in files:
    data = json.load(open(f'data/processed/{f}_processed.json', encoding='utf-8'))
    count = len(data)
    total += count
    pct = (count / 1293) * 100
    print(f"{f:10s}: {count:4d} ({pct:5.1f}%)")
print(f"{'Total':10s}: {total:4d} (100.0%)")

# 2. Split 데이터 분포
print("\n📊 Train/Valid/Test Split (data/final/)")
print("-" * 70)

splits = ['train', 'valid', 'test']
split_counts = {}

for split in splits:
    data = json.load(open(f'data/final/{split}.json', encoding='utf-8'))
    split_counts[split] = len(data)
    
    # 클래스별 분포
    labels = [s['label'] for s in data]
    label_dist = Counter(labels)
    
    total_split = len(data)
    print(f"\n{split.upper()}: {total_split} samples")
    for label in files:
        count = label_dist.get(label, 0)
        pct = (count / total_split * 100) if total_split > 0 else 0
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")

# 3. Split 비율
print("\n📊 Split 비율")
print("-" * 70)
total_split = sum(split_counts.values())
for split, count in split_counts.items():
    pct = (count / total_split) * 100
    print(f"{split:10s}: {count:4d} ({pct:5.1f}%)")
print(f"{'Total':10s}: {total_split:4d} (100.0%)")
