"""Check turn count distribution across all classes."""
import json
import numpy as np
from collections import Counter

# Load all data
file_map = {
    'normal': 'data/generated/normal_400.json',
    'v1': 'data/generated/v1_240.json',
    'v2': 'data/generated/v2_160.json',
    'v3': 'data/generated/v3_200.json',
    'v4': 'data/pilot/v4_full_150.json',
    'v5': 'data/pilot/v5_full_150.json',
}

classes = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
data = {}

for cls in classes:
    with open(file_map[cls], encoding='utf-8') as f:
        data[cls] = json.load(f)

print("\n=== 턴 수 분포 분석 ===\n")

for cls in classes:
    turns = [len(s['dialog']) for s in data[cls]]
    print(f"{cls.upper()}:")
    print(f"  평균: {np.mean(turns):.1f}턴")
    print(f"  범위: {np.min(turns)}-{np.max(turns)}턴")
    print(f"  중앙값: {np.median(turns):.1f}턴")
    
    # Distribution
    counter = Counter(turns)
    print(f"  분포: ", end="")
    for k in sorted(counter.keys())[:5]:  # First 5
        print(f"{k}턴({counter[k]}), ", end="")
    print("...")
    print()

# Compare with token lengths
print("\n=== 토큰 길이 vs 턴 수 비교 ===\n")

# Load token analysis results
with open('data/final/judge_all_results.json', encoding='utf-8') as f:
    judge_results = json.load(f)

# Calculate avg tokens per class
for cls in classes:
    sessions = data[cls]
    avg_turns = np.mean([len(s['dialog']) for s in sessions])
    
    # Get avg tokens from judge results
    cls_sessions = [s for s in judge_results if s['session_id'].startswith(cls)]
    if cls_sessions:
        # Approximate token count (you can calculate actual if needed)
        print(f"{cls.upper()}: 평균 {avg_turns:.1f}턴")
