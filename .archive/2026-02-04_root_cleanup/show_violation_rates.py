"""
V3 vs V4 위반율 표 생성
"""
import json

# Load data
with open('test_judge_v3_100.json', 'r', encoding='utf-8') as f:
    v3_results = json.load(f)

with open('test_judge_v4_100.json', 'r', encoding='utf-8') as f:
    v4_results = json.load(f)

# Count labels
v3_counts = {}
v4_counts = {}

for r in v3_results:
    label = r['label']
    v3_counts[label] = v3_counts.get(label, 0) + 1

for r in v4_results:
    label = r['label']
    v4_counts[label] = v4_counts.get(label, 0) + 1

# Print table
print("=" * 80)
print("V3 vs V4 위반율 비교 (전체 100개 샘플)")
print("=" * 80)
print()
print(f"{'Label':<15} {'V3 Count':<15} {'V3 Rate':<15} {'V4 Count':<15} {'V4 Rate':<15}")
print("-" * 80)

total = 100
for label in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
    v3_cnt = v3_counts.get(label, 0)
    v4_cnt = v4_counts.get(label, 0)
    v3_pct = f"{v3_cnt/total*100:.1f}%"
    v4_pct = f"{v4_cnt/total*100:.1f}%"
    
    print(f"{label.upper():<15} {v3_cnt}개{'':<12} {v3_pct:<15} {v4_cnt}개{'':<12} {v4_pct:<15}")

print("-" * 80)

# Total violations
v3_total_viol = total - v3_counts.get('normal', 0)
v4_total_viol = total - v4_counts.get('normal', 0)

print(f"{'TOTAL VIOLATIONS':<15} {v3_total_viol}개{'':<12} {v3_total_viol}%{'':<13} {v4_total_viol}개{'':<12} {v4_total_viol}%")
print("=" * 80)
