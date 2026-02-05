import json

v4_full_path = ".archive/2026-02-02_pilot/pilot/v4_full_150.json"
v5_full_path = ".archive/2026-02-02_pilot/pilot/v5_full_150.json"

v4_full = json.load(open(v4_full_path, encoding='utf-8'))
v5_full = json.load(open(v5_full_path, encoding='utf-8'))

print(f"v4_full_150.json: {len(v4_full)}개")
print(f"v5_full_150.json: {len(v5_full)}개")
print()

# 구조 확인
r4 = v4_full[0]
print("V4 샘플 구조:")
print(f"  primary_label: {r4.get('primary_label', '없음')}")
print(f"  dialog 길이: {len(r4['dialog'])}턴")
print(f"  generation_method: {r4.get('generation_method', '없음')}")
print(f"  prefix_length: {r4.get('prefix_length', '없음')}")
print(f"  insertion_length: {r4.get('insertion_length', '없음')}")
print(f"  violation_turn_index: {r4.get('violation_turn_index', '없음')}")
print()

r5 = v5_full[0]
print("V5 샘플 구조:")
print(f"  primary_label: {r5.get('primary_label', '없음')}")
print(f"  dialog 길이: {len(r5['dialog'])}턴")
print(f"  generation_method: {r5.get('generation_method', '없음')}")
print(f"  prefix_length: {r5.get('prefix_length', '없음')}")
print(f"  insertion_length: {r5.get('insertion_length', '없음')}")
print(f"  violation_turn_index: {r5.get('violation_turn_index', '없음')}")
