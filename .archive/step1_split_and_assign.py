"""
Step 1: ESConv 1300 Prefixes 분리 및 라벨 배정

Gold 300:
- Normal 80, V1 50, V2 60, V3 50, V4 30, V5 30

Train/Val 1000:
- Normal 290, V1 170, V2 170, V3 170, V4 100, V5 100
"""
import json
import random

# 재현성을 위한 시드 고정
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print("="*80)
print("Step 1: 데이터 분리 및 라벨 배정")
print("="*80)

# 데이터 로드
print("\nLoading data...")
all_prefixes = json.load(open('ESConv_1300_prefixes.json', encoding='utf-8'))
v5_prefixes = json.load(open('ESConv_v5_prefixes.json', encoding='utf-8'))

# V5 트리거 세션 ID 집합
v5_session_ids = {p['esconv_session_id'] for p in v5_prefixes}
print(f"V5 trigger sessions: {len(v5_session_ids)}개")

# 일반 세션 ID 집합
normal_session_ids = [p['esconv_session_id'] for p in all_prefixes 
                      if p['esconv_session_id'] not in v5_session_ids]
print(f"Normal sessions: {len(normal_session_ids)}개")

# ============================================================================
# 1단계: V5 75개 분배 (Gold 30 + TrainVal 45)
# ============================================================================
v5_list = list(v5_session_ids)
random.shuffle(v5_list)

gold_v5_ids = set(v5_list[:30])
trainval_v5_ids = set(v5_list[30:75])

print(f"\nV5 분배:")
print(f"  Gold: {len(gold_v5_ids)}개")
print(f"  Train/Val: {len(trainval_v5_ids)}개")

# ============================================================================
# 2단계: 일반 세션 분배 (Gold 270 + TrainVal 955)
# ============================================================================
random.shuffle(normal_session_ids)

gold_normal_ids = set(normal_session_ids[:270])  # 270 = 300 - 30
trainval_normal_ids = set(normal_session_ids[270:1225])  # 955 = 1000 - 45

print(f"\n일반 세션 분배:")
print(f"  Gold: {len(gold_normal_ids)}개")
print(f"  Train/Val: {len(trainval_normal_ids)}개")

# ============================================================================
# 3단계: Gold 300 라벨 배정
# ============================================================================
gold_label_distribution = {
    'Normal': 80,
    'V1': 50,
    'V2': 60,
    'V3': 50,
    'V4': 30,
    'V5': 30
}

# Gold 세션 ID 통합 (V5 30 + Normal 270)
gold_all_ids = list(gold_v5_ids) + list(gold_normal_ids)
random.shuffle(gold_all_ids)

gold_assigned = []
label_idx = 0

for label, count in gold_label_distribution.items():
    for _ in range(count):
        session_id = gold_all_ids[label_idx]
        
        # 원본 prefix 찾기
        prefix = next(p for p in all_prefixes if p['esconv_session_id'] == session_id)
        
        # assigned_label과 has_v5_trigger 추가
        assigned_item = prefix.copy()
        assigned_item['assigned_label'] = label
        assigned_item['has_v5_trigger'] = session_id in v5_session_ids
        
        gold_assigned.append(assigned_item)
        label_idx += 1

# ============================================================================
# 4단계: Train/Val 1000 라벨 배정
# ============================================================================
trainval_label_distribution = {
    'Normal': 290,
    'V1': 170,
    'V2': 170,
    'V3': 170,
    'V4': 100,
    'V5': 100
}

# Train/Val 세션 ID 통합 (V5 45 + Normal 955)
trainval_all_ids = list(trainval_v5_ids) + list(trainval_normal_ids)
random.shuffle(trainval_all_ids)

trainval_assigned = []
label_idx = 0

for label, count in trainval_label_distribution.items():
    for _ in range(count):
        session_id = trainval_all_ids[label_idx]
        
        # 원본 prefix 찾기
        prefix = next(p for p in all_prefixes if p['esconv_session_id'] == session_id)
        
        # assigned_label과 has_v5_trigger 추가
        assigned_item = prefix.copy()
        assigned_item['assigned_label'] = label
        assigned_item['has_v5_trigger'] = session_id in v5_session_ids
        
        trainval_assigned.append(assigned_item)
        label_idx += 1

# ============================================================================
# 5단계: 저장
# ============================================================================
with open('gold_300_assigned.json', 'w', encoding='utf-8') as f:
    json.dump(gold_assigned, f, ensure_ascii=False, indent=2)

with open('trainval_1000_assigned.json', 'w', encoding='utf-8') as f:
    json.dump(trainval_assigned, f, ensure_ascii=False, indent=2)

# ============================================================================
# 6단계: 검증
# ============================================================================
print("\n" + "="*80)
print("검증 결과")
print("="*80)

# Gold 검증
print(f"\n[Gold 300]")
print(f"Total: {len(gold_assigned)}개")
gold_label_count = {}
gold_v5_trigger_count = sum(1 for item in gold_assigned if item['has_v5_trigger'])
for item in gold_assigned:
    label = item['assigned_label']
    gold_label_count[label] = gold_label_count.get(label, 0) + 1

for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
    count = gold_label_count.get(label, 0)
    expected = gold_label_distribution[label]
    status = "✓" if count == expected else "✗"
    print(f"  {label}: {count}/{expected} {status}")
print(f"  V5 triggers: {gold_v5_trigger_count}개")

# Train/Val 검증
print(f"\n[Train/Val 1000]")
print(f"Total: {len(trainval_assigned)}개")
trainval_label_count = {}
trainval_v5_trigger_count = sum(1 for item in trainval_assigned if item['has_v5_trigger'])
for item in trainval_assigned:
    label = item['assigned_label']
    trainval_label_count[label] = trainval_label_count.get(label, 0) + 1

for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
    count = trainval_label_count.get(label, 0)
    expected = trainval_label_distribution[label]
    status = "✓" if count == expected else "✗"
    print(f"  {label}: {count}/{expected} {status}")
print(f"  V5 triggers: {trainval_v5_trigger_count}개")

# 누수 검증
gold_ids = {item['esconv_session_id'] for item in gold_assigned}
trainval_ids = {item['esconv_session_id'] for item in trainval_assigned}
overlap = gold_ids & trainval_ids
print(f"\n[누수 검증]")
print(f"  Gold session IDs: {len(gold_ids)}개")
print(f"  Train/Val session IDs: {len(trainval_ids)}개")
print(f"  Overlap: {len(overlap)}개 {'✓' if len(overlap) == 0 else '✗ LEAKAGE!'}")
print(f"  Total unique: {len(gold_ids | trainval_ids)}개 (should be 1300)")

print("\n" + "="*80)
print("✅ 완료!")
print("="*80)
print(f"생성된 파일:")
print(f"  - gold_300_assigned.json")
print(f"  - trainval_1000_assigned.json")
