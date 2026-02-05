import json

# V1 데이터 로드
v1 = json.load(open('.archive/2026-02-02_generated/generated/v1_240.json', encoding='utf-8'))
v2 = json.load(open('.archive/2026-02-02_generated/generated/v2_160.json', encoding='utf-8'))
v3 = json.load(open('.archive/2026-02-02_generated/generated/v3_200.json', encoding='utf-8'))
v4 = json.load(open('.archive/2026-02-02_pilot/pilot/v4.json', encoding='utf-8'))
v5 = json.load(open('.archive/2026-02-02_pilot/pilot/v5.json', encoding='utf-8'))
normal = json.load(open('.archive/2026-02-02_generated/generated/normal_400.json', encoding='utf-8'))

print('=' * 60)
print('이전 생성 데이터 구조 분석')
print('=' * 60)
print()

print(f'V1 데이터: {len(v1)}개')
print(f'V2 데이터: {len(v2)}개')
print(f'V3 데이터: {len(v3)}개')
print(f'V4 데이터: {len(v4)}개')
print(f'V5 데이터: {len(v5)}개')
print(f'Normal 데이터: {len(normal)}개')
print(f'총 데이터: {len(v1)+len(v2)+len(v3)+len(v4)+len(v5)+len(normal)}개')
print()

# 첫 번째 V1 샘플 상세 분석
r = v1[0]
print('=' * 60)
print('V1 샘플 구조 (첫 번째):')
print('=' * 60)
print(f'  primary_label: {r["primary_label"]}')
print(f'  situation: {r["situation"][:80]}...')
print(f'  dialog 길이: {len(r["dialog"])}턴')
print(f'  generation_method: {r.get("generation_method", "없음")}')
print(f'  prefix_length: {r.get("prefix_length", "없음")}')
print(f'  insertion_length: {r.get("insertion_length", "없음")}')
print(f'  violation_turn_index: {r.get("violation_turn_index", "없음")}')
print()

print('Dialog 구조:')
prefix_len = r.get("prefix_length", 0)
insert_len = r.get("insertion_length", 0)
print(f'  Prefix (ESConv 원본): 0~{prefix_len-1}턴')
print(f'  Insertion (생성된 부분): {prefix_len}~{prefix_len+insert_len-1}턴')
print(f'  위반 턴 인덱스: {r.get("violation_turn_index", "없음")}턴 (마지막 턴)')
print()

# 마지막 5턴 확인
print('마지막 5턴:')
for i, turn in enumerate(r["dialog"][-5:], len(r["dialog"])-4):
    print(f'  [{i}] {turn["speaker"]}: {turn["content"][:80]}...')
print()

# V2 샘플도 확인
r2 = v2[0]
print('=' * 60)
print('V2 샘플 구조 (첫 번째):')
print('=' * 60)
print(f'  primary_label: {r2["primary_label"]}')
print(f'  dialog 길이: {len(r2["dialog"])}턴')
print(f'  prefix_length: {r2.get("prefix_length", "없음")}')
print(f'  insertion_length: {r2.get("insertion_length", "없음")}')
print(f'  violation_turn_index: {r2.get("violation_turn_index", "없음")}')
print()

print('마지막 턴:')
last = r2["dialog"][-1]
print(f'  {last["speaker"]}: {last["content"][:150]}')
