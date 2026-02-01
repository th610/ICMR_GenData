import json

print("=" * 70)
print("ESConv 사용 분석")
print("=" * 70)

# 원본 ESConv
esconv = json.load(open('ESConv.json', encoding='utf-8'))
print(f"\n📊 원본 ESConv: {len(esconv)}개")

# 우리가 생성한 데이터
print(f"\n📊 우리 생성 데이터에서 ESConv 사용:")

files = {
    'normal': 'data/generated/normal_400.json',
    'v1': 'data/generated/v1_240.json', 
    'v2': 'data/generated/v2_160.json',
    'v3': 'data/generated/v3_200.json',
}

total_used = 0
for label, path in files.items():
    data = json.load(open(path, encoding='utf-8'))
    count = len(data)
    total_used += count
    print(f"  {label}: {count}개")

print(f"\n  ESConv 사용 추정: {total_used}개")
print(f"  미사용: {len(esconv) - total_used}개 ({(len(esconv) - total_used) / len(esconv) * 100:.1f}%)")

print(f"\n💡 분석:")
print(f"  - Normal 400개: ESConv에서 랜덤 400개 샘플링")
print(f"  - V1-V3 593개: ESConv에서 각각 다른 세션 prefix 사용")
print(f"  - V4-V5 300개: ESConv 사용 안 함 (full multiturn 생성)")
print(f"\n  → 전체 1300개 중 ~993개 활용")
print(f"  → 나머지 ~307개는 랜덤 샘플링에서 선택 안 됨")
