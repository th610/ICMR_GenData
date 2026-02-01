import json

print("=" * 70)
print("ESConv 데이터 필터링 분석")
print("=" * 70)

# 전체 ESConv 로드
esconv = json.load(open('ESConv.json', encoding='utf-8'))
print(f"\n📊 전체 ESConv: {len(esconv)}개")

# 필터링 조건 확인
# judge_esconv_full.py의 extract_window 조건 재현
filtered = []
skip_reasons = {
    'too_short': 0,  # < 13 turns
    'no_supporter_end': 0,  # 마지막이 supporter가 아님
    'valid': 0
}

for session in esconv:
    dialog = session.get('dialog', [])
    
    # 조건 1: 최소 13턴
    if len(dialog) < 13:
        skip_reasons['too_short'] += 1
        continue
    
    # 조건 2: 13-20턴 중 마지막이 supporter인 경우가 있는지
    max_turns = min(20, len(dialog))
    has_valid = False
    
    for length in range(13, max_turns + 1):
        if dialog[length - 1]['speaker'] == 'supporter':
            has_valid = True
            break
    
    if has_valid:
        skip_reasons['valid'] += 1
        filtered.append(session)
    else:
        skip_reasons['no_supporter_end'] += 1

print(f"\n📋 필터링 결과:")
print(f"  ✅ 사용 가능: {skip_reasons['valid']}개")
print(f"  ❌ 너무 짧음 (<13턴): {skip_reasons['too_short']}개")
print(f"  ❌ Supporter로 끝나지 않음: {skip_reasons['no_supporter_end']}개")
print(f"\n  Total: {sum(skip_reasons.values())}개")

# Judge 결과 파일 확인
judge_file = 'data/pilot/judge_esconv_full_1300.json'
judge_data = json.load(open(judge_file, encoding='utf-8'))
print(f"\n📊 Judge 평가 결과: {len(judge_data)}개")

# 차이 확인
diff = 1300 - len(judge_data)
if diff > 0:
    print(f"\n⚠️  예상 1300개 vs 실제 {len(judge_data)}개")
    print(f"   차이: {diff}개 (필터링 또는 에러로 제외됨)")
