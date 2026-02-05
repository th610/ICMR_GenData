import json

# V10 평가 결과 로드
results = json.load(open('test_v10_synthetic_sample_results.json', encoding='utf-8'))

# V3 결과만 필터링
v3_results = [r for r in results if r['label'] == 'V3']

print('='*70)
print('V3 평가 결과 분석')
print('='*70)
print(f'총 V3 샘플: {len(v3_results)}개')
print(f'정답: {sum(1 for r in v3_results if r["is_correct"])}개')
print(f'오답: {sum(1 for r in v3_results if not r["is_correct"])}개')
print()

# 예측 분포
from collections import Counter
predictions = [r['prediction'] for r in v3_results]
pred_counts = Counter(predictions)

print('예측 분포:')
for pred, count in pred_counts.items():
    print(f'  {pred}: {count}개')
print()

# 오답 케이스 상세
print('='*70)
print('V3 → Normal 오판 케이스')
print('='*70)

wrong_cases = [r for r in v3_results if not r['is_correct']]

for i, case in enumerate(wrong_cases, 1):
    print(f'\n[케이스 {i}]')
    print(f'  generation_method: {case["generation_method"]}')
    print(f'  dialog_length: {case["dialog_length"]}턴')
    print(f'  Ground Truth: {case["ground_truth"]}')
    print(f'  Prediction: {case["prediction"]}')
