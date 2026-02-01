import json
from collections import Counter

print("=" * 70)
print("Judge 평가 상세 분석")
print("=" * 70)

# Judge 결과 로드
judge_data = json.load(open('data/final/judge_all_results.json', encoding='utf-8'))
summary = judge_data['summary']
results = judge_data['results']

# 전체 통계
print(f"\n📊 전체 정확도: {summary['overall_accuracy']:.2f}% ({summary['overall_correct']}/{summary['overall_total']})")

# 클래스별 상세
print("\n" + "=" * 70)
print("클래스별 Judge 평가 결과")
print("=" * 70)
print(f"{'Class':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Failed':<10}")
print("-" * 70)

for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
    stats = summary['by_class'][cls]
    failed = stats['total'] - stats['correct']
    print(f"{cls:<10} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:>6.2f}%      {failed:<10}")

# 실패 케이스 분석
print("\n" + "=" * 70)
print("실패 케이스 분석 (Judge가 틀린 경우)")
print("=" * 70)

for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
    class_results = results[cls]
    failures = [r for r in class_results if not r['correct']]
    
    if failures:
        print(f"\n{cls.upper()} 실패 케이스: {len(failures)}개")
        
        # 어떤 레이블로 잘못 예측했는지
        wrong_predictions = Counter([f['predicted'] for f in failures])
        print(f"  잘못 예측된 레이블:")
        for pred, count in wrong_predictions.most_common():
            print(f"    → {pred}: {count}개")
    else:
        print(f"\n{cls.upper()}: ✅ 실패 케이스 없음 (100%)")

# 혼동 행렬 시뮬레이션
print("\n" + "=" * 70)
print("Judge 혼동 패턴 (주요 오류)")
print("=" * 70)

confusion = {}
for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
    for result in results[cls]:
        if not result['correct']:
            key = f"{cls} → {result['predicted']}"
            confusion[key] = confusion.get(key, 0) + 1

if confusion:
    for pattern, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}개")
else:
    print("  오류 패턴 없음")
