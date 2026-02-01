import json
from collections import Counter

print("=" * 70)
print("원본 ESConv 데이터셋 Judge 평가 결과")
print("=" * 70)

# ESConv Judge 결과 로드
judge_data = json.load(open('data/pilot/judge_esconv_full_1300.json', encoding='utf-8'))

# 구조 확인
if isinstance(judge_data, dict) and 'summary' in judge_data:
    # 요약 정보가 있는 경우
    summary = judge_data['summary']
    results = judge_data.get('results', [])
    
    print(f"\n📊 전체 통계")
    print("-" * 70)
    if 'overall_accuracy' in summary:
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
        print(f"Total: {summary['overall_correct']}/{summary['overall_total']}")
    
    if 'by_class' in summary:
        print(f"\n📊 클래스별 정확도")
        print("-" * 70)
        print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
        print("-" * 70)
        
        for cls, stats in summary['by_class'].items():
            print(f"{cls:<15} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:>6.2f}%")

elif isinstance(judge_data, list):
    # 리스트 형식인 경우
    print(f"\n총 {len(judge_data)}개 세션 평가됨")
    
    # 예측된 레이블 분포
    predictions = [item.get('predicted', item.get('label', 'unknown')) for item in judge_data]
    pred_dist = Counter(predictions)
    
    print(f"\n📊 예측 레이블 분포")
    print("-" * 70)
    for label, count in sorted(pred_dist.items()):
        pct = count / len(judge_data) * 100
        print(f"{label:<15} {count:>5} ({pct:>5.1f}%)")
    
    # 첫 5개 샘플 확인
    print(f"\n샘플 데이터:")
    for i, item in enumerate(judge_data[:5], 1):
        print(f"\n{i}. Session ID: {item.get('session_id', 'unknown')}")
        print(f"   Predicted: {item.get('predicted', item.get('label', 'unknown'))}")
        if 'reasoning' in item:
            print(f"   Reasoning: {item['reasoning'][:100]}...")

else:
    print("데이터 형식을 인식할 수 없습니다.")
    print(f"Type: {type(judge_data)}")
    if isinstance(judge_data, dict):
        print(f"Keys: {list(judge_data.keys())[:10]}")
