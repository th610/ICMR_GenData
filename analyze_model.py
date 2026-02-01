import json
import numpy as np

print("=" * 70)
print("RoBERTa 모델 평가 결과 분석")
print("=" * 70)

# 모델 결과 로드
results = json.load(open('results/test_results.json', encoding='utf-8'))

# 전체 성능
print(f"\n📊 전체 성능")
print("-" * 70)
print(f"Accuracy:      {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
print(f"Macro F1:      {results['metrics']['f1_macro']:.4f}")

# 클래스별 성능
print("\n" + "=" * 70)
print("클래스별 성능")
print("=" * 70)
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 70)

classes = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
for cls in classes:
    stats = results['per_class'][cls]
    print(f"{cls:<10} {stats['precision']:<12.4f} {stats['recall']:<12.4f} {stats['f1']:<12.4f} {stats['support']:<10}")

# Confusion Matrix 분석
print("\n" + "=" * 70)
print("혼동 행렬 (Confusion Matrix)")
print("=" * 70)
print("True \\ Pred  ", end="")
for cls in classes:
    print(f"{cls:<8}", end="")
print()
print("-" * 70)

cm = results['confusion_matrix']
for i, true_cls in enumerate(classes):
    print(f"{true_cls:<13}", end="")
    for j in range(len(classes)):
        print(f"{cm[i][j]:<8}", end="")
    print()

# 오류 분석
print("\n" + "=" * 70)
print("주요 오류 패턴")
print("=" * 70)

errors = []
for i, true_cls in enumerate(classes):
    for j, pred_cls in enumerate(classes):
        if i != j and cm[i][j] > 0:
            errors.append((true_cls, pred_cls, cm[i][j]))

errors.sort(key=lambda x: x[2], reverse=True)
for true_cls, pred_cls, count in errors[:10]:
    print(f"  {true_cls} → {pred_cls}: {count}개")

# 클래스별 성공률
print("\n" + "=" * 70)
print("클래스별 정확도 (Recall)")
print("=" * 70)

for cls in classes:
    recall = results['per_class'][cls]['recall']
    support = results['per_class'][cls]['support']
    correct = int(recall * support)
    
    bar = "█" * int(recall * 20)
    print(f"{cls:<10} {recall*100:>6.2f}% [{bar:<20}] ({correct}/{support})")

# Judge vs 모델 비교
print("\n" + "=" * 70)
print("Judge vs RoBERTa 모델 비교")
print("=" * 70)

judge_data = json.load(open('data/final/judge_all_results.json', encoding='utf-8'))
judge_summary = judge_data['summary']

print(f"{'Class':<10} {'Judge Acc':<15} {'Model Recall':<15} {'차이':<10}")
print("-" * 70)

for cls in classes:
    judge_acc = judge_summary['by_class'][cls]['accuracy']
    model_recall = results['per_class'][cls]['recall'] * 100
    diff = model_recall - judge_acc
    
    symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"{cls:<10} {judge_acc:>6.2f}%        {model_recall:>6.2f}%        {symbol} {abs(diff):>5.2f}%")

print(f"\n{'Overall':<10} {judge_summary['overall_accuracy']:>6.2f}%        {results['metrics']['accuracy']*100:>6.2f}%        {'↓' if results['metrics']['accuracy']*100 < judge_summary['overall_accuracy'] else '↑'} {abs(results['metrics']['accuracy']*100 - judge_summary['overall_accuracy']):>5.2f}%")
