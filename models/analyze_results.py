"""
학습 결과 분석 스크립트
"""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 1. Test metrics 로드
print("="*80)
print("최종 Test Gold 성능")
print("="*80)

with open('models/outputs/test_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"\nBest Epoch: {metrics['best_epoch']}")
print(f"Best Metric (Macro F1): {metrics['best_metric']:.4f}")
print(f"\nFinal Test Results:")
print(f"  Accuracy: {metrics['final_metrics']['accuracy']:.4f}")
print(f"  Macro F1: {metrics['final_metrics']['macro_f1']:.4f}")
print(f"  Macro Precision: {metrics['final_metrics']['macro_precision']:.4f}")
print(f"  Macro Recall: {metrics['final_metrics']['macro_recall']:.4f}")
print(f"  V5 Recall: {metrics['final_metrics']['v5_recall']:.4f}")

print(f"\nPer-class F1 scores:")
labels = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']
for i, (label, f1) in enumerate(zip(labels, metrics['final_metrics']['per_class_f1'])):
    print(f"  {label}: {f1:.4f}")

# 2. Confusion Matrix
print(f"\n{'='*80}")
print("Confusion Matrix (Test Gold)")
print(f"{'='*80}")
cm = np.array(metrics['final_metrics']['confusion_matrix'])
print("\nPredicted →")
print(f"True ↓    {'  '.join([f'{l:>8}' for l in labels])}")
for i, label in enumerate(labels):
    row = '  '.join([f'{v:>8}' for v in cm[i]])
    print(f"{label:>8}  {row}")

# 3. 학습 로그 파싱
print(f"\n{'='*80}")
print("학습 곡선 분석")
print(f"{'='*80}")

with open('models/training.log', 'r') as f:
    log_content = f.read()

# Extract epoch results
epoch_pattern = r'Epoch (\d+)/10'
train_loss_pattern = r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)'
valid_acc_pattern = r'Accuracy: ([\d.]+)'
valid_f1_pattern = r'Macro F1: ([\d.]+)'
v5_recall_pattern = r'🎯 V5 Recall \(Key Metric\): ([\d.]+)'

epochs = []
train_losses = []
train_accs = []
valid_accs = []
valid_f1s = []
v5_recalls = []

# Split by epoch
epoch_sections = log_content.split('Epoch ')
for section in epoch_sections[1:]:  # Skip first empty section
    # Extract epoch number
    epoch_match = re.search(r'^(\d+)/10', section)
    if not epoch_match:
        continue
    epoch = int(epoch_match.group(1))
    
    # Extract metrics
    train_loss_match = re.search(train_loss_pattern, section)
    valid_acc_match = re.search(valid_acc_pattern, section)
    valid_f1_match = re.search(valid_f1_pattern, section)
    v5_recall_match = re.search(v5_recall_pattern, section)
    
    if train_loss_match and valid_f1_match:
        epochs.append(epoch)
        train_losses.append(float(train_loss_match.group(1)))
        train_accs.append(float(train_loss_match.group(2)))
        valid_accs.append(float(valid_acc_match.group(1)))
        valid_f1s.append(float(valid_f1_match.group(1)))
        v5_recalls.append(float(v5_recall_match.group(1)))

print(f"\n학습 진행 (Epochs 1-{len(epochs)}):")
print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Valid Acc':<12} {'Valid F1':<12} {'V5 Recall':<12}")
print("-" * 80)
for i in range(len(epochs)):
    print(f"{epochs[i]:<8} {train_losses[i]:<12.4f} {train_accs[i]:<12.4f} {valid_accs[i]:<12.4f} {valid_f1s[i]:<12.4f} {v5_recalls[i]:<12.4f}")

# 4. 주요 발견사항
print(f"\n{'='*80}")
print("주요 발견사항")
print(f"{'='*80}")

print(f"\n1. 학습 속도:")
print(f"   - Epoch 1: Macro F1 {valid_f1s[0]:.4f} (모든 샘플을 Normal로 예측)")
print(f"   - Epoch 2: Macro F1 {valid_f1s[1]:.4f} (급격한 향상!)")
print(f"   - Epoch 4: Macro F1 {valid_f1s[3]:.4f} (최고 성능)")

print(f"\n2. Overfitting 분석:")
epoch_4_idx = 3
if epoch_4_idx < len(epochs):
    print(f"   - Epoch 4 Train Acc: {train_accs[epoch_4_idx]:.4f}")
    print(f"   - Epoch 4 Valid Acc: {valid_accs[epoch_4_idx]:.4f}")
    print(f"   - Train-Valid Gap: {abs(train_accs[epoch_4_idx] - valid_accs[epoch_4_idx]):.4f}")
    
    if len(epochs) >= 7:
        print(f"   - Epoch 7 Train Acc: {train_accs[6]:.4f}")
        print(f"   - Epoch 7 Valid Acc: {valid_accs[6]:.4f}")
        print(f"   - Train Loss Epoch 7: {train_losses[6]:.4f} (거의 0)")

print(f"\n3. V5 Recall 추이:")
for i in range(len(epochs)):
    if i == 0:
        print(f"   - Epoch {epochs[i]}: {v5_recalls[i]:.4f} (실패)")
    elif i == 1:
        print(f"   - Epoch {epochs[i]}: {v5_recalls[i]:.4f} (큰 향상!)")
    elif v5_recalls[i] == 1.0:
        print(f"   - Epoch {epochs[i]}: {v5_recalls[i]:.4f} (완벽)")

print(f"\n4. Early Stopping:")
print(f"   - Best Epoch: 4")
print(f"   - Stopped at: Epoch 7")
print(f"   - Patience: 3 epochs")
print(f"   - Reason: Valid F1이 Epoch 4 이후 개선되지 않음")

print(f"\n5. 최종 평가:")
print(f"   - Test Gold (300개) Accuracy: 100%")
print(f"   - 모든 클래스 완벽 분류")
print(f"   - 오분류 케이스: 0개")

print(f"\n{'='*80}")
print("결론")
print(f"{'='*80}")
print("""
✅ 긍정적 측면:
   - V5 Recall 목표 달성 (100%)
   - 모든 클래스에서 완벽한 성능
   - Early stopping 정상 작동
   - 학습 속도 매우 빠름 (Epoch 2에서 93% F1)

⚠️  주의 사항:
   - Valid와 Test 모두 100% → 의심스러움
   - Epoch 4 이후 train loss가 거의 0 → overfitting 가능성
   - 데이터가 너무 잘 분리되어 있거나 패턴이 너무 명확할 수 있음
   
🔍 권장 사항:
   1. 실제 운영 데이터로 추가 검증 필요
   2. Cross-validation으로 재검증
   3. 새로운 세션 데이터로 테스트
   4. 오분류 케이스 분석 (현재 0개)
""")
