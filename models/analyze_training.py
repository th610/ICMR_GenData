"""
학습된 모델 분석
- Epoch별 성능 추이
- 클래스별 성능
- 모델 정보
"""
import json
import re
from pathlib import Path

# 로그 파일 읽기
log_file = Path("models/training.log")
with open(log_file, 'r') as f:
    log_content = f.read()

# Epoch별 성능 추출
epoch_pattern = r"Epoch (\d+)/10"
valid_metrics_pattern = r"Loss: ([\d.]+)\nAccuracy: ([\d.]+)\nMacro Precision: ([\d.]+)\nMacro Recall: ([\d.]+)\nMacro F1: ([\d.]+)"

epochs = re.findall(epoch_pattern, log_content)
sections = re.split(epoch_pattern, log_content)

print("="*80)
print("학습 성능 추이 (Epoch별)")
print("="*80)
print(f"{'Epoch':<8} {'Loss':<10} {'Accuracy':<12} {'Macro P':<12} {'Macro R':<12} {'Macro F1':<12}")
print("-"*80)

epoch_data = []
for i in range(1, len(sections), 2):
    epoch_num = sections[i]
    epoch_content = sections[i+1]
    
    # Valid 성능 찾기
    valid_match = re.search(valid_metrics_pattern, epoch_content)
    if valid_match:
        loss, acc, prec, rec, f1 = valid_match.groups()
        epoch_data.append({
            'epoch': int(epoch_num),
            'loss': float(loss),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })
        print(f"{epoch_num:<8} {loss:<10} {acc:<12} {prec:<12} {rec:<12} {f1:<12}")

# 성능 개선 분석
print("\n" + "="*80)
print("성능 개선 분석")
print("="*80)

if len(epoch_data) > 1:
    print(f"\nEpoch 1 → Epoch {len(epoch_data)}:")
    print(f"  Accuracy: {epoch_data[0]['accuracy']:.4f} → {epoch_data[-1]['accuracy']:.4f} (+{epoch_data[-1]['accuracy'] - epoch_data[0]['accuracy']:.4f})")
    print(f"  Macro F1: {epoch_data[0]['f1']:.4f} → {epoch_data[-1]['f1']:.4f} (+{epoch_data[-1]['f1'] - epoch_data[0]['f1']:.4f})")
    print(f"  Loss:     {epoch_data[0]['loss']:.4f} → {epoch_data[-1]['loss']:.4f} ({epoch_data[-1]['loss'] - epoch_data[0]['loss']:.4f})")

# Best epoch 찾기
best_epoch = max(epoch_data, key=lambda x: x['f1'])
print(f"\nBest Epoch: {best_epoch['epoch']}")
print(f"  Macro F1: {best_epoch['f1']:.4f}")
print(f"  Accuracy: {best_epoch['accuracy']:.4f}")

# Test metrics 읽기
print("\n" + "="*80)
print("최종 Test Gold 성능")
print("="*80)

metrics_file = Path("models/outputs/test_metrics.json")
if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"\nBest Epoch: {metrics['best_epoch']}")
    print(f"Best Metric (Macro F1): {metrics['best_metric']:.4f}")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy:       {metrics['final_metrics']['accuracy']:.4f}")
    print(f"  Macro F1:       {metrics['final_metrics']['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['final_metrics']['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['final_metrics']['macro_recall']:.4f}")
    print(f"  V5 Recall:       {metrics['final_metrics']['v5_recall']:.4f} 🎯")
    
    print(f"\nPer-Class F1:")
    labels = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']
    for label, f1 in zip(labels, metrics['final_metrics']['per_class_f1']):
        print(f"  {label:<8}: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"{'True↓/Pred→':<12} {'Normal':<8} {'V1':<8} {'V2':<8} {'V3':<8} {'V4':<8} {'V5':<8}")
    print("-"*80)
    cm = metrics['final_metrics']['confusion_matrix']
    for i, row in enumerate(cm):
        print(f"{labels[i]:<12} {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8}")

# 모델 정보
print("\n" + "="*80)
print("모델 정보")
print("="*80)

model_file = Path("models/outputs/best_model.pt")
if model_file.exists():
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"\nModel File: {model_file}")
    print(f"Size: {size_mb:.1f} MB")
    
    import torch
    checkpoint = torch.load(model_file, map_location='cpu')
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Metric: {checkpoint['best_metric']:.4f}")
    if 'train_config' in checkpoint:
        cfg = checkpoint['train_config']
        print(f"\nTrain Config:")
        for key, val in cfg.items():
            print(f"  {key}: {val}")

# Early stopping 분석
print("\n" + "="*80)
print("Early Stopping 분석")
print("="*80)

if "Early stopping triggered" in log_content:
    match = re.search(r"Early stopping triggered at epoch (\d+)", log_content)
    if match:
        stopped_epoch = match.group(1)
        print(f"\n✅ Early stopping 작동: Epoch {stopped_epoch}")
        print(f"Patience: 3 (설정값)")
        print(f"Best epoch이 {best_epoch['epoch']}이므로, {int(stopped_epoch) - best_epoch['epoch']}번 연속 개선 없었음")

# 오버피팅 분석
print("\n" + "="*80)
print("오버피팅 분석")
print("="*80)

if epoch_data[-1]['accuracy'] == 1.0:
    print("\n⚠️  주의: Valid Accuracy 100%")
    print("가능한 원인:")
    print("  1. 데이터가 매우 명확하게 구분됨 (좋은 경우)")
    print("  2. Overfitting (문제 가능성)")
    print("  3. 데이터 누수 (가능성 낮음)")
    print("\n권장 조치:")
    print("  - 실제 새로운 데이터로 테스트")
    print("  - Cross-validation 수행")
    print("  - 더 어려운 샘플 추가")
else:
    print("\n✅ Valid Accuracy < 100%: 건강한 학습")

print("\n" + "="*80)
