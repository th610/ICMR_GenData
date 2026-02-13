"""
Table 1: RoBERTa Classifier Detection Performance (Gold Test Set)

학습된 RoBERTa 분류기의 Gold 테스트 셋(300개) 직접 분류 성능.
- 모델 학습 시 저장된 test_metrics.json 확인
- 모델을 직접 실행하여 Gold 텍스트 분류 결과 재검증
- Per-Class Precision, Recall, F1, Confusion Matrix

NOTE: 이 테이블의 데이터 원천은 test_gold_300_prefix.json의 **원본 Gold 텍스트**
      (generated_dialog 내 위반/정상 응답)를 분류기에 직접 넣은 결과입니다.
      run_full_experiment.py의 GPT 생성 응답 분류가 아님.

Input:
    models/outputs/test_metrics.json   (학습 시 저장된 메트릭)
    models/outputs/best_model.pt       (직접 재검증용)
    test_gold_300_prefix.json          (테스트 데이터)

Output:
    table1/
        table1_report.txt
        table1_metrics.json
"""
import json
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent
MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
DATA_PATH = Path(__file__).parent.parent / "data" / "final" / "test_gold_300.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"
OUTPUT_DIR = RESULTS_DIR / "table1"

LABELS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
LABEL_DISPLAY = {'normal': 'Normal', 'v1': 'V1', 'v2': 'V2',
                 'v3': 'V3', 'v4': 'V4', 'v5': 'V5'}


def load_training_metrics():
    """학습 시 저장된 test_metrics.json 로드"""
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_data():
    """Gold 테스트 데이터 로드"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples']


def run_direct_verification(samples):
    """
    모델로 직접 Gold 텍스트를 분류하여 재검증.
    학습 시와 동일한 방식(ViolationDataset)으로 평가.
    """
    import sys
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
    from data_utils import ViolationDataset, DataConfig
    from transformers import RobertaTokenizer
    from model import ViolationClassifier
    
    # 데이터 준비
    test_data = {'samples': samples}
    
    # 토크나이저 로드
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.add_tokens([DataConfig.SEEKER_TOKEN, DataConfig.SUPPORTER_TOKEN, DataConfig.SUPPORTER_TARGET_TOKEN])
    
    # 데이터셋 생성
    dataset = ViolationDataset(test_data, tokenizer, max_length=512, is_test=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViolationClassifier('roberta-base', num_labels=6)
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    model_file = MODEL_DIR / "best_model_v3.pt"
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"[OK] Model loaded from {model_file} (device: {device})")
    
    # 추론
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in loader:
            t0 = time.time()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            latency_ms = (time.time() - t0) * 1000 / len(ids)  # per sample
            
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch['label'].tolist()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            latencies.extend([latency_ms] * len(ids))
    
    # 결과 포맷팅
    results = []
    LABEL_NAMES = DataConfig.LABELS
    for i, (pred, gold) in enumerate(zip(all_preds, all_labels)):
        results.append({
            'index': i,
            'gold': LABEL_NAMES[gold].lower(),
            'pred': LABEL_NAMES[pred].lower(),
            'confidence': 1.0  # argmax 사용이므로 confidence는 고정
        })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            correct = sum(1 for r in results if r['gold'] == r['pred'])
            print(f"  [{i+1:3d}/{len(samples)}] acc={correct/(i+1):.4f}")

    return results, latencies


def compute_metrics(results):
    """Per-Class Precision, Recall, F1 + Macro + Accuracy"""
    cm = defaultdict(lambda: defaultdict(int))
    for r in results:
        cm[r['gold']][r['pred']] += 1

    per_class = {}
    for label in LABELS:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in LABELS if other != label)
        fn = sum(cm[label][other] for other in LABELS if other != label)
        support = sum(cm[label][pred] for pred in LABELS)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'support': support, 'tp': tp, 'fp': fp, 'fn': fn
        }

    total = len(results)
    correct = sum(per_class[l]['tp'] for l in LABELS)
    accuracy = correct / total if total > 0 else 0.0

    macro_p = sum(m['precision'] for m in per_class.values()) / len(LABELS)
    macro_r = sum(m['recall'] for m in per_class.values()) / len(LABELS)
    macro_f1 = sum(m['f1'] for m in per_class.values()) / len(LABELS)

    cm_matrix = [[cm[gold][pred] for pred in LABELS] for gold in LABELS]

    return {
        'per_class': per_class,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'total': total,
        'confusion_matrix': cm_matrix
    }


def format_report(m, latency_stats=None, training_metrics=None):
    lines = []
    pc = m['per_class']

    lines.append("")
    lines.append("=" * 80)
    lines.append("  TABLE 1: RoBERTa Classifier Detection Performance (Gold Test Set)")
    lines.append("=" * 80)
    lines.append("")

    if training_metrics:
        fm = training_metrics['final_metrics']
        lines.append("  [A] Training Result (models/outputs/test_metrics.json)")
        lines.append(f"      Best Epoch:      {training_metrics['best_epoch']}")
        lines.append(f"      Test Accuracy:   {fm['accuracy']:.4f}")
        lines.append(f"      Test Macro F1:   {fm['macro_f1']:.4f}")
        lines.append(f"      Test Loss:       {fm['loss']:.6f}")
        lines.append("")

    lines.append("  [B] Direct Verification")
    lines.append("")

    header = f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}"
    lines.append(header)
    lines.append("  " + "-" * 52)

    for label in LABELS:
        p = pc[label]
        lines.append(f"  {LABEL_DISPLAY[label]:<10}"
                      f" {p['precision']:>10.4f} {p['recall']:>10.4f}"
                      f" {p['f1']:>10.4f} {p['support']:>8d}")

    lines.append("  " + "-" * 52)
    lines.append(f"  {'Macro':<10}"
                  f" {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f}"
                  f" {m['macro_f1']:>10.4f} {m['total']:>8d}")
    lines.append("")
    lines.append(f"  Accuracy: {m['accuracy']:.4f}  ({m['total']} samples)")
    lines.append("")

    if latency_stats:
        lines.append(f"  Detection Latency (ms):")
        lines.append(f"    Mean:   {latency_stats['mean']:.1f}")
        lines.append(f"    Median: {latency_stats['median']:.1f}")
        lines.append(f"    P95:    {latency_stats['p95']:.1f}")
        lines.append("")

    lines.append("  Confusion Matrix (Gold ↓ / Pred →)")
    hdr = f"  {'':>8}"
    for l in LABELS:
        hdr += f" {LABEL_DISPLAY[l]:>6}"
    lines.append(hdr)
    lines.append("  " + "-" * 50)
    for i, gold in enumerate(LABELS):
        row = f"  {LABEL_DISPLAY[gold]:>8}"
        for j in range(len(LABELS)):
            row += f" {m['confusion_matrix'][i][j]:>6}"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Table 1: Detection Performance")
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip direct verification, use test_metrics.json only')
    args = parser.parse_args()

    # 1. Load training metrics
    training_metrics = None
    if METRICS_PATH.exists():
        training_metrics = load_training_metrics()
        fm = training_metrics['final_metrics']
        print(f"[Training] Accuracy={fm['accuracy']:.4f}, "
              f"Macro F1={fm['macro_f1']:.4f}, Epoch={training_metrics['best_epoch']}")

    # 2. Direct verification
    latency_stats = None
    if args.skip_verify and training_metrics:
        print("[SKIP] Using training metrics only")
        cm = training_metrics['final_metrics']['confusion_matrix']
        results = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                for _ in range(cm[i][j]):
                    results.append({'gold': LABELS[i], 'pred': LABELS[j], 'confidence': 1.0, 'index': len(results)})
        metrics = compute_metrics(results)
    else:
        print(f"\n[VERIFY] Running model on Gold test data...")
        samples = load_test_data()
        print(f"  Loaded {len(samples)} samples")

        results, latencies = run_direct_verification(samples)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        latency_stats = {
            'mean': sum(latencies) / n,
            'median': latencies_sorted[n // 2],
            'p95': latencies_sorted[int(n * 0.95)],
            'min': latencies_sorted[0],
            'max': latencies_sorted[-1]
        }
        metrics = compute_metrics(results)
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")

    # 3. Report
    report = format_report(metrics, latency_stats, training_metrics)
    print(report)

    # 4. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "table1_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    save_metrics = {
        'source': 'direct_verification' if not args.skip_verify else 'test_metrics.json',
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'per_class': {label: {
            'precision': m['precision'], 'recall': m['recall'],
            'f1': m['f1'], 'support': m['support']
        } for label, m in metrics['per_class'].items()},
        'confusion_matrix': metrics['confusion_matrix'],
        'total_samples': metrics['total']
    }
    if latency_stats:
        save_metrics['latency_ms'] = latency_stats
    if training_metrics:
        save_metrics['training_result'] = {
            'best_epoch': training_metrics['best_epoch'],
            'accuracy': training_metrics['final_metrics']['accuracy'],
            'macro_f1': training_metrics['final_metrics']['macro_f1']
        }

    with open(OUTPUT_DIR / "table1_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
