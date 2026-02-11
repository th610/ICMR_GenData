"""
Table 2: Model vs LLM-Judge Classification Comparison

동일한 300개 테스트 샘플(Gold 텍스트)에 대해:
  - RoBERTa 분류기 (Detection Model) 직접 분류
  - GPT-4o-mini Judge (Classify-Only)
를 비교.

Metrics:
  - Accuracy, Macro F1, Per-Class F1
  - FPR (False Positive Rate): Normal을 Violation으로 오탐한 비율
  - FNR (False Negative Rate): Violation을 Normal로 놓친 비율
  - Agreement, Cohen's Kappa, Latency

Input:
    test_gold_300_prefix.json (Gold 테스트 데이터)

Output:
    table2/
        table2_report.txt
        table2_metrics.json
        judge_log.jsonl
"""
import json
import sys
import time
import os
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts_judge_classify import (
    JUDGE_CLASSIFY_SYSTEM,
    build_classify_prompt,
    JUDGE_CLASSIFY_RETRY,
)
from src.agent.step3_violation_detector import ViolationDetector

# ── Config ──────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
DATA_PATH = Path(__file__).parent / "test_gold_300_prefix.json"
OUTPUT_DIR = Path(__file__).parent / "table2"

LABELS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
LABEL_MAP = {l.upper(): l for l in LABELS}
LABEL_MAP.update({l: l for l in LABELS})


def normalize_label(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in LABELS:
        return raw
    for label in LABELS:
        if raw == label:
            return label
    return raw


def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples']


def init_components():
    print("[INIT] Loading RoBERTa model...")
    detector = ViolationDetector(
        mode="model",
        model_path=str(MODEL_DIR),
        temperature=1.0
    )

    print("[INIT] Creating OpenAI client for Judge...")
    judge_client = OpenAIClient(
        model=LLM_MODEL,
        max_tokens=30,
        temperature=0.0,
        max_retries=2
    )

    print("[INIT] Done.\n")
    return detector, judge_client


def prepare_context_for_model(sample):
    history = []
    for turn in sample['prefix_dialog'] + sample['generated_dialog']:
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    last_turn = history[-1]['text'] if history else ""
    context = {
        'situation': sample.get('situation', ''),
        'history': history,
        'recent_turns': history,
        'summary': '',
        'formatted': '\n'.join(f"{t['speaker']}: {t['text']}" for t in history)
    }
    return context, last_turn


def run_model_detection(detector, sample):
    context, last_text = prepare_context_for_model(sample)
    candidate = {'id': 0, 'text': last_text}
    result = detector.detect(context, candidate)
    return {
        'label': result['top_violation'],
        'confidence': float(result['confidence']),
        'probabilities': {k: float(v) for k, v in result.get('all_probabilities', {}).items()}
    }


def run_judge_detection(judge_client, sample):
    dialog = sample['prefix_dialog'] + sample['generated_dialog']
    user_prompt = build_classify_prompt(dialog)

    t0 = time.time()
    try:
        result = judge_client.call(
            system_prompt=JUDGE_CLASSIFY_SYSTEM,
            user_prompt=user_prompt,
            retry_message=JUDGE_CLASSIFY_RETRY
        )
        elapsed_ms = (time.time() - t0) * 1000
        label = normalize_label(result.get('label', 'unknown'))
        return {
            'label': label,
            'raw_response': result,
            'latency_ms': round(elapsed_ms, 1),
            'error': None
        }
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        return {
            'label': 'error',
            'raw_response': str(e),
            'latency_ms': round(elapsed_ms, 1),
            'error': str(e)
        }


def get_completed_indices(log_path):
    completed = set()
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        completed.add(entry['sample_index'])
                    except json.JSONDecodeError:
                        pass
    return completed


def cohens_kappa(preds_a, preds_b, labels):
    n = len(preds_a)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(preds_a, preds_b) if a == b) / n
    pe = 0.0
    for label in labels:
        pa = sum(1 for a in preds_a if a == label) / n
        pb = sum(1 for b in preds_b if b == label) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def per_class_f1(golds, preds):
    result = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[label] = round(f1, 4)
    return result


def compute_metrics(results):
    valid = [r for r in results if r['judge']['error'] is None]
    n = len(valid)
    if n == 0:
        return None

    gold_labels = [r['gold_label'].lower() for r in valid]
    model_preds = [r['model']['label'].lower() for r in valid]
    judge_preds = [r['judge']['label'].lower() for r in valid]

    model_acc = sum(1 for g, p in zip(gold_labels, model_preds) if g == p) / n
    judge_acc = sum(1 for g, p in zip(gold_labels, judge_preds) if g == p) / n
    agree = sum(1 for m, j in zip(model_preds, judge_preds) if m == j)
    agreement_rate = agree / n
    kappa = cohens_kappa(model_preds, judge_preds, LABELS)
    model_f1 = per_class_f1(gold_labels, model_preds)
    judge_f1 = per_class_f1(gold_labels, judge_preds)

    # FPR: Normal 샘플 중 Violation으로 오탐한 비율
    # FNR: Violation 샘플 중 Normal로 놓친 비율
    normals = [(g, mp, jp) for g, mp, jp in zip(gold_labels, model_preds, judge_preds) if g == 'normal']
    violations = [(g, mp, jp) for g, mp, jp in zip(gold_labels, model_preds, judge_preds) if g != 'normal']

    model_fp = sum(1 for _, mp, _ in normals if mp != 'normal')
    judge_fp = sum(1 for _, _, jp in normals if jp != 'normal')
    model_fn = sum(1 for _, mp, _ in violations if mp == 'normal')
    judge_fn = sum(1 for _, _, jp in violations if jp == 'normal')

    model_fpr = model_fp / len(normals) if normals else 0.0
    judge_fpr = judge_fp / len(normals) if normals else 0.0
    model_fnr = model_fn / len(violations) if violations else 0.0
    judge_fnr = judge_fn / len(violations) if violations else 0.0

    disagreements = []
    for r in valid:
        ml = r['model']['label'].lower()
        jl = r['judge']['label'].lower()
        gl = r['gold_label'].lower()
        if ml != jl:
            disagreements.append({'idx': r['sample_index'], 'gold': gl, 'model': ml, 'judge': jl})

    model_latencies = [r['model_latency_ms'] for r in valid]
    judge_latencies = [r['judge']['latency_ms'] for r in valid]

    return {
        'n_samples': n,
        'n_errors': len(results) - n,
        'n_normals': len(normals),
        'n_violations': len(violations),
        'model_accuracy': round(model_acc, 4),
        'judge_accuracy': round(judge_acc, 4),
        'model_fpr': round(model_fpr, 4),
        'judge_fpr': round(judge_fpr, 4),
        'model_fnr': round(model_fnr, 4),
        'judge_fnr': round(judge_fnr, 4),
        'model_fp': model_fp,
        'judge_fp': judge_fp,
        'model_fn': model_fn,
        'judge_fn': judge_fn,
        'agreement_rate': round(agreement_rate, 4),
        'cohens_kappa': round(kappa, 4),
        'model_per_class_f1': model_f1,
        'judge_per_class_f1': judge_f1,
        'model_macro_f1': round(sum(model_f1.values()) / len(model_f1), 4),
        'judge_macro_f1': round(sum(judge_f1.values()) / len(judge_f1), 4),
        'model_latency_mean_ms': round(sum(model_latencies) / n, 1),
        'judge_latency_mean_ms': round(sum(judge_latencies) / n, 1),
        'n_disagreements': len(disagreements),
        'disagreement_rate': round(len(disagreements) / n, 4),
        'disagreement_details': disagreements[:20]
    }


def format_report(metrics):
    lines = []
    lines.append("=" * 72)
    lines.append("   TABLE 2: Model vs LLM-Judge Classification Comparison")
    lines.append("=" * 72)
    lines.append(f"  Samples evaluated: {metrics['n_samples']}  (errors: {metrics['n_errors']})")
    lines.append("")

    lines.append("─" * 72)
    lines.append("  A. Overall Performance (vs Gold)")
    lines.append("─" * 72)
    lines.append(f"  {'Metric':<30} {'Model':>12}  {'Judge':>12}")
    lines.append(f"  {'─'*30} {'─'*12}  {'─'*12}")
    lines.append(f"  {'Accuracy':<30} {metrics['model_accuracy']:>12.4f}  {metrics['judge_accuracy']:>12.4f}")
    lines.append(f"  {'Macro F1':<30} {metrics['model_macro_f1']:>12.4f}  {metrics['judge_macro_f1']:>12.4f}")
    lines.append(f"  {'Mean Latency (ms)':<30} {metrics['model_latency_mean_ms']:>12.1f}  {metrics['judge_latency_mean_ms']:>12.1f}")
    lines.append("")

    lines.append("─" * 72)
    lines.append("  B. Error Rates (Binary: Violation vs Normal)")
    lines.append("─" * 72)
    lines.append(f"  {'Metric':<30} {'Model':>12}  {'Judge':>12}")
    lines.append(f"  {'─'*30} {'─'*12}  {'─'*12}")
    lines.append(f"  {'FPR (Normal→Violation)':<30} {metrics['model_fpr']:>12.4f}  {metrics['judge_fpr']:>12.4f}")
    lines.append(f"  {'FNR (Violation→Normal)':<30} {metrics['model_fnr']:>12.4f}  {metrics['judge_fnr']:>12.4f}")
    lines.append(f"  {'False Positives':<30} {metrics['model_fp']:>12d}  {metrics['judge_fp']:>12d}")
    lines.append(f"  {'False Negatives':<30} {metrics['model_fn']:>12d}  {metrics['judge_fn']:>12d}")
    lines.append(f"  (Normal samples: {metrics['n_normals']}, Violation samples: {metrics['n_violations']})")
    lines.append("")

    lines.append("─" * 72)
    lines.append("  C. Per-Class F1 (vs Gold)")
    lines.append("─" * 72)
    lines.append(f"  {'Class':<12} {'Model F1':>12}  {'Judge F1':>12}  {'Δ':>8}")
    lines.append(f"  {'─'*12} {'─'*12}  {'─'*12}  {'─'*8}")
    for label in LABELS:
        mf = metrics['model_per_class_f1'].get(label, 0.0)
        jf = metrics['judge_per_class_f1'].get(label, 0.0)
        delta = jf - mf
        lines.append(f"  {label.upper():<12} {mf:>12.4f}  {jf:>12.4f}  {delta:>+8.4f}")
    lines.append("")

    lines.append("─" * 72)
    lines.append("  D. Model-Judge Agreement")
    lines.append("─" * 72)
    lines.append(f"  Agreement Rate:   {metrics['agreement_rate']:.4f}")
    lines.append(f"  Cohen's Kappa:    {metrics['cohens_kappa']:.4f}")
    lines.append(f"  Disagreements:    {metrics['n_disagreements']}/{metrics['n_samples']}")
    lines.append("")

    if metrics['disagreement_details']:
        lines.append("─" * 72)
        lines.append("  E. Sample Disagreements (first 20)")
        lines.append("─" * 72)
        lines.append(f"  {'Idx':>5}  {'Gold':<8}  {'Model':<8}  {'Judge':<8}")
        lines.append(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
        for d in metrics['disagreement_details']:
            lines.append(f"  {d['idx']:>5}  {d['gold']:<8}  {d['model']:<8}  {d['judge']:<8}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Table 2: Model vs Judge Comparison")
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    samples = load_data()
    if args.n:
        samples = samples[args.start:args.start + args.n]
    else:
        samples = samples[args.start:]
    print(f"[DATA] {len(samples)} samples loaded\n")

    detector, judge_client = init_components()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "judge_log.jsonl"

    completed = get_completed_indices(log_path) if args.resume else set()
    if completed:
        print(f"[RESUME] Skipping {len(completed)} already completed samples\n")

    results = []
    if args.resume and log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    n_total = len(samples)
    n_done = len(completed)

    with open(log_path, 'a', encoding='utf-8') as log_f:
        for i, sample in enumerate(samples):
            idx = args.start + i
            if idx in completed:
                continue

            t_model = time.time()
            model_result = run_model_detection(detector, sample)
            model_ms = (time.time() - t_model) * 1000

            judge_result = run_judge_detection(judge_client, sample)

            gold = sample.get('primary_label', 'unknown')
            entry = {
                'sample_index': idx,
                'session_id': sample.get('esconv_session_id', f'unknown_{idx}'),
                'gold_label': gold,
                'model': model_result,
                'model_latency_ms': round(model_ms, 1),
                'judge': judge_result
            }

            log_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            log_f.flush()
            results.append(entry)
            n_done += 1

            model_lbl = model_result['label']
            judge_lbl = judge_result['label']
            match = "✓" if model_lbl == judge_lbl else "✗"
            print(f"  [{n_done:3d}/{n_total}] idx={idx:3d}  gold={gold:<8s}  "
                  f"model={model_lbl:<8s}  judge={judge_lbl:<8s}  agree={match}")

    print(f"\n{'='*60}")
    metrics = compute_metrics(results)
    if metrics is None:
        print("[ERROR] No valid results")
        return

    report = format_report(metrics)
    print(report)

    with open(OUTPUT_DIR / "table2_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    with open(OUTPUT_DIR / "table2_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
