"""
Table 5: Cross-Domain Generalization (False Positive Analysis)

외부 상담/공감 데이터셋(모두 Normal)에 RoBERTa 모델을 적용하여
도메인 일반화 성능(FPR) 측정.

Sources:
  - CounselChat (실제 심리상담 Q&A)
  - EmpatheticDialogues (공감 대화)
  - ESConv (원본 상담 대화)

Output:
    experiment_results/table5/
        table5_report.txt
        table5_metrics.json
        table5_latex.tex
        counsel_chat_log.jsonl
        empathetic_log.jsonl
        esconv_log.jsonl
"""
import json
import sys
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.step3_violation_detector import ViolationDetector

# ── Config ──────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
DATA_DIR = Path(__file__).parent.parent / "data" / "external"
OUTPUT_DIR = Path(__file__).parent / "table5"

LABELS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
RANDOM_SEED = 42


# ═════════════════════════════════════════════════════════════
# Data Loaders
# ═════════════════════════════════════════════════════════════

def load_counsel_chat(max_n=None):
    path = DATA_DIR / "counsel_chat.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for i, entry in enumerate(data):
        question = entry.get('question_text') or entry.get('question') or ''
        answer = entry.get('answer') or ''
        if not question or not question.strip() or not answer or not answer.strip():
            continue

        dialog = [
            {'speaker': 'seeker', 'text': question.strip()},
            {'speaker': 'supporter', 'text': answer.strip()}
        ]
        context = {
            'situation': question[:200],
            'history': dialog,
            'recent_turns': dialog,
            'summary': '',
            'formatted': f"seeker: {question}\nsupporter: {answer}"
        }
        samples.append({
            'source': 'counsel_chat',
            'index': i,
            'context': context,
            'response_text': answer.strip(),
            'gold_label': 'normal'
        })

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    if max_n:
        samples = samples[:max_n]
    return samples


def load_empathetic_dialogues(max_n=None):
    import pyarrow.parquet as pq

    path = DATA_DIR / "empathetic_train.parquet"
    table = pq.read_table(path)

    conv_groups = defaultdict(list)
    for i in range(table.num_rows):
        conv_id = str(table.column('conv_id')[i])
        utt_idx = int(str(table.column('utterance_idx')[i]))
        speaker_idx = int(str(table.column('speaker_idx')[i]))
        utterance = str(table.column('utterance')[i])
        context_label = str(table.column('context')[i])

        conv_groups[conv_id].append({
            'utt_idx': utt_idx,
            'speaker_idx': speaker_idx,
            'utterance': utterance.replace('_comma_', ','),
            'context': context_label
        })

    samples = []
    for conv_id, turns in conv_groups.items():
        turns = sorted(turns, key=lambda x: x['utt_idx'])

        last_listener_idx = None
        for t_idx, turn in enumerate(turns):
            if turn['speaker_idx'] == 0:
                last_listener_idx = t_idx

        if last_listener_idx is None or last_listener_idx == 0:
            continue

        dialog = []
        for turn in turns[:last_listener_idx + 1]:
            speaker = 'supporter' if turn['speaker_idx'] == 0 else 'seeker'
            dialog.append({'speaker': speaker, 'text': turn['utterance']})

        response_text = turns[last_listener_idx]['utterance']
        situation = turns[0].get('context', '')

        context = {
            'situation': situation,
            'history': dialog,
            'recent_turns': dialog,
            'summary': '',
            'formatted': '\n'.join(f"{t['speaker']}: {t['text']}" for t in dialog)
        }

        samples.append({
            'source': 'empathetic',
            'index': len(samples),
            'context': context,
            'response_text': response_text,
            'gold_label': 'normal'
        })

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    if max_n:
        samples = samples[:max_n]
    return samples


def load_esconv_original(max_n=None):
    path = DATA_DIR / "ESConv.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for i, session in enumerate(data):
        dialog_raw = session.get('dialog', [])
        situation = session.get('situation', '')

        if len(dialog_raw) < 2:
            continue

        last_sup_idx = None
        for t_idx, turn in enumerate(dialog_raw):
            if turn.get('speaker') == 'supporter':
                last_sup_idx = t_idx

        if last_sup_idx is None or last_sup_idx == 0:
            continue

        dialog = []
        for turn in dialog_raw[:last_sup_idx + 1]:
            dialog.append({
                'speaker': turn['speaker'],
                'text': turn.get('content', turn.get('text', '')).strip()
            })

        response_text = dialog[-1]['text']
        context = {
            'situation': situation,
            'history': dialog,
            'recent_turns': dialog,
            'summary': '',
            'formatted': '\n'.join(f"{t['speaker']}: {t['text']}" for t in dialog)
        }

        samples.append({
            'source': 'esconv',
            'index': i,
            'context': context,
            'response_text': response_text,
            'gold_label': 'normal'
        })

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    if max_n:
        samples = samples[:max_n]
    return samples


# ═════════════════════════════════════════════════════════════
# Detection
# ═════════════════════════════════════════════════════════════

def run_detection(detector, samples, source_name):
    results = []
    n = len(samples)
    print(f"\n[{source_name.upper()}] Running detection on {n} samples...")

    for i, sample in enumerate(samples):
        t0 = time.time()
        candidate = {'id': 0, 'text': sample['response_text']}
        try:
            det = detector.detect(sample['context'], candidate)
            elapsed = (time.time() - t0) * 1000
            result = {
                'source': source_name,
                'index': sample['index'],
                'gold_label': 'normal',
                'predicted_label': det['top_violation'],
                'confidence': float(det['confidence']),
                'probabilities': {k: float(v) for k, v in det.get('all_probabilities', {}).items()},
                'latency_ms': round(elapsed, 1),
                'error': None
            }
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            result = {
                'source': source_name,
                'index': sample['index'],
                'gold_label': 'normal',
                'predicted_label': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'latency_ms': round(elapsed, 1),
                'error': str(e)
            }

        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            n_fp = sum(1 for r in results if r['predicted_label'] != 'normal' and r['error'] is None)
            print(f"  [{i+1:4d}/{n}] FP so far: {n_fp}")

    return results


# ═════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════

def compute_source_metrics(results):
    valid = [r for r in results if r['error'] is None]
    n = len(valid)
    if n == 0:
        return None

    fp = [r for r in valid if r['predicted_label'] != 'normal']
    tp = [r for r in valid if r['predicted_label'] == 'normal']
    fpr = len(fp) / n if n > 0 else 0.0
    fp_types = Counter(r['predicted_label'] for r in fp)

    normal_confs = [r['confidence'] for r in tp]
    fp_confs = [r['confidence'] for r in fp]
    latencies = [r['latency_ms'] for r in valid]

    return {
        'n_total': n,
        'n_errors': len(results) - n,
        'n_correct': len(tp),
        'n_false_positive': len(fp),
        'false_positive_rate': round(fpr, 4),
        'specificity': round(1 - fpr, 4),
        'fp_type_distribution': dict(fp_types),
        'normal_confidence_mean': round(sum(normal_confs) / len(normal_confs), 4) if normal_confs else 0.0,
        'normal_confidence_min': round(min(normal_confs), 4) if normal_confs else 0.0,
        'fp_confidence_mean': round(sum(fp_confs) / len(fp_confs), 4) if fp_confs else 0.0,
        'fp_confidence_max': round(max(fp_confs), 4) if fp_confs else 0.0,
        'latency_mean_ms': round(sum(latencies) / len(latencies), 1),
        'latency_p95_ms': round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0.0
    }


# ═════════════════════════════════════════════════════════════
# Report Formatting
# ═════════════════════════════════════════════════════════════

SOURCE_NAMES = {
    'counsel_chat': 'CounselChat',
    'empathetic': 'EmpatheticDial.',
    'esconv': 'ESConv (orig)',
}


def format_report(all_metrics):
    lines = []
    lines.append("=" * 80)
    lines.append("   TABLE 5: Cross-Domain Generalization (False Positive Analysis)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  All external samples are 'Normal' counseling responses.")
    lines.append("  Any violation detection = False Positive → measures model specificity.")
    lines.append("")

    lines.append("─" * 80)
    lines.append("  A. Summary by Source")
    lines.append("─" * 80)
    lines.append(f"  {'Source':<20} {'N':>6}  {'FP':>5}  {'FPR':>8}  {'Spec.':>8}  {'Conf(N)':>8}  {'Lat(ms)':>8}")
    lines.append(f"  {'─'*20} {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    total_n = total_fp = 0
    for source, metrics in all_metrics.items():
        if metrics is None:
            continue
        name = SOURCE_NAMES.get(source, source)
        total_n += metrics['n_total']
        total_fp += metrics['n_false_positive']
        lines.append(
            f"  {name:<20} {metrics['n_total']:>6}  {metrics['n_false_positive']:>5}  "
            f"{metrics['false_positive_rate']:>8.4f}  {metrics['specificity']:>8.4f}  "
            f"{metrics['normal_confidence_mean']:>8.4f}  {metrics['latency_mean_ms']:>8.1f}")

    if total_n > 0:
        total_fpr = total_fp / total_n
        lines.append(f"  {'─'*20} {'─'*6}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        lines.append(f"  {'TOTAL':<20} {total_n:>6}  {total_fp:>5}  "
                      f"{total_fpr:>8.4f}  {1-total_fpr:>8.4f}  {'':>8}  {'':>8}")

    lines.append("")

    lines.append("─" * 80)
    lines.append("  B. False Positive Breakdown by Violation Type")
    lines.append("─" * 80)
    lines.append(f"  {'Source':<20} {'V1':>6}  {'V2':>6}  {'V3':>6}  {'V4':>6}  {'V5':>6}")
    lines.append(f"  {'─'*20} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for source, metrics in all_metrics.items():
        if metrics is None:
            continue
        name = SOURCE_NAMES.get(source, source)
        fp_dist = metrics.get('fp_type_distribution', {})
        lines.append(f"  {name:<20} {fp_dist.get('v1',0):>6}  {fp_dist.get('v2',0):>6}  "
                      f"{fp_dist.get('v3',0):>6}  {fp_dist.get('v4',0):>6}  {fp_dist.get('v5',0):>6}")

    lines.append("")

    lines.append("─" * 80)
    lines.append("  C. Confidence Distribution (FP cases)")
    lines.append("─" * 80)
    lines.append(f"  {'Source':<20} {'FP Conf Mean':>14}  {'FP Conf Max':>14}  {'Normal Conf Min':>16}")
    lines.append(f"  {'─'*20} {'─'*14}  {'─'*14}  {'─'*16}")
    for source, metrics in all_metrics.items():
        if metrics is None:
            continue
        name = SOURCE_NAMES.get(source, source)
        lines.append(f"  {name:<20} {metrics['fp_confidence_mean']:>14.4f}  "
                      f"{metrics['fp_confidence_max']:>14.4f}  {metrics['normal_confidence_min']:>16.4f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_latex(all_metrics):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-Domain Generalization: False Positive Analysis}")
    lines.append(r"\label{tab:cross-domain}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Source & N & FP & FPR & Specificity & Conf (Normal) \\")
    lines.append(r"\midrule")
    total_n = total_fp = 0
    for source, metrics in all_metrics.items():
        if metrics is None:
            continue
        name = SOURCE_NAMES.get(source, source)
        total_n += metrics['n_total']
        total_fp += metrics['n_false_positive']
        lines.append(f"{name} & {metrics['n_total']} & {metrics['n_false_positive']} & "
                      f"{metrics['false_positive_rate']:.4f} & {metrics['specificity']:.4f} & "
                      f"{metrics['normal_confidence_mean']:.4f} \\\\")
    if total_n > 0:
        total_fpr = total_fp / total_n
        lines.append(r"\midrule")
        lines.append(f"Total & {total_n} & {total_fp} & {total_fpr:.4f} & {1-total_fpr:.4f} & -- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Table 5: Cross-Domain Generalization")
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--source', choices=['counsel', 'empathetic', 'esconv', 'all'], default='all')
    parser.add_argument('--format', choices=['text', 'latex', 'both'], default='both')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    print("[INIT] Loading RoBERTa model...")
    detector = ViolationDetector(mode="model", model_path=str(MODEL_DIR), temperature=1.0)
    print("[INIT] Done.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_loaders = {
        'counsel': ('counsel_chat', load_counsel_chat),
        'empathetic': ('empathetic', load_empathetic_dialogues),
        'esconv': ('esconv', load_esconv_original),
    }

    sources_to_run = list(source_loaders.keys()) if args.source == 'all' else [args.source]

    all_results = {}
    all_metrics = {}

    for source_key in sources_to_run:
        source_name, loader = source_loaders[source_key]
        try:
            samples = loader(max_n=args.n)
            print(f"[{source_name.upper()}] Loaded {len(samples)} samples")
        except Exception as e:
            print(f"[{source_name.upper()}] SKIP — Load error: {e}")
            all_metrics[source_name] = None
            continue

        if not samples:
            all_metrics[source_name] = None
            continue

        results = run_detection(detector, samples, source_name)
        all_results[source_name] = results
        all_metrics[source_name] = compute_source_metrics(results)

        if args.save:
            log_path = OUTPUT_DIR / f"{source_name}_log.jsonl"
            with open(log_path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

    report = format_report(all_metrics)
    print(report)

    with open(OUTPUT_DIR / "table5_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    if args.format in ('latex', 'both'):
        latex = format_latex(all_metrics)
        with open(OUTPUT_DIR / "table5_latex.tex", 'w') as f:
            f.write(latex)

    saveable = {k: v for k, v in all_metrics.items() if v is not None}
    with open(OUTPUT_DIR / "table5_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
