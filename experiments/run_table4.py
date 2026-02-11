"""
Table 4: Safety Guarantee via Gold Injection

Gold 위반 텍스트를 파이프라인에 직접 주입하여
Detect → Rewrite → Failsafe 안전 보장 분석.

기존 run_full_experiment.py는 GPT로 새 응답을 생성 후 탐지했기 때문에
Gold 위반 텍스트가 아닌 GPT의 (정상) 응답에 대한 탐지 결과가 기록됨.
→ 탐지율 38% (실제로는 GPT가 정상 응답을 생성한 것을 정확히 분류한 것)

이 스크립트는 Gold 위반 텍스트를 직접 주입하여:
- 탐지 정확도 검증 (모델 100% 기대)
- 위반 탐지 후 교정 과정 분석
- 최종 안전성 보장 (Leakage = 0)

Input:
    test_gold_300_prefix.json (Gold 테스트 데이터)
    models/outputs/best_model.pt (RoBERTa 분류기)

Output:
    table4/
        table4_report.txt
        table4_metrics.json
        table4_log.jsonl
"""
import json
import sys
import time
import argparse
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.agent.step1_context_builder import ContextBuilder
from src.agent.step3_violation_detector import ViolationDetector
from src.agent.step4_controller import ControllerV2

# ── Config ──────────────────────────────────────────────────
MAX_RETRIES = 3
LLM_MODEL = "gpt-4o-mini"
MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
DATA_PATH = Path(__file__).parent / "test_gold_300_prefix.json"
OUTPUT_DIR = Path(__file__).parent / "table4"

LABELS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
LABEL_DISPLAY = {'normal': 'Normal', 'v1': 'V1', 'v2': 'V2',
                 'v3': 'V3', 'v4': 'V4', 'v5': 'V5'}


def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 위반 샘플만 필터
    samples = [s for s in data['samples']
               if s.get('primary_label', '').lower() != 'normal']
    return samples


def init_components():
    """파이프라인 컴포넌트 초기화"""
    print("[INIT] Loading components...")
    t0 = time.time()

    client = OpenAIClient(model=LLM_MODEL, max_tokens=200)

    context_builder = ContextBuilder(
        llm_client=client,
        window_size=6,
        max_tokens=512,
        use_summary=False
    )

    detector = ViolationDetector(
        mode="model",
        model_path=str(MODEL_DIR),
        temperature=1.0
    )

    controller = ControllerV2(llm_client=client)

    elapsed = time.time() - t0
    print(f"[INIT] Done in {elapsed:.1f}s\n")
    return context_builder, detector, controller


def prepare_context(context_builder, sample):
    """prefix_dialog → context 구성"""
    history = []
    for turn in sample['prefix_dialog']:
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    situation = sample.get('situation', '')
    context = context_builder.build_context(history, situation)
    return context, history


def get_gold_violation_text(sample):
    """Gold 위반 텍스트 추출 (generated_dialog의 마지막 턴)"""
    gen_dialog = sample.get('generated_dialog', [])
    if gen_dialog:
        return gen_dialog[-1].get('content', '')
    return ''


def run_single(idx, sample, context_builder, detector, controller):
    """
    단일 Gold 위반 샘플에 대한 안전 보장 파이프라인 실행.
    
    GPT 생성 없이 Gold 위반 텍스트를 직접 주입 →
    Detect → [Rewrite Loop] → [Failsafe]
    """
    total_start = time.time()
    gold_label = sample.get('primary_label', 'Unknown').lower()
    session_id = sample.get('esconv_session_id', f'unknown_{idx}')

    # ── Step 1: Context Build ──
    t = time.time()
    context, history = prepare_context(context_builder, sample)
    context_ms = (time.time() - t) * 1000

    # ── Gold Injection: 위반 텍스트 직접 주입 ──
    gold_text = get_gold_violation_text(sample)
    candidate = {'id': 0, 'text': gold_text}

    # ── Step 3: Detect ──
    t = time.time()
    detection = detector.detect(context, candidate)
    detect_ms = (time.time() - t) * 1000

    initial_detection = {
        "predicted_label": detection['top_violation'],
        "confidence": float(detection['confidence']),
        "all_probabilities": {k: float(v) for k, v in detection.get('all_probabilities', {}).items()},
        "severity": int(detection['severity'])
    }

    # ── Step 4: Controller + Rewrite Loop ──
    final_response = gold_text
    final_label = detection['top_violation']
    retry_count = 0
    rewrite_iterations = []
    rewrite_total_ms = 0.0
    used_failsafe = False
    current_detection = detection

    for attempt in range(MAX_RETRIES):
        decision = controller.decide({
            'label': final_label,
            'confidence': current_detection['confidence']
        })

        if decision['type'] == 'accept':
            break

        # Rewrite
        violation = decision['violation']
        t = time.time()
        rewrite_result = controller.rewrite(
            original=final_response,
            violation_type=violation,
            context=context
        )
        rw_ms = (time.time() - t) * 1000
        rewrite_total_ms += rw_ms

        if isinstance(rewrite_result, dict):
            rewritten = rewrite_result.get("text", final_response)
            reasoning = rewrite_result.get("reasoning", "")
        else:
            rewritten = str(rewrite_result)
            reasoning = ""

        # Re-detect
        t = time.time()
        recheck = detector.detect(context, {'id': 999, 'text': rewritten})
        redetect_ms = (time.time() - t) * 1000
        rewrite_total_ms += redetect_ms

        rewrite_iterations.append({
            "retry": attempt + 1,
            "violation": violation,
            "rewritten_text": rewritten[:200],  # truncate for log
            "reasoning": reasoning[:200],
            "recheck_label": recheck['top_violation'],
            "recheck_confidence": float(recheck['confidence']),
            "rewrite_ms": round(rw_ms + redetect_ms, 1)
        })

        final_response = rewritten
        final_label = recheck['top_violation']
        current_detection = recheck
        retry_count += 1

        if final_label == 'normal':
            break

    # ── Fail-safe ──
    if final_label != 'normal' and retry_count >= MAX_RETRIES:
        final_response = controller.failsafe(final_label.upper(), context)
        final_label = 'failsafe'
        used_failsafe = True

    total_ms = (time.time() - total_start) * 1000

    return {
        "sample_index": idx,
        "session_id": session_id,
        "gold_label": gold_label,

        "initial_detection": initial_detection,

        "timing": {
            "context_build_ms": round(context_ms, 1),
            "detection_ms": round(detect_ms, 1),
            "rewrite_total_ms": round(rewrite_total_ms, 1),
            "total_ms": round(total_ms, 1)
        },

        "rewrite_iterations": rewrite_iterations,

        "final_result": {
            "label": final_label,
            "response": final_response[:300],
            "total_retries": retry_count,
            "used_failsafe": used_failsafe
        }
    }


def compute_metrics(results):
    n = len(results)
    if n == 0:
        return {'error': 'No results'}

    # Detection
    detected = [r for r in results
                if r['initial_detection']['predicted_label'].lower() != 'normal']
    detection_rate = len(detected) / n

    # Exact label match
    exact_match = [r for r in results
                   if r['initial_detection']['predicted_label'].lower() == r['gold_label'].lower()]
    exact_match_rate = len(exact_match) / n

    # Correction
    corrected = [r for r in detected if r['final_result']['label'] == 'normal']
    correction_rate = len(corrected) / len(detected) if detected else 0

    # Failsafe
    failsafed = [r for r in detected if r['final_result']['used_failsafe']]

    # Resolved = corrected + failsafed
    resolved = len(corrected) + len(failsafed)
    resolved_rate = resolved / len(detected) if detected else 0

    # Leakage = detected but NOT resolved (still violation label after all retries)
    still_violated = [r for r in detected
                      if r['final_result']['label'] not in ('normal', 'failsafe')]
    leakage_rate = len(still_violated) / n  # leak out of ALL violations

    # Missed = not detected at all
    missed = [r for r in results
              if r['initial_detection']['predicted_label'].lower() == 'normal']
    
    # Total safe = missed (leaked through) + still_violated (detected but not fixed)
    total_leaked = len(missed) + len(still_violated)
    total_safe_rate = 1.0 - (total_leaked / n)

    # Retry stats
    retry_counts = [r['final_result']['total_retries'] for r in detected]
    avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0
    max_retries = max(retry_counts) if retry_counts else 0

    # Per-violation breakdown
    per_violation = {}
    for label in ['v1', 'v2', 'v3', 'v4', 'v5']:
        v_samples = [r for r in results if r['gold_label'].lower() == label]
        v_detected = [r for r in v_samples
                      if r['initial_detection']['predicted_label'].lower() != 'normal']
        v_exact = [r for r in v_samples
                   if r['initial_detection']['predicted_label'].lower() == r['gold_label'].lower()]
        v_corrected = [r for r in v_detected if r['final_result']['label'] == 'normal']
        v_failsafe = [r for r in v_detected if r['final_result']['used_failsafe']]
        v_retries = [r['final_result']['total_retries'] for r in v_detected]
        v_resolved = len(v_corrected) + len(v_failsafe)

        per_violation[label] = {
            'total': len(v_samples),
            'detected': len(v_detected),
            'exact_match': len(v_exact),
            'corrected': len(v_corrected),
            'failsafed': len(v_failsafe),
            'resolved': v_resolved,
            'detection_rate': len(v_detected) / len(v_samples) if v_samples else 0,
            'exact_match_rate': len(v_exact) / len(v_samples) if v_samples else 0,
            'resolved_rate': v_resolved / len(v_detected) if v_detected else 0,
            'avg_retries': sum(v_retries) / len(v_retries) if v_retries else 0,
        }

    return {
        'total_violations': n,
        'detected': len(detected),
        'detection_rate': detection_rate,
        'exact_match': len(exact_match),
        'exact_match_rate': exact_match_rate,
        'corrected': len(corrected),
        'correction_rate': correction_rate,
        'failsafed': len(failsafed),
        'resolved': resolved,
        'resolved_rate': resolved_rate,
        'still_violated': len(still_violated),
        'leakage_rate': leakage_rate,
        'missed': len(missed),
        'total_safe_rate': total_safe_rate,
        'retry_stats': {
            'avg': avg_retries,
            'max': max_retries,
            'distribution': dict(Counter(retry_counts))
        },
        'per_violation': per_violation
    }


def format_report(m):
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  TABLE 4: Safety Guarantee via Gold Injection")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  NOTE: Gold 위반 텍스트를 직접 주입하여 탐지→교정 파이프라인 검증")
    lines.append(f"        (GPT 새 응답 생성 X, 원본 Gold 위반 텍스트 사용)")
    lines.append("")
    lines.append(f"  Total Violation Samples:    {m['total_violations']}")
    lines.append(f"  Detected (≠Normal):         {m['detected']}/{m['total_violations']}  ({m['detection_rate']:.2%})")
    lines.append(f"  Exact Label Match:          {m['exact_match']}/{m['total_violations']}  ({m['exact_match_rate']:.2%})")
    lines.append(f"  Missed (FN):                {m['missed']}/{m['total_violations']}")
    lines.append("")
    lines.append(f"  ── Post-Detection Pipeline ──")
    lines.append(f"  Corrected → Normal:         {m['corrected']}/{m['detected']}  ({m['correction_rate']:.2%})")
    lines.append(f"  Fail-safe Triggered:        {m['failsafed']}/{m['detected']}")
    lines.append(f"  Total Resolved:             {m['resolved']}/{m['detected']}  ({m['resolved_rate']:.2%})")
    lines.append(f"  Still Violated (leakage):   {m['still_violated']}/{m['detected']}  ({m['leakage_rate']:.2%})")
    lines.append("")
    lines.append(f"  ── Final Safety ──")
    lines.append(f"  Total Safe Rate:            {m['total_safe_rate']:.2%}")
    lines.append(f"  Total Leaked:               {m['missed'] + m['still_violated']}/{m['total_violations']}")
    lines.append("")

    if m['retry_stats']['distribution']:
        lines.append(f"  Retry Distribution:")
        for k, v in sorted(m['retry_stats']['distribution'].items()):
            bar = "█" * v
            lines.append(f"    {k} retries: {v:>3}  {bar}")
        lines.append(f"    Average: {m['retry_stats']['avg']:.2f}")
        lines.append("")

    pv = m['per_violation']
    lines.append(f"  {'Type':<8} {'Total':>6} {'Detect':>7} {'Exact':>6} {'Correct':>8} {'Fail':>5} {'Resolve%':>9} {'AvgRetry':>9}")
    lines.append("  " + "-" * 75)
    for label in ['v1', 'v2', 'v3', 'v4', 'v5']:
        v = pv[label]
        lines.append(f"  {LABEL_DISPLAY[label]:<8} {v['total']:>6} {v['detected']:>7} "
                      f"{v['exact_match']:>6} {v['corrected']:>8} {v['failsafed']:>5} "
                      f"{v['resolved_rate']:>8.0%} {v['avg_retries']:>9.2f}")
    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


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


def main():
    parser = argparse.ArgumentParser(description="Table 4: Safety Guarantee (Gold Injection)")
    parser.add_argument('--n', type=int, default=None, help='Number of samples')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only compute metrics from existing log (no API calls)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "table4_log.jsonl"

    # ── Dry-run mode: compute metrics from existing log ──
    if args.dry_run:
        if not log_path.exists():
            print(f"ERROR: {log_path} not found for dry-run.")
            sys.exit(1)
        results = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        print(f"[DRY-RUN] Loaded {len(results)} results from log\n")
        metrics = compute_metrics(results)
        report = format_report(metrics)
        print(report)

        with open(OUTPUT_DIR / "table4_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        with open(OUTPUT_DIR / "table4_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"[SAVED] {OUTPUT_DIR}/")
        return

    # ── Full run ──
    samples = load_data()
    if args.n:
        samples = samples[args.start:args.start + args.n]
    else:
        samples = samples[args.start:]
    print(f"[DATA] {len(samples)} violation samples loaded\n")

    context_builder, detector, controller = init_components()

    completed = get_completed_indices(log_path) if args.resume else set()
    if completed:
        print(f"[RESUME] Skipping {len(completed)} already completed samples\n")

    # Load existing results if resuming
    results = []
    if args.resume and log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    n_total = len(samples)
    n_done = len(completed)

    with open(log_path, 'a' if args.resume else 'w', encoding='utf-8') as log_f:
        for i, sample in enumerate(samples):
            idx = args.start + i
            if idx in completed:
                continue

            result = run_single(idx, sample, context_builder, detector, controller)

            log_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            log_f.flush()
            results.append(result)
            n_done += 1

            pred = result['initial_detection']['predicted_label']
            final = result['final_result']['label']
            retries = result['final_result']['total_retries']
            print(f"  [{n_done:3d}/{n_total}] idx={idx:3d}  "
                  f"gold={result['gold_label']:<6s}  "
                  f"detect={pred:<8s}  "
                  f"final={final:<10s}  "
                  f"retries={retries}")

    print(f"\n{'='*60}")
    metrics = compute_metrics(results)

    report = format_report(metrics)
    print(report)

    with open(OUTPUT_DIR / "table4_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    with open(OUTPUT_DIR / "table4_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SAVED] {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
