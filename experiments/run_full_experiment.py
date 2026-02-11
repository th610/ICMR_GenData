"""
Full Experiment Runner — Set A / B / C 메트릭 수집

Usage:
    python run_full_experiment.py              # 전체 300개
    python run_full_experiment.py --n 10       # 처음 10개만 (테스트용)
    python run_full_experiment.py --resume     # 이전 실행 이어서

Output:
    experiment_results/
        per_sample/           # 샘플별 JSON 로그
            sample_000.json
            sample_001.json
            ...
        experiment_log.jsonl  # 한 줄씩 append (실시간 저장)
        summary.json          # 전체 집계 결과
"""
import json
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.agent.step1_context_builder import ContextBuilder
from src.agent.step2_candidate_generator import CandidateGenerator
from src.agent.step3_violation_detector import ViolationDetector
from src.agent.step4_controller import ControllerV2

# ── Config ──────────────────────────────────────────────────
MAX_RETRIES = 3          # 최대 rewrite 재시도 횟수
NUM_CANDIDATES = 1       # 후보 수 (현재 1)
LLM_MODEL = "gpt-4o-mini"
MODEL_DIR = Path(__file__).parent.parent / "models" / "outputs"
DATA_PATH = Path(__file__).parent / "test_gold_300_prefix.json"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """테스트 데이터 로드"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples']


def init_components():
    """파이프라인 컴포넌트 초기화 (1회만)"""
    print("[INIT] Loading components...")
    t0 = time.time()

    client = OpenAIClient(model=LLM_MODEL, max_tokens=200)

    context_builder = ContextBuilder(
        llm_client=client,
        window_size=6,
        max_tokens=512,
        use_summary=False
    )

    generator = CandidateGenerator(
        llm_client=client,
        num_candidates=NUM_CANDIDATES,
        temperature=0.7
    )

    detector = ViolationDetector(
        mode="model",
        model_path=str(MODEL_DIR),
        temperature=1.0
    )

    controller = ControllerV2(llm_client=client)

    elapsed = time.time() - t0
    print(f"[INIT] Done in {elapsed:.1f}s  (model={LLM_MODEL}, detector=RoBERTa)\n")
    return context_builder, generator, detector, controller


def prepare_history(sample):
    """prefix_dialog → history 형식 변환"""
    history = []
    for turn in sample['prefix_dialog']:
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    return history


def run_single(idx, sample, context_builder, generator, detector, controller):
    """
    단일 샘플 실행 → 결과 dict 반환

    Returns:
        {
            "sample_index": int,
            "session_id": str,
            "gold_label": str,
            "situation": str,

            # Set A: Detection
            "initial_detection": {
                "predicted_label": str,
                "confidence": float,
                "all_probabilities": dict,
                "severity": int
            },

            # Set B: Efficiency
            "timing": {
                "context_build_ms": float,
                "generation_ms": float,
                "detection_ms": float,
                "rewrite_total_ms": float,
                "total_ms": float
            },

            # Set C: Safety
            "rewrite_iterations": [
                {
                    "retry": int,
                    "violation": str,
                    "rewritten_text": str,
                    "reasoning": str,
                    "recheck_label": str,
                    "recheck_confidence": float,
                    "recheck_probabilities": dict,
                    "rewrite_ms": float
                }
            ],
            "final_result": {
                "label": str,
                "response": str,
                "total_retries": int,
                "used_failsafe": bool
            }
        }
    """
    total_start = time.time()
    gold_label = sample.get('primary_label', 'Unknown')
    session_id = sample.get('esconv_session_id', f'unknown_{idx}')
    situation = sample.get('situation', '')

    history = prepare_history(sample)

    # ── Step 1: Context ──
    t = time.time()
    context = context_builder.build_context(history, situation)
    context_ms = (time.time() - t) * 1000

    # ── Step 2: Generate ──
    t = time.time()
    candidates = generator.generate_candidates(context)
    gen_ms = (time.time() - t) * 1000

    # ── Step 3: Detect ──
    t = time.time()
    detections = detector.detect_batch(context, candidates)
    detect_ms = (time.time() - t) * 1000

    best_idx_c = min(range(len(detections)), key=lambda i: detections[i]['severity'])
    best_candidate = candidates[best_idx_c]
    best_detection = detections[best_idx_c]

    initial_detection = {
        "predicted_label": best_detection['top_violation'],
        "confidence": float(best_detection['confidence']),
        "all_probabilities": {k: float(v) for k, v in best_detection.get('all_probabilities', {}).items()},
        "severity": int(best_detection['severity'])
    }

    # ── Step 4-5: Controller + Rewrite Loop ──
    final_response = best_candidate['text']
    final_label = best_detection['top_violation']
    retry_count = 0
    rewrite_iterations = []
    rewrite_total_ms = 0.0
    used_failsafe = False

    current_detection = best_detection

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
        rw_ms += redetect_ms
        rewrite_total_ms += redetect_ms

        rewrite_iterations.append({
            "retry": attempt + 1,
            "violation": violation,
            "rewritten_text": rewritten,
            "reasoning": reasoning,
            "recheck_label": recheck['top_violation'],
            "recheck_confidence": float(recheck['confidence']),
            "recheck_probabilities": {k: float(v) for k, v in recheck.get('all_probabilities', {}).items()},
            "rewrite_ms": round(rw_ms, 1)
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
        "situation": situation[:100],

        "initial_detection": initial_detection,

        "timing": {
            "context_build_ms": round(context_ms, 1),
            "generation_ms": round(gen_ms, 1),
            "detection_ms": round(detect_ms, 1),
            "rewrite_total_ms": round(rewrite_total_ms, 1),
            "total_ms": round(total_ms, 1)
        },

        "rewrite_iterations": rewrite_iterations,

        "final_result": {
            "label": final_label,
            "response": final_response,
            "total_retries": retry_count,
            "used_failsafe": used_failsafe
        }
    }


def get_completed_indices(log_path):
    """이미 완료된 샘플 인덱스 목록 반환"""
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
    parser = argparse.ArgumentParser(description="Run full agent experiment")
    parser.add_argument('--n', type=int, default=None, help='Number of samples (default: all)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    args = parser.parse_args()

    # ── Data ──
    samples = load_data()
    total = len(samples)
    n = args.n if args.n else total
    n = min(n, total)

    print(f"{'='*70}")
    print(f" FULL EXPERIMENT: {n}/{total} samples")
    print(f" Max retries: {MAX_RETRIES}, Candidates: {NUM_CANDIDATES}")
    print(f"{'='*70}\n")

    # ── Output dirs ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sample_dir = OUTPUT_DIR / "per_sample"
    per_sample_dir.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "experiment_log.jsonl"

    # ── Resume ──
    completed = set()
    if args.resume:
        completed = get_completed_indices(log_path)
        print(f"[RESUME] {len(completed)} samples already completed\n")

    # ── Init ──
    ctx_builder, gen, det, ctrl = init_components()

    # ── Run ──
    results = []
    errors = []
    start_time = time.time()

    for i in range(args.start, n):
        if i in completed:
            continue

        sample = samples[i]
        label = sample.get('primary_label', '?')
        sid = sample.get('esconv_session_id', '?')

        print(f"[{i+1:3d}/{n}] Session {sid} (Gold: {label}) ... ", end='', flush=True)

        try:
            result = run_single(i, sample, ctx_builder, gen, det, ctrl)
            results.append(result)

            # 즉시 저장 (crash-safe)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # 개별 파일 저장
            per_sample_path = per_sample_dir / f"sample_{i:03d}.json"
            with open(per_sample_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Status
            det_label = result['initial_detection']['predicted_label']
            final = result['final_result']['label']
            retries = result['final_result']['total_retries']
            total_ms = result['timing']['total_ms']

            status = "✓" if final == 'normal' else "✗"
            print(f"{det_label:>8} → {final:>8} ({retries} retries) [{total_ms:.0f}ms] {status}")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"index": i, "session_id": sid, "error": str(e)})

    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" EXPERIMENT COMPLETE")
    print(f" Processed: {len(results)} | Errors: {len(errors)} | Time: {elapsed:.0f}s")
    print(f"{'='*70}\n")

    # Quick stats
    if results:
        # Detection accuracy (initial)
        correct = sum(1 for r in results
                      if r['initial_detection']['predicted_label'] == r['gold_label'].lower())
        print(f"Initial Detection Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")

        # Correction stats (for violations only)
        violations = [r for r in results if r['gold_label'].lower() != 'normal']
        if violations:
            corrected = sum(1 for r in violations if r['final_result']['label'] == 'normal')
            failsafed = sum(1 for r in violations if r['final_result']['used_failsafe'])
            avg_retries = sum(r['final_result']['total_retries'] for r in violations) / len(violations)
            print(f"Violation Correction:       {corrected}/{len(violations)} ({corrected/len(violations):.1%})")
            print(f"Fail-safe Used:             {failsafed}/{len(violations)}")
            print(f"Avg Retries (violations):   {avg_retries:.2f}")

        # FPR (Normal → False Positive)
        normals = [r for r in results if r['gold_label'].lower() == 'normal']
        if normals:
            false_pos = sum(1 for r in normals
                           if r['initial_detection']['predicted_label'] != 'normal')
            print(f"False Positive Rate:        {false_pos}/{len(normals)} ({false_pos/len(normals):.1%})")

        # Timing
        avg_total = sum(r['timing']['total_ms'] for r in results) / len(results)
        avg_detect = sum(r['timing']['detection_ms'] for r in results) / len(results)
        print(f"Avg Total Latency:          {avg_total:.0f}ms")
        print(f"Avg Detector Latency:       {avg_detect:.0f}ms")

    # Save summary
    summary = {
        "total_samples": len(results),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "error_details": errors
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  → experiment_log.jsonl  (for analyze_results.py)")
    print(f"  → per_sample/           (individual logs)")
    print(f"  → summary.json")


if __name__ == "__main__":
    main()
