"""
Full Agent Integration Test - 골드 데이터로 Turn 4 생성 및 평가

Process:
    1. Context Builder: History → Context
    2. Candidate Generator: Context → candidates
    3. Violation Detector: Candidates → Violation scores
    4. Controller: Select best or rewrite
    5. Compare with gold label
"""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv(override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.openai_client import OpenAIClient
from src.agent.step1_context_builder import ContextBuilder
from src.agent.step2_candidate_generator import CandidateGenerator
from src.agent.step3_violation_detector import ViolationDetector
from src.agent.step4_controller import ControllerV2


def load_gold_sample(index=None):
    """골드 데이터 로드"""
    import random
    
    gold_path = Path(__file__).parent / "test_gold_300_prefix.json"
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # index가 None이면 랜덤 선택
    if index is None:
        index = random.randint(0, len(data['samples']) - 1)
        print(f"[INFO] Randomly selected sample index: {index}")
    
    return data['samples'][index]


def prepare_dialog_history(sample):
    """Turn 3까지 히스토리 준비 (이미 7턴으로 정리됨)"""
    history = []
    
    # prefix_dialog는 이미 7턴으로 정리됨
    for turn in sample['prefix_dialog']:
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    
    return history


def print_section_header(title, char="="):
    """섹션 헤더 출력"""
    print("\n" + char*80)
    print(f"{title:^80}")
    print(char*80)


def print_box(title, content_lines, width=78):
    """박스 형태로 내용 출력"""
    print(f"\n+-- {title} " + "-"*(width - len(title) - 5) + "+")
    for line in content_lines:
        if line:  # 빈 줄이 아닌 경우만
            print(f"| {line:<{width-2}} |")
    print("+" + "-"*width + "+")


def test_single_session(session_index=0):
    """
    단일 세션 테스트 (확장 가능)
    
    Args:
        session_index: 테스트할 샘플 인덱스 (0부터 시작)
        
    Returns:
        dict: 테스트 결과 로그 데이터
    """
    print_section_header(f"AGENT PIPELINE TEST - Session #{session_index}")
    
    # 샘플 로드
    sample = load_gold_sample(index=session_index)
    situation = sample['situation']
    history = prepare_dialog_history(sample)
    gold_label = sample.get('gold_label', 'Unknown')
    gold_response = sample.get('gold_turn4', 'N/A')
    
    # Sample Info
    print_box("SAMPLE INFO", [
        f"Session ID : {sample['esconv_session_id']}",
        f"Situation  : {situation[:65]}...",
        f"Gold Label : {gold_label}"
    ])
    
    # Dialog History
    print(f"\n+-- DIALOG HISTORY ({len(history)} turns) " + "-"*47 + "+")
    for i, turn in enumerate(history, 1):
        speaker_icon = "USER" if turn['speaker'] == 'seeker' else "ASST"
        text = turn['text'][:60]
        padding = 78 - len(f"Turn {i} [{speaker_icon}]: {text}...")
        print(f"| Turn {i} [{speaker_icon}]: {text}..." + " "*padding + " |")
    print("+" + "-"*78 + "+")
    
    # Gold Response
    gold_lines = []
    for i in range(0, len(gold_response), 74):
        chunk = gold_response[i:i+74]
        gold_lines.append(chunk)
    print_box("GOLD RESPONSE (Turn 4)", gold_lines)
    
    # ================================================================================
    # STEP 1: Initialize
    # ================================================================================
    print_section_header("STEP 1: INITIALIZATION", "-")
    
    client = OpenAIClient(model="gpt-4o-mini", max_tokens=200)
    
    context_builder = ContextBuilder(
        llm_client=client,
        window_size=6,
        use_summary=False
    )
    
    generator = CandidateGenerator(
        llm_client=client,
        num_candidates=1,
        temperature=0.7
    )
    
    model_path = Path(__file__).parent.parent.parent / "models" / "best_model"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    detector = ViolationDetector(
        mode="model",
        model_path=str(model_path),
        temperature=1.0  # TODO: Fit 후 최적 T로 변경
    )
    
    controller = ControllerV2(llm_client=client)
    
    print("SUCCESS: Components initialized")
    print("  - LLM: gpt-4o-mini")
    print("  - Detector: RoBERTa-base (6 labels)")
    print("  - Candidates: 1")
    
    # ================================================================================
    # STEP 2: Build Context
    # ================================================================================
    print_section_header("STEP 2: BUILD CONTEXT", "-")
    context = context_builder.build_context(history, situation)
    print(f"SUCCESS: Context built with {len(context['recent_turns'])} recent turns\n")
    
    # 마지막 3턴 출력
    print("+-- LAST 3 TURNS IN CONTEXT ----------------------------------------------+")
    last_3 = context['recent_turns'][-3:] if len(context['recent_turns']) >= 3 else context['recent_turns']
    for i, turn in enumerate(last_3, start=len(context['recent_turns'])-len(last_3)+1):
        speaker_label = "USER" if turn['speaker'] == 'seeker' else "ASST"
        content_preview = turn.get('content') or turn.get('text', '')
        if len(content_preview) > 70:
            content_preview = content_preview[:70] + "..."
        print(f"| Turn {i} [{speaker_label}]: {content_preview:<68} |")
    print("+------------------------------------------------------------------------------+\n")
    
    # ================================================================================
    # STEP 3: Generate Candidates
    # ================================================================================
    print_section_header("STEP 3: GENERATE CANDIDATES", "-")
    candidates = generator.generate_candidates(context)
    print(f"SUCCESS: Generated {len(candidates)} candidate\n")
    
    for i, cand in enumerate(candidates):
        cand_lines = []
        for j in range(0, len(cand['text']), 74):
            cand_lines.append(cand['text'][j:j+74])
        print_box(f"Candidate #{i+1}", cand_lines)
    
    # ================================================================================
    # STEP 4: Detect Violations
    # ================================================================================
    print_section_header("STEP 4: DETECT VIOLATIONS", "-")
    detections = detector.detect_batch(context, candidates)
    
    for i, det in enumerate(detections):
        label = det['top_violation']
        conf = det['confidence']
        severity = det['severity']
        print(f"Candidate #{i+1}: {label.upper()} (confidence: {conf:.2%}, severity: {severity})")
    
    # ================================================================================
    # STEP 5: Controller & Rewrite Loop
    # ================================================================================
    print_section_header("STEP 5: CONTROLLER & REWRITE LOOP", "-")
    
    # 가장 안전한 후보 선택
    # NOTE: 현재는 num_candidates=1이라 의미 없음
    # 여러 후보 생성 시(예: temperature sampling으로 3개 생성)
    # severity가 낮은(=안전한) 후보를 우선 선택하기 위한 로직
    best_idx = min(range(len(detections)), key=lambda i: detections[i]['severity'])
    best_candidate = candidates[best_idx]
    best_detection = detections[best_idx]
    
    print(f"\nInitial Selection: Candidate #{best_idx+1} -> {best_detection['top_violation'].upper()}")
    print(f"(Note: Currently only 1 candidate, severity check is for multi-candidate mode)\n")
    
    # Rewrite loop
    max_retries = 2
    retry_count = 0
    final_response = best_candidate['text']
    final_label = best_detection['top_violation']
    
    # Rewrite 루프
    retry_count = 0
    rewrite_log = []  # JSON 로그용
    
    while retry_count < max_retries:
        decision = controller.decide({'label': final_label, 'confidence': best_detection['confidence']})
        
        # Accept이면 rewrite 없이 종료 (로그에 추가하지 않음)
        if decision['type'] == 'accept':
            print(f"DECISION: ACCEPT (Response is NORMAL)\n")
            break
        
        # Rewrite가 필요한 경우만 아래 진행
        print(f"+-- RETRY #{retry_count + 1} / {max_retries} - Rewriting {decision['violation'].upper()} " + "-"*30 + "+")
        
        # Rewrite
        rewrite_result = controller.rewrite(
            original=final_response,
            violation_type=decision['violation'],
            context=context
        )
        
        # rewrite_result는 dict: {text, reasoning, raw}
        if isinstance(rewrite_result, dict):
            rewritten = rewrite_result.get("text", final_response)
            reasoning = rewrite_result.get("reasoning", "")
        else:
            rewritten = str(rewrite_result)
            reasoning = ""
        
        print("|")
        print("| ORIGINAL:")
        for i in range(0, len(final_response), 72):
            print(f"|    {final_response[i:i+72]}")
        print("|")
        
        if reasoning:
            print("| REASONING:")
            for i in range(0, len(reasoning), 72):
                print(f"|    {reasoning[i:i+72]}")
            print("|")
        
        print("| REWRITTEN:")
        for i in range(0, len(rewritten), 72):
            print(f"|    {rewritten[i:i+72]}")
        print("|")
        
        # 실제로 변경되었는지 확인
        if rewritten == final_response:
            print("| WARNING: Rewrite returned SAME text!")
        
        # Re-check
        recheck_candidate = {'id': 999, 'text': rewritten}
        recheck_detection = detector.detect(context, recheck_candidate)
        
        print(f"| RE-CHECK: {recheck_detection['top_violation'].upper()} (confidence: {recheck_detection['confidence']:.2%}, severity: {recheck_detection['severity']})")
        print("+" + "-"*78 + "+\n")
        
        # 로그 저장
        rewrite_log.append({
            "retry": retry_count + 1,
            "violation": decision['violation'],
            "original": final_response,
            "reasoning": reasoning,
            "rewritten": rewritten,
            "recheck_label": recheck_detection['top_violation'],
            "recheck_confidence": recheck_detection['confidence'],
            "recheck_all_probabilities": recheck_detection.get('all_probabilities', {})  # 추가: 재검 확률 분포
        })
        
        final_response = rewritten
        final_label = recheck_detection['top_violation']
        best_detection = recheck_detection
        retry_count += 1
        
        if final_label == 'normal':
            print(f"SUCCESS: Converted to NORMAL after {retry_count} retry(s)\n")
            break
    
    # Fail-safe if still violated
    if final_label != 'normal' and retry_count >= max_retries:
        print(f"+-- FAIL-SAFE ACTIVATED " + "-"*55 + "+")
        print(f"| Max retries ({max_retries}) reached, violation still detected" + " "*15 + "|")
        final_response = controller.failsafe(final_label.upper(), context)
        final_label = 'failsafe'
        print(f"| Fail-safe template: {final_response[:50]}..." + " "*7 + "|")
        print("+" + "-"*78 + "+\n")
    
    final_result_lines = [
        f"Label   : {final_label.upper()}",
        f"Retries : {retry_count}",
        "Response:"
    ]
    for i in range(0, len(final_response), 72):
        final_result_lines.append("  " + final_response[i:i+72])
    print_box("FINAL RESULT", final_result_lines)
    
    # ================================================================================
    # JSON 로그 저장
    # ================================================================================
    
    log_data = {
        "session_id": sample['esconv_session_id'],
        "gold_label": gold_label,
        "gold_turn4": gold_response,
        "dialog_history": context['recent_turns'],  # 추가: 전체 대화 이력
        "context_turns": len(context['recent_turns']),
        "initial_generation": {
            "text": candidates[0]['text'],
            "detected_label": detections[0]['top_violation'],
            "confidence": detections[0]['confidence'],
            "all_probabilities": detections[0].get('all_probabilities', {})  # 추가: 전체 확률 분포
        },
        "rewrite_iterations": rewrite_log,
        "final_result": {
            "label": final_label,
            "response": final_response,
            "total_retries": retry_count
        }
    }
    
    log_path = Path("src/agent/test_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[LOG] JSON log saved to: {log_path}\n")
    
    # ================================================================================
    # FINAL COMPARISON
    # ================================================================================
    print_section_header("FINAL COMPARISON")
    
    print("\n+-- AGENT OUTPUT " + "-"*61 + "+")
    print(f"| Label   : {final_label.upper():<67} |")
    print(f"| Retries : {retry_count:<67} |")
    print("| Response:" + " "*68 + "|")
    for i in range(0, len(final_response), 72):
        chunk = final_response[i:i+72]
        print(f"|    {chunk:<74} |")
    print("+" + "-"*78 + "+")
    
    print("\n+-- GOLD OUTPUT " + "-"*62 + "+")
    print(f"| Label   : {gold_label:<67} |")
    print("| Response:" + " "*68 + "|")
    for i in range(0, len(gold_response), 72):
        chunk = gold_response[i:i+72]
        print(f"|    {chunk:<74} |")
    print("+" + "-"*78 + "+")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    
    return log_data


def main():
    """단일 세션 테스트 (인덱스 0)"""
    test_single_session(session_index=0)


def batch_test(num_samples=5, random_seed=42):
    """여러 샘플 배치 테스트"""
    import random
    random.seed(random_seed)
    
    # Load all samples
    test_path = Path("src/agent/test_gold_300_prefix.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_samples = data['samples']  # Extract samples array
    
    total_samples = len(all_samples)
    print(f"\n{'='*80}")
    print(f"BATCH TEST: {num_samples} random samples from {total_samples} total")
    print(f"{'='*80}\n")
    
    # Select random indices
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    results_summary = []
    log_dir = Path("src/agent/batch_logs")
    log_dir.mkdir(exist_ok=True)
    
    for i, idx in enumerate(indices, 1):
        print(f"\n{'#'*80}")
        print(f"# TEST {i}/{num_samples}: Sample Index {idx}")
        print(f"{'#'*80}\n")
        
        # Run test
        sample = all_samples[idx]
        session_id = sample['esconv_session_id']  # Updated key
        gold_label = sample['primary_label']  # Updated key
        
        # Initialize components (same as main())
        llm_client = OpenAIClient(model="gpt-4o-mini")
        context_builder = ContextBuilder(llm_client=llm_client, max_tokens=512)
        candidate_generator = CandidateGenerator(llm_client)
        violation_detector = ViolationDetector(model_name="th610/empathetic-roberta-base-6class")
        controller = ControllerV2(
            candidate_generator=candidate_generator,
            violation_detector=violation_detector
        )
        
        # Build context
        history = sample['prefix_dialog']  # Updated key
        context = context_builder.build(history)
        
        # Initial generation
        print_section_header("INITIAL GENERATION")
        candidate = candidate_generator.generate(context, num_candidates=1)[0]
        detection = violation_detector.detect(context, candidate)
        
        initial_label = detection['label']
        initial_conf = detection['confidence']
        
        print(f"\n[CANDIDATE] {candidate['text'][:100]}...")
        print(f"\n[DETECTION] Label: {initial_label} | Confidence: {initial_conf:.2%}")
        print(f"[ALL PROBS] {format_probabilities(detection['all_probabilities'])}")
        
        # Decision & retries
        decision = controller.decide(detection)
        retry_count = 0
        rewrite_iterations = []
        
        if decision['type'] == 'rewrite':
            print(f"\n[DECISION] REWRITE needed for {decision['violation']}")
            
            for retry_num in range(1, 3):
                print(f"\n{'─'*80}")
                print(f"RETRY {retry_num}")
                print(f"{'─'*80}")
                
                rewrite_result = controller.rewrite(
                    context=context,
                    original_response=candidate['text'],
                    violation_type=decision['violation']
                )
                
                print(f"\n[REASONING] {rewrite_result['reasoning'][:150]}...")
                print(f"\n[REWRITTEN] {rewrite_result['response'][:100]}...")
                
                recheck = violation_detector.detect(context, {'text': rewrite_result['response']})
                print(f"\n[RE-CHECK] Label: {recheck['label']} | Confidence: {recheck['confidence']:.2%}")
                print(f"[ALL PROBS] {format_probabilities(recheck['all_probabilities'])}")
                
                rewrite_iterations.append({
                    'retry': retry_num,
                    'violation': decision['violation'],
                    'original': candidate['text'],
                    'reasoning': rewrite_result['reasoning'],
                    'rewritten': rewrite_result['response'],
                    'recheck_label': recheck['label'],
                    'recheck_confidence': float(recheck['confidence']),
                    'recheck_all_probabilities': {k: float(v) for k, v in recheck['all_probabilities'].items()}
                })
                
                retry_count += 1
                candidate['text'] = rewrite_result['response']
                
                new_decision = controller.decide(recheck)
                if new_decision['type'] == 'accept':
                    detection = recheck
                    print(f"\n[SUCCESS] Violation resolved after {retry_num} retry(s)")
                    break
                elif retry_num == 2:
                    print(f"\n[FAIL-SAFE] Max retries reached")
                    detection = recheck
                    break
                else:
                    decision = new_decision
        else:
            print(f"\n[DECISION] ACCEPT (confidence: {initial_conf:.2%})")
        
        final_label = detection['label']
        final_response = candidate['text']
        
        # Save individual log
        log_data = {
            'session_id': session_id,
            'sample_index': idx,
            'gold_label': gold_label,
            'dialog_history': [
                {'speaker': turn['speaker'], 'text': turn['text']}
                for turn in history
            ],
            'all_probabilities': {k: float(v) for k, v in detection['all_probabilities'].items()},
            'initial_generation': {
                'text': candidate['text'],  # Current generated text
                'label': initial_label,
                'confidence': float(initial_conf),
                'all_probabilities': {k: float(v) for k, v in detection['all_probabilities'].items()}
            },
            'rewrite_iterations': rewrite_iterations,
            'final_result': {
                'label': final_label,
                'response': final_response,
                'total_retries': retry_count
            }
        }
        
        log_path = log_dir / f"test_log_{session_id}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[LOG] Saved to: {log_path}")
        
        # Summary
        match = "✓" if final_label == gold_label else "✗"
        results_summary.append({
            'index': idx,
            'session_id': session_id,
            'gold': gold_label,
            'predicted': final_label,
            'match': match,
            'retries': retry_count,
            'initial_conf': initial_conf,
            'final_conf': detection['confidence']
        })
        
        print(f"\n[RESULT] Gold: {gold_label} | Predicted: {final_label} | Match: {match}")
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Index':<8} {'Session':<10} {'Gold':<10} {'Predicted':<10} {'Match':<7} {'Retries':<8} {'Init%':<8} {'Final%':<8}")
    print("-" * 80)
    
    for r in results_summary:
        print(f"{r['index']:<8} {r['session_id']:<10} {r['gold']:<10} {r['predicted']:<10} {r['match']:<7} {r['retries']:<8} {r['initial_conf']*100:<8.1f} {r['final_conf']*100:<8.1f}")
    
    # Statistics
    total = len(results_summary)
    matches = sum(1 for r in results_summary if r['match'] == '✓')
    accuracy = matches / total if total > 0 else 0
    avg_retries = sum(r['retries'] for r in results_summary) / total if total > 0 else 0
    
    print("\n" + "-" * 80)
    print(f"Accuracy: {matches}/{total} ({accuracy:.1%})")
    print(f"Avg Retries: {avg_retries:.2f}")
    print(f"Logs saved to: {log_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # Batch mode: python test_full_pipeline.py batch [num_samples]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        batch_test(num_samples=num_samples)
    else:
        # Single test mode (default)
        main()
