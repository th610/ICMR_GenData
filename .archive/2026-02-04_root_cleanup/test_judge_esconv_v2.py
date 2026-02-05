"""
ESConv 100개 샘플 Judge 테스트 - V2 PROMPT 버전
83% Normal 나왔던 중간 버전 프롬프트로 테스트
"""
import json
import random
from pathlib import Path
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v2 import JUDGE_SYSTEM, build_judge_prompt
from src.utils import load_json, save_json


def extract_window(dialog, min_turns=13, max_turns=20):
    """Extract window from dialog"""
    if len(dialog) < min_turns:
        return None, None
    
    max_possible = min(max_turns, len(dialog))
    possible_lengths = list(range(min_turns, max_possible + 1))
    
    # Filter lengths where last turn is supporter
    valid_lengths = []
    for length in possible_lengths:
        if dialog[length - 1]['speaker'] == 'supporter':
            valid_lengths.append(length)
    
    if not valid_lengths:
        return None, None
    
    window_length = random.choice(valid_lengths)
    window_dialog = dialog[:window_length]
    
    return window_dialog, window_length


def main():
    print("=" * 60)
    print("ESConv 100개 샘플 Judge 테스트 - V2 PROMPT (83% Normal)")
    print("=" * 60)
    
    # Load ESConv
    esconv_path = "ESConv.json"
    print(f"\n📂 {esconv_path} 로드 중...")
    esconv_sessions = load_json(esconv_path)
    
    # Use same random seed for reproducibility
    random.seed(42)
    
    # Sample 100 sessions
    sample_sessions = random.sample(esconv_sessions, 100)
    print(f"✅ 100개 샘플 선택 완료 (seed=42)")
    
    # Initialize Judge
    print(f"\n🔧 LLM Judge 초기화...")
    judge = OpenAIClient()
    
    results = []
    stats = {"normal": 0, "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0}
    
    for idx, session in enumerate(sample_sessions, 1):
        dialog = session['dialog']
        situation = session.get('situation', 'unknown')
        
        # Extract window
        window_dialog, window_length = extract_window(dialog)
        
        if window_dialog is None:
            print(f"[{idx}/100] ⚠️  대화 너무 짧음 (< 13턴), 스킵")
            continue
        
        # Build prompt (V2 format: full dialog only)
        prompt = build_judge_prompt(window_dialog)
        
        # Judge
        print(f"[{idx}/100] 🔍 평가 중... (윈도우 길이: {window_length}턴)", end=" ")
        
        try:
            result = judge.call(JUDGE_SYSTEM, prompt)
            
            label = result.get("label", "unknown").lower()
            reason = result.get("reason", "")
            confidence = result.get("confidence", "unknown")
            
            print(f"→ {label.upper()}")
            if reason:
                print(f"     이유: {reason[:100]}...")
            
            # Update stats
            stats[label] = stats.get(label, 0) + 1
            
            # Save result
            results.append({
                "session_id": f"sample_{idx}",
                "situation": situation,
                "window_length": window_length,
                "dialog": window_dialog,
                "label": label,
                "reason": reason,
                "confidence": confidence
            })
            
        except Exception as e:
            print(f"❌ 에러: {e}")
            continue
    
    # Save results
    output_path = "test_judge_v2_100.json"
    save_json(results, output_path)
    print(f"\n✅ 결과 저장: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📊 결과 요약")
    print("=" * 60)
    total = sum(stats.values())
    for label, count in sorted(stats.items()):
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{label.upper()}: {count}개 ({percentage:.1f}%)")
    
    # Confidence distribution
    print("\n" + "=" * 60)
    print("📊 Confidence 분포")
    print("=" * 60)
    conf_stats = {}
    for r in results:
        conf = r.get("confidence", "unknown")
        conf_stats[conf] = conf_stats.get(conf, 0) + 1
    for conf, count in sorted(conf_stats.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{conf}: {count}개 ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
