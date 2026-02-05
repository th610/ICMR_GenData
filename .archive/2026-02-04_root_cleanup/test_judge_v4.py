"""
V4 프롬프트 테스트 (NEW 기반 + 전체 대화 + 강화된 V2 경계)
ESConv 100개 샘플
"""
import json
import random
from pathlib import Path
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v4 import JUDGE_SYSTEM, build_judge_prompt
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
    print("V4 프롬프트 테스트 (NEW + 전체 대화 + 강화된 V2)")
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
        
        # Build prompt
        prompt = build_judge_prompt(window_dialog)
        
        # Judge
        print(f"[{idx}/100] 🔍 평가 중... (윈도우 길이: {window_length}턴)", end=" ")
        
        try:
            result = judge.call(JUDGE_SYSTEM, prompt)
            label = result.get("label", "unknown").lower()
            reason = result.get("reason", "")
            evidence = result.get("evidence", "N/A")
            
            print(f"→ {label.upper()}")
            if reason:
                print(f"     이유: {reason[:100]}...")
            
            # Save result
            results.append({
                "session_id": f"sample_{idx}",
                "situation": situation,
                "window_length": window_length,
                "dialog": window_dialog,
                "label": label,
                "reason": reason,
                "evidence": evidence,
                "confidence": result.get("confidence", "unknown")
            })
            
            stats[label] = stats.get(label, 0) + 1
            
        except Exception as e:
            print(f"→ ❌ 오류: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 V4 결과 요약")
    print("=" * 60)
    total_evaluated = len(results)
    for label, count in sorted(stats.items()):
        if count > 0:
            pct = (count / total_evaluated) * 100
            print(f"{label.upper()}: {count}개 ({pct:.1f}%)")
    
    # Save results
    output_path = "test_judge_v4_100.json"
    save_json(results, output_path)
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
