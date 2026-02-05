"""
V10 프롬프트 테스트 (V8 + STATE MEMORY)
ESConv 100개 샘플
"""
import json
import random
from pathlib import Path
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_v10 import JUDGE_SYSTEM, build_judge_prompt
from src.llm.state_memory import STATE_MEMORY_SYSTEM, build_state_memory_prompt
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


def generate_state_memory(situation, dialog_history, llm_client):
    """Generate STATE MEMORY (FACTS ONLY)"""
    try:
        prompt = build_state_memory_prompt(situation, dialog_history)
        result = llm_client.call(STATE_MEMORY_SYSTEM, prompt)
        state_memory = result.get('state_memory', '')
        return state_memory if state_memory else ""
    except Exception as e:
        print(f"  Warning: STATE MEMORY generation failed: {e}")
        return ""


def main():
    print("=" * 60)
    print("V10 프롬프트 테스트 (V8 + STATE MEMORY)")
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
        
        # Generate STATE MEMORY for turns before last 7
        # WINDOW(6) + TARGET(1) = 7턴
        if len(window_dialog) > 7:
            state_memory = generate_state_memory(
                situation, 
                window_dialog[:-7], 
                judge
            )
        else:
            state_memory = ""  # 7턴 이하면 히스토리 없음
        
        # Build prompt (최근 7턴만 사용)
        recent_dialog = window_dialog[-7:] if len(window_dialog) >= 7 else window_dialog
        prompt = build_judge_prompt(recent_dialog, summary=state_memory)
        
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
                "state_memory": state_memory,
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
    print("📊 V10 결과 요약")
    print("=" * 60)
    total_evaluated = len(results)
    for label, count in sorted(stats.items()):
        if count > 0:
            pct = (count / total_evaluated) * 100
            print(f"{label.upper()}: {count}개 ({pct:.1f}%)")
    
    # Save results
    output_path = "test_judge_v10_100.json"
    save_json(results, output_path)
    print(f"\n💾 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
