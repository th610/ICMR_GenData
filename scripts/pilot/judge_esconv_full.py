"""
Judge all 1300 ESConv sessions with checkpoint saving
Output: Original session + judge fields added
"""
import json
import random
from pathlib import Path
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt
from src.utils import load_json, save_json


def extract_window(dialog, min_turns=13, max_turns=20):
    """Extract window from dialog (same as judge_esconv_window.py)"""
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
    print("=" * 80)
    print("ESConv 전체 데이터 Judge 평가 (1300개)")
    print("방식: 윈도우 (13-20턴), 매 100개 체크포인트 저장")
    print("=" * 80)
    
    # Load ESConv
    esconv_path = "ESConv.json"
    print(f"\n📂 {esconv_path} 로드 중...")
    esconv_sessions = load_json(esconv_path)
    total = len(esconv_sessions)
    print(f"   총 세션: {total}개")
    
    # Initialize Judge
    print(f"\n🔧 LLM Judge 초기화...")
    llm_client = OpenAIClient(model="gpt-4o-mini", temperature=0.3)
    
    # Output directory
    output_dir = Path("data/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all sessions
    results = []
    stats = {"Normal": 0, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0}
    skip_count = 0
    error_count = 0
    checkpoint_interval = 100
    progress_interval = 10
    
    print(f"\n🔍 Judge 평가 시작...")
    print("=" * 80)
    
    for idx, session in enumerate(esconv_sessions):
        # Extract window
        dialog = session.get('dialog', [])
        window_dialog, window_turns = extract_window(dialog)
        
        if window_dialog is None:
            # Skip if cannot extract valid window
            skip_count += 1
            continue
        
        # Judge
        try:
            # Build dialog text
            dialog_lines = []
            for i, turn in enumerate(window_dialog):
                speaker = turn.get('speaker', 'unknown')
                content = turn.get('content', '')
                dialog_lines.append(f"[Turn {i}] {speaker.capitalize()}: {content}")
            
            dialog_text = "\n".join(dialog_lines)
            full_dialog = f"[Situation]\n{session['situation']}\n\n[Dialogue]\n{dialog_text}"
            
            # Call Judge
            user_prompt = build_judge_prompt(full_dialog)
            result = llm_client.call(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=user_prompt
            )
            
            label = result.get('label', 'Error')
            confidence = result.get('confidence', 'low')
            reason = result.get('reason', '')
            
            stats[label] += 1
            
        except Exception as e:
            error_count += 1
            label = "Error"
            confidence = "low"
            reason = str(e)
        
        # Build result (original session + judge fields)
        result_session = session.copy()
        result_session['judge_label'] = label
        result_session['judge_confidence'] = confidence
        result_session['judge_reason'] = reason
        result_session['window_turns'] = window_turns
        result_session['total_turns'] = len(dialog)
        result_session['session_index'] = idx
        
        results.append(result_session)
        
        # Progress display every 10
        if (idx + 1) % progress_interval == 0:
            judged = len(results)
            pct = (idx + 1) / total * 100
            normal_pct = stats["Normal"] / judged * 100 if judged > 0 else 0
            v_total = sum(v for k, v in stats.items() if k != "Normal")
            print(f"[{idx+1}/{total}] {pct:5.1f}% | Judged: {judged}, Skip: {skip_count} | "
                  f"Normal: {stats['Normal']} ({normal_pct:.0f}%), Violations: {v_total}")
        
        # Checkpoint save every 100
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"judge_esconv_checkpoint_{idx+1:04d}.json"
            save_json(results, str(checkpoint_path))
            
            # Print detailed progress
            judged = len(results)
            normal_pct = stats["Normal"] / judged * 100 if judged > 0 else 0
            
            print(f"\n{'─'*80}")
            print(f"📍 Checkpoint {idx+1}/{total}")
            print(f"   Judged: {judged}, Skip: {skip_count}, Error: {error_count}")
            print(f"   Normal: {stats['Normal']} ({normal_pct:.1f}%)")
            print(f"   V1: {stats['V1']}, V2: {stats['V2']}, V3: {stats['V3']}, "
                  f"V4: {stats['V4']}, V5: {stats['V5']}")
            print(f"   💾 Saved: {checkpoint_path.name}")
            print(f"{'─'*80}\n")
    
    # Final save
    final_path = output_dir / "judge_esconv_full_1300.json"
    save_json(results, str(final_path))
    
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)
    
    total_judged = len(results)
    print(f"\n총 처리: {total}/{total}개")
    print(f"  평가 완료: {total_judged}개")
    print(f"  Skip: {skip_count}개 (대화 짧거나 supporter 없음)")
    print(f"  Error: {error_count}개")
    
    print(f"\n레이블 분포:")
    for label in ["Normal", "V1", "V2", "V3", "V4", "V5"]:
        count = stats[label]
        pct = count / total_judged * 100 if total_judged > 0 else 0
        print(f"  {label:8s} {count:4d}  ({pct:5.1f}%)")
    
    print(f"\n💾 최종 결과 저장: {final_path}")
    print("\n✅ 완료!")


if __name__ == '__main__':
    main()
