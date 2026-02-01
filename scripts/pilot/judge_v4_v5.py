"""
Judge V4-V5 Generated Data

V4-V5 생성된 데이터 150개씩을 Judge로 품질 평가
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt


def build_judge_input(session: Dict) -> str:
    """전체 대화를 Judge 입력으로 구성"""
    
    situation = session.get('situation', '')
    dialog = session.get('dialog', [])
    
    # 전체 대화 구성
    dialog_lines = [f"[상황]\n{situation}\n"]
    dialog_lines.append("[전체 대화]")
    
    for i, turn in enumerate(dialog):
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', turn.get('text', ''))
        # 마지막 턴 표시
        marker = " ← 평가 대상" if i == len(dialog) - 1 else ""
        dialog_lines.append(f"[{speaker.upper()}] {content}{marker}")
    
    full_dialog_text = "\n".join(dialog_lines)
    
    return full_dialog_text


def judge_session(llm_client: OpenAIClient, session: Dict, expected_label: str) -> Dict:
    """개별 세션 평가"""
    
    session_id = session.get('session_id', 'unknown')
    
    try:
        # 전체 대화 생성
        full_dialog_text = build_judge_input(session)
        
        # prompts.py의 build_judge_prompt 사용
        user_prompt = build_judge_prompt(full_dialog=full_dialog_text)
        
        # LLM 호출 (call은 이미 JSON dict 반환)
        judge_result = llm_client.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt
        )
        
        predicted_label = judge_result.get('label', 'unknown')
        correct = (predicted_label.lower() == expected_label.lower())
        
        return {
            'session_id': session_id,
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': correct,
            'reasoning': judge_result.get('reasoning', ''),
            'violations': judge_result.get('violations', [])
        }
        
    except Exception as e:
        return {
            'session_id': session_id,
            'expected': expected_label,
            'predicted': 'error',
            'correct': False,
            'error': str(e)
        }


def main():
    print("="*80)
    print("V4-V5 Judge Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # LLM client
    llm_client = OpenAIClient(model="gpt-4o-mini")
    
    # Load data
    data_dir = project_root / "data" / "pilot"
    v4_file = data_dir / "v4_full_150.json"
    v5_file = data_dir / "v5_full_150.json"
    
    print(f"📂 Loading V4: {v4_file}")
    with open(v4_file, 'r', encoding='utf-8') as f:
        v4_data = json.load(f)
    print(f"   Loaded: {len(v4_data)} sessions")
    
    print(f"📂 Loading V5: {v5_file}")
    with open(v5_file, 'r', encoding='utf-8') as f:
        v5_data = json.load(f)
    print(f"   Loaded: {len(v5_data)} sessions")
    print()
    
    # Judge V4
    print("="*80)
    print("Judging V4 (Reality Distortion)")
    print("="*80)
    v4_results = []
    v4_correct = 0
    
    for i, session in enumerate(v4_data):
        print(f"[{i+1}/{len(v4_data)}] {session.get('session_id', 'unknown')}", end='')
        result = judge_session(llm_client, session, 'v4')
        v4_results.append(result)
        
        if result['correct']:
            v4_correct += 1
            print(" ✅")
        else:
            print(f" ❌ (predicted: {result['predicted']})")
        
        # Progress every 10
        if (i+1) % 10 == 0:
            acc = v4_correct / (i+1) * 100
            print(f"   Progress: {i+1}/{len(v4_data)}, Accuracy: {acc:.1f}%")
    
    v4_accuracy = v4_correct / len(v4_data) * 100
    print()
    print(f"V4 Accuracy: {v4_correct}/{len(v4_data)} = {v4_accuracy:.1f}%")
    print()
    
    # Judge V5
    print("="*80)
    print("Judging V5 (Crisis Safety Failure)")
    print("="*80)
    v5_results = []
    v5_correct = 0
    
    for i, session in enumerate(v5_data):
        print(f"[{i+1}/{len(v5_data)}] {session.get('session_id', 'unknown')}", end='')
        result = judge_session(llm_client, session, 'v5')
        v5_results.append(result)
        
        if result['correct']:
            v5_correct += 1
            print(" ✅")
        else:
            print(f" ❌ (predicted: {result['predicted']})")
        
        # Progress every 10
        if (i+1) % 10 == 0:
            acc = v5_correct / (i+1) * 100
            print(f"   Progress: {i+1}/{len(v5_data)}, Accuracy: {acc:.1f}%")
    
    v5_accuracy = v5_correct / len(v5_data) * 100
    print()
    print(f"V5 Accuracy: {v5_correct}/{len(v5_data)} = {v5_accuracy:.1f}%")
    print()
    
    # Overall summary
    total_correct = v4_correct + v5_correct
    total_sessions = len(v4_data) + len(v5_data)
    overall_accuracy = total_correct / total_sessions * 100
    
    print("="*80)
    print("Overall Summary")
    print("="*80)
    print(f"V4 (Reality Distortion):     {v4_correct}/{len(v4_data)} = {v4_accuracy:.1f}%")
    print(f"V5 (Crisis Safety Failure):  {v5_correct}/{len(v5_data)} = {v5_accuracy:.1f}%")
    print(f"Overall:                     {total_correct}/{total_sessions} = {overall_accuracy:.1f}%")
    print()
    
    # Save results
    output_file = data_dir / "judge_v4_v5_results.json"
    results = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'v4_accuracy': v4_accuracy,
            'v4_correct': v4_correct,
            'v4_total': len(v4_data),
            'v5_accuracy': v5_accuracy,
            'v5_correct': v5_correct,
            'v5_total': len(v5_data),
            'overall_accuracy': overall_accuracy,
            'overall_correct': total_correct,
            'overall_total': total_sessions
        },
        'v4_results': v4_results,
        'v5_results': v5_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved: {output_file}")
    print()
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
