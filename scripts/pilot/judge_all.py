"""
Judge All Generated Data (Normal + V1-V5)

각 클래스별 생성 데이터를 Judge로 품질 평가
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


def judge_class(llm_client: OpenAIClient, data: List[Dict], class_label: str) -> tuple:
    """특정 클래스 데이터 전체 평가"""
    
    print("="*80)
    print(f"Judging {class_label.upper()} ({len(data)} sessions)")
    print("="*80)
    
    results = []
    correct_count = 0
    
    for i, session in enumerate(data):
        session_id = session.get('session_id', f'{class_label}_{i:04d}')
        print(f"[{i+1}/{len(data)}] {session_id}", end='')
        
        result = judge_session(llm_client, session, class_label)
        results.append(result)
        
        if result['correct']:
            correct_count += 1
            print(" ✅")
        else:
            print(f" ❌ (predicted: {result['predicted']})")
        
        # Progress every 20
        if (i+1) % 20 == 0:
            acc = correct_count / (i+1) * 100
            print(f"   Progress: {i+1}/{len(data)}, Accuracy: {acc:.1f}%")
    
    accuracy = correct_count / len(data) * 100 if data else 0
    print()
    print(f"{class_label.upper()} Accuracy: {correct_count}/{len(data)} = {accuracy:.1f}%")
    print()
    
    return results, correct_count, accuracy


def main():
    print("="*80)
    print("Judge All Generated Data")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # LLM client
    llm_client = OpenAIClient(model="gpt-4o-mini")
    
    # Data files
    data_files = [
        ('data/generated/normal_400.json', 'normal'),
        ('data/generated/v1_240.json', 'v1'),
        ('data/generated/v2_160.json', 'v2'),
        ('data/generated/v3_200.json', 'v3'),
        ('data/pilot/v4_full_150.json', 'v4'),
        ('data/pilot/v5_full_150.json', 'v5')
    ]
    
    all_results = {}
    summary = {}
    total_correct = 0
    total_sessions = 0
    
    for filepath, class_label in data_files:
        print(f"📂 Loading {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Loaded: {len(data)} sessions")
        print()
        
        # Judge this class
        results, correct, accuracy = judge_class(llm_client, data, class_label)
        
        all_results[class_label] = results
        summary[class_label] = {
            'total': len(data),
            'correct': correct,
            'accuracy': accuracy
        }
        
        total_correct += correct
        total_sessions += len(data)
    
    # Overall summary
    overall_accuracy = total_correct / total_sessions * 100 if total_sessions else 0
    
    print("="*80)
    print("Overall Summary")
    print("="*80)
    for class_label, stats in summary.items():
        print(f"{class_label.upper():8s}: {stats['correct']:3d}/{stats['total']:3d} = {stats['accuracy']:5.1f}%")
    print("-"*80)
    print(f"Overall:  {total_correct:3d}/{total_sessions:3d} = {overall_accuracy:5.1f}%")
    print("="*80)
    print()
    
    # Save results
    output_dir = Path("data/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "judge_all_results.json"
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'overall_accuracy': overall_accuracy,
            'overall_correct': total_correct,
            'overall_total': total_sessions,
            'by_class': summary
        },
        'results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved: {output_file}")
    print()
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
