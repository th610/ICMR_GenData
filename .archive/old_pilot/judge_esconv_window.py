"""
ESConv 원본 데이터에서 윈도우 추출 후 Judge 평가
전체 대화가 아닌 13~20턴 랜덤 구간을 추출하여 평가
마지막 발화는 반드시 supporter여야 함
"""

import json
import random
from pathlib import Path
import sys
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt


def extract_window(dialog: list, min_turns: int = 13, max_turns: int = 20):
    """대화 처음부터 13~20턴 추출, 마지막은 반드시 supporter"""
    
    total_turns = len(dialog)
    
    # 최소 턴 수 확인
    if total_turns < min_turns:
        return None
    
    # 목표 윈도우 크기 결정 (13~20 사이)
    target_window_size = random.randint(min_turns, min(max_turns, total_turns))
    
    # 처음부터 target_window_size만큼 자르기
    window = dialog[:target_window_size]
    
    # 마지막이 supporter인지 확인
    if window and window[-1].get('speaker') == 'supporter':
        return window
    
    # supporter가 아니면 앞에서부터 supporter 찾기
    for i in range(min_turns - 1, min(max_turns, total_turns)):
        if dialog[i].get('speaker') == 'supporter':
            return dialog[:i+1]
    
    return None


def load_and_extract_esconv(filepath: Path, num_samples: int = 20):
    """ESConv.json에서 랜덤 샘플링 후 윈도우 추출"""
    
    print(f"📂 {filepath.name} 로드 중...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   총 세션: {len(data)}개")
    
    # 랜덤 샘플링
    random.seed(42)  # 재현성
    sampled = random.sample(data, min(num_samples, len(data)))
    
    # 윈도우 추출
    windowed_sessions = []
    failed_count = 0
    
    for i, session in enumerate(sampled):
        dialog = session.get('dialog', [])
        
        # 윈도우 추출
        window = extract_window(dialog, min_turns=13, max_turns=20)
        
        if window:
            windowed_session = {
                'session_id': f"esconv_window_{i:04d}",
                'situation': session.get('situation', ''),
                'dialog': window,
                'original_turns': len(dialog),
                'window_turns': len(window)
            }
            windowed_sessions.append(windowed_session)
        else:
            failed_count += 1
    
    print(f"   ✅ 윈도우 추출 성공: {len(windowed_sessions)}개")
    print(f"   ❌ 윈도우 추출 실패: {failed_count}개 (대화 너무 짧거나 supporter 없음)")
    
    return windowed_sessions


def judge_session(session: dict, llm_client: OpenAIClient):
    """개별 세션 Judge 평가"""
    
    session_id = session.get('session_id', 'unknown')
    
    try:
        # 대화 구성
        situation = session.get('situation', '')
        dialog = session.get('dialog', [])
        
        dialog_lines = [f"[상황]\n{situation}\n"]
        dialog_lines.append("[대화]")
        
        for i, turn in enumerate(dialog):
            speaker = turn.get('speaker', 'unknown')
            content = turn.get('content', '')
            marker = " ← 평가 대상" if i == len(dialog) - 1 else ""
            dialog_lines.append(f"[{speaker.upper()}] {content}{marker}")
        
        full_dialog_text = "\n".join(dialog_lines)
        
        # Judge 프롬프트
        user_prompt = build_judge_prompt(full_dialog=full_dialog_text)
        
        # LLM 호출
        response = llm_client.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt
        )
        
        result = {
            'session_id': session_id,
            'predicted_label': response.get('label', 'Unknown'),
            'reason': response.get('reason', ''),
            'confidence': response.get('confidence', 'unknown'),
            'situation': situation,
            'dialog': dialog,
            'window_turns': session.get('window_turns', len(dialog)),
            'original_turns': session.get('original_turns', len(dialog))
        }
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'session_id': session_id,
            'predicted_label': 'Error',
            'reason': str(e),
            'confidence': 'error'
        }


def main():
    """메인 함수"""
    
    print("=" * 80)
    print("ESConv 원본 데이터 Judge 평가 (윈도우 방식)")
    print("방식: 처음부터 13~20턴 추출, 마지막은 supporter")
    print("=" * 80)
    print()
    
    # ESConv 파일 경로
    esconv_path = Path(__file__).parent.parent.parent / "ESConv.json"
    output_path = Path(__file__).parent.parent.parent / "data" / "pilot" / "judge_esconv_window_20.json"
    
    if not esconv_path.exists():
        print(f"❌ ESConv.json을 찾을 수 없습니다: {esconv_path}")
        return
    
    # 윈도우 추출
    sessions = load_and_extract_esconv(esconv_path, num_samples=20)
    
    if not sessions:
        print("❌ 윈도우 추출된 세션이 없습니다.")
        return
    
    print(f"🎲 평가 대상: {len(sessions)}개")
    print()
    
    # LLM 클라이언트 초기화
    print("🔧 LLM 클라이언트 초기화...")
    llm_client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=300
    )
    print()
    
    # Judge 평가
    print("🔍 Judge 평가 시작...")
    print("=" * 80)
    
    results = []
    
    for i, session in enumerate(sessions, 1):
        session_id = session.get('session_id', 'unknown')
        window_turns = session.get('window_turns', 0)
        original_turns = session.get('original_turns', 0)
        
        print(f"\n[{i}/{len(sessions)}] {session_id} ({window_turns}/{original_turns}턴)...", end=" ", flush=True)
        
        result = judge_session(session, llm_client)
        results.append(result)
        
        # 간단한 출력
        predicted = result.get('predicted_label', 'Unknown')
        confidence = result.get('confidence', 'unknown')
        
        print(f"{predicted} ({confidence})")
        if predicted != 'Normal':
            reason = result.get('reason', 'N/A')[:60]
            print(f"      이유: {reason}...")
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("평가 결과 통계")
    print("=" * 80)
    
    label_counts = Counter([r['predicted_label'] for r in results])
    
    print(f"\n{'레이블':<15} {'개수':<8} {'비율':<10}")
    print("─" * 40)
    
    for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5', 'Error']:
        count = label_counts.get(label, 0)
        ratio = (count / len(results) * 100) if len(results) > 0 else 0
        if count > 0:
            print(f"{label:<15} {count:<8} {ratio:<10.1f}%")
    
    print("─" * 40)
    print(f"{'TOTAL':<15} {len(results):<8} {'100.0%':<10}")
    
    # 위반 비율
    violation_count = sum(label_counts[label] for label in ['V1', 'V2', 'V3', 'V4', 'V5'])
    violation_ratio = (violation_count / len(results) * 100) if len(results) > 0 else 0
    
    print(f"\n📊 Normal: {label_counts.get('Normal', 0)}개 ({100-violation_ratio:.1f}%)")
    print(f"📊 위반 있음: {violation_count}개 ({violation_ratio:.1f}%)")
    
    # 결과 저장
    output_data = {
        'total': len(results),
        'statistics': dict(label_counts),
        'violation_ratio': violation_ratio,
        'window_config': {
            'min_turns': 13,
            'max_turns': 20,
            'last_speaker': 'supporter'
        },
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ ESConv 윈도우 Judge 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
