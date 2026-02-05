"""
ESConv 원본 데이터에서 랜덤 20개를 Judge로 평가
얼마나 많은 원본 세션이 위반으로 판단되는지 확인
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


def load_esconv_sessions(filepath: Path, num_samples: int = 20):
    """ESConv.json에서 랜덤 샘플링"""
    
    print(f"📂 {filepath.name} 로드 중...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   총 세션: {len(data)}개")
    
    # 랜덤 샘플링
    random.seed(42)  # 재현성
    sampled = random.sample(data, min(num_samples, len(data)))
    
    # session_id 추가 (없으면)
    for i, session in enumerate(sampled):
        if 'session_id' not in session:
            session['session_id'] = f"esconv_original_{i:04d}"
    
    return sampled


def judge_session(session: dict, llm_client: OpenAIClient):
    """개별 세션 Judge 평가 (전체 대화)"""
    
    session_id = session.get('session_id', 'unknown')
    
    try:
        # 전체 대화 구성
        situation = session.get('situation', '')
        dialog = session.get('dialog', [])
        
        dialog_lines = [f"[상황]\n{situation}\n"]
        dialog_lines.append("[전체 대화]")
        
        for i, turn in enumerate(dialog):
            speaker = turn.get('speaker', 'unknown')
            content = turn.get('content', '')
            marker = " ← 평가 대상" if i == len(dialog) - 1 else ""
            dialog_lines.append(f"[{speaker.upper()}] {content}{marker}")
        
        full_dialog_text = "\n".join(dialog_lines)
        
        # prompts.py의 build_judge_prompt 사용
        user_prompt = build_judge_prompt(full_dialog=full_dialog_text)
        
        # LLM 호출
        response = llm_client.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt
        )
        
        result = {
            'session_id': session_id,
            'predicted_label': response.get('label', 'Unknown'),  # 'label' 키 사용
            'reason': response.get('reason', ''),
            'confidence': response.get('confidence', 'unknown'),
            'situation': situation,
            'dialog': dialog,
            'num_turns': len(dialog)
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
    print("ESConv 원본 데이터 Judge 평가")
    print("목적: 원본 데이터가 얼마나 위반으로 판단되는지 확인")
    print("=" * 80)
    print()
    
    # ESConv 파일 경로
    esconv_path = Path(__file__).parent.parent.parent / "ESConv.json"
    output_path = Path(__file__).parent.parent.parent / "data" / "pilot" / "judge_esconv_original_20.json"
    
    if not esconv_path.exists():
        print(f"❌ ESConv.json을 찾을 수 없습니다: {esconv_path}")
        return
    
    # 샘플링
    sessions = load_esconv_sessions(esconv_path, num_samples=20)
    print(f"🎲 랜덤 샘플링: {len(sessions)}개")
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
        
        print(f"\n[{i}/{len(sessions)}] {session_id}...", end=" ", flush=True)
        
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
    
    # High confidence 위반만 상세 출력
    high_violations = [
        r for r in results 
        if r['predicted_label'] not in ['Normal', 'Error'] and r['confidence'] == 'high'
    ]
    
    if high_violations:
        print("\n" + "=" * 80)
        print("🔥 HIGH Confidence 위반 세션 상세 (대화 내용 포함)")
        print("=" * 80)
        
        for r in high_violations:
            print(f"\n{'=' * 80}")
            print(f"[{r['session_id']}] → {r['predicted_label']} (confidence: {r['confidence']})")
            print(f"{'=' * 80}")
            
            # 상황
            print(f"\n📌 상황:")
            print(f"   {r.get('situation', 'N/A')[:150]}...")
            
            # 이유
            print(f"\n❌ 위반 이유:")
            print(f"   {r['reason']}")
            
            # 대화 내용 (마지막 6턴)
            dialog = r.get('dialog', [])
            print(f"\n💬 대화 내용 (마지막 6턴, 총 {len(dialog)}턴):")
            
            last_6 = dialog[-6:] if len(dialog) >= 6 else dialog
            for i, turn in enumerate(last_6):
                speaker = turn.get('speaker', 'unknown')
                content = turn.get('content', '')[:200]  # 200자 제한
                marker = " ← 평가 대상" if i == len(last_6) - 1 else ""
                print(f"\n   [{speaker.upper()}]{marker}")
                print(f"   {content}...")
            
            print("\n" + "-" * 80)
    
    # 결과 저장
    output_data = {
        'total': len(results),
        'statistics': dict(label_counts),
        'violation_ratio': violation_ratio,
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ ESConv 원본 Judge 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
