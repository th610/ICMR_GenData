"""
Phase 6 Extended: LLM Judge for Quality Verification (전체 대화 방식)

생성된 파일럿 데이터의 품질을 LLM으로 검증:
1. 전체 대화를 Judge에게 제공
2. 마지막 Supporter 응답 평가
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
    """전체 대화를 Judge 입력으로 구성
    
    Returns:
        전체 대화 텍스트
    """
    
    situation = session.get('situation', '')
    dialog = session.get('dialog', [])
    
    # 전체 대화 구성
    dialog_lines = [f"[상황]\n{situation}\n"]
    dialog_lines.append("[전체 대화]")
    
    for i, turn in enumerate(dialog):
        speaker = turn.get('speaker', 'unknown')
        # 마지막 턴이고 text 필드가 있으면 그걸 사용 (V1-V3 생성 데이터)
        if i == len(dialog) - 1 and 'text' in turn:
            content = turn.get('text', '')
        else:
            content = turn.get('content', '')
        # 마지막 턴 표시
        marker = " ← 평가 대상" if i == len(dialog) - 1 else ""
        dialog_lines.append(f"[{speaker.upper()}] {content}{marker}")
    
    full_dialog_text = "\n".join(dialog_lines)
    
    return full_dialog_text


class QualityJudge:
    """LLM을 사용한 품질 평가 (요약+윈도우 방식)"""
    
    def __init__(self, llm_client: OpenAIClient):
        self.llm_client = llm_client
        self.results = {
            'normal': [],
            'v1': [],
            'v2': [],
            'v3': [],
            'v4': [],
            'v5': []
        }
    
    def judge_session(self, session: Dict, expected_label: str) -> Dict:
        """개별 세션 평가 (전체 대화)"""
        
        session_id = session.get('session_id', 'unknown')
        
        try:
            # 전체 대화 생성
            full_dialog_text = build_judge_input(session)
            
            # prompts.py의 build_judge_prompt 사용
            user_prompt = build_judge_prompt(full_dialog=full_dialog_text)
            
            # LLM 호출
            response = self.llm_client.call(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=user_prompt
            )
            
            # 원본 데이터 추출
            situation = session.get('situation', '')
            dialog = session.get('dialog', [])
            
            # 결과 파싱 (원본 데이터 포함)
            result = {
                'session_id': session_id,
                'expected_label': expected_label,
                'predicted_label': response.get('label', 'Unknown'),  # 'label' 키 사용
                'reason': response.get('reason', ''),
                'confidence': response.get('confidence', 'unknown'),
                'matches': response.get('label') == expected_label,
                # 원본 데이터 추가
                'situation': situation,
                'dialog': dialog,
                'num_turns': len(dialog)
            }
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error judging {session_id}: {e}")
            return {
                'session_id': session_id,
                'expected_label': expected_label,
                'predicted_label': 'Error',
                'reason': str(e),
                'confidence': 'error',
                'matches': False
            }
    
    def judge_all(self, data_dir: Path, sample_per_class: int = None):
        """모든 세션 평가 (또는 샘플링)"""
        
        print("=" * 80)
        print("LLM Judge: Quality Verification")
        print("=" * 80)
        print()
        
        classes = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
        
        for cls in classes:
            filepath = data_dir / f"{cls}.json"
            
            if not filepath.exists():
                print(f"⚠️  Skipping {cls}: file not found")
                continue
            
            # 파일 로드
            with open(filepath, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            # 샘플링 (지정되면)
            if sample_per_class:
                sessions = sessions[:sample_per_class]
            
            print(f"\n{'─' * 80}")
            print(f"Judging {cls.upper()}: {len(sessions)} sessions")
            print(f"{'─' * 80}")
            
            # 각 세션 평가
            for i, session in enumerate(sessions, 1):
                session_id = session.get('session_id', f'{cls}_{i}')
                print(f"\n  [{i}/{len(sessions)}] {session_id}...", end=" ", flush=True)
                
                expected_label = cls.upper() if cls != 'normal' else 'Normal'
                result = self.judge_session(session, expected_label)
                
                # 결과 저장
                self.results[cls].append(result)
                
                # 간단한 출력
                matches = result.get('matches', False)
                predicted = result.get('predicted_label', 'Unknown')
                confidence = result.get('confidence', 'unknown')
                
                if matches:
                    print(f"✅ MATCH ({confidence})")
                else:
                    print(f"❌ MISMATCH: {expected_label} → {predicted} ({confidence})")
                    reason = result.get('reason', 'N/A')[:60]
                    print(f"      이유: {reason}...")
    
    def print_summary(self):
        """결과 요약 출력"""
        
        print("\n" + "=" * 80)
        print("LLM Judge Summary")
        print("=" * 80)
        
        total_match = 0
        total_mismatch = 0
        total_error = 0
        
        print(f"\n{'Class':<10} {'Total':<8} {'Match':<8} {'Mismatch':<10} {'Error':<8} {'Accuracy':<10}")
        print("─" * 70)
        
        for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
            results = self.results[cls]
            
            if not results:
                continue
            
            match_count = sum(1 for r in results if r.get('matches', False))
            error_count = sum(1 for r in results if r.get('predicted_label') == 'Error')
            mismatch_count = len(results) - match_count - error_count
            
            accuracy = (match_count / len(results) * 100) if len(results) > 0 else 0
            
            total_match += match_count
            total_mismatch += mismatch_count
            total_error += error_count
            
            print(f"{cls.upper():<10} {len(results):<8} {match_count:<8} "
                  f"{mismatch_count:<10} {error_count:<8} {accuracy:<10.1f}%")
        
        print("─" * 70)
        total = total_match + total_mismatch + total_error
        total_accuracy = (total_match / total * 100) if total > 0 else 0
        print(f"{'TOTAL':<10} {total:<8} {total_match:<8} "
              f"{total_mismatch:<10} {total_error:<8} {total_accuracy:<10.1f}%")
        
        # Accuracy 출력
        if total > 0:
            print(f"\n✅ Overall Accuracy: {total_accuracy:.1f}% ({total_match}/{total})")
            
            if total_mismatch > 0:
                print(f"❌ Mismatches: {total_mismatch}")
        
        # Mismatch 상세 출력
        if total_mismatch > 0:
            print("\n" + "=" * 80)
            print("Mismatch Details")
            print("=" * 80)
            
            for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
                mismatches = [
                    r for r in self.results[cls]
                    if not r.get('matches', False) and r.get('predicted_label') != 'Error'
                ]
                
                if mismatches:
                    print(f"\n{cls.upper()}:")
                    for r in mismatches:
                        session_id = r.get('session_id', 'unknown')
                        expected = r.get('expected_label', '')
                        predicted = r.get('predicted_label', '')
                        reason = r.get('reason', 'N/A')
                        print(f"  [{session_id}] {expected} → {predicted}")
                        print(f"    이유: {reason}")
    
    def save_results(self, output_path: Path):
        """결과를 JSON으로 저장"""
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """메인 함수"""
    
    print("=" * 80)
    print("Phase 6: LLM Judge Quality Verification")
    print("방식: 요약 + 윈도우 (실제 환경과 동일)")
    print("=" * 80)
    print()
    
    # 데이터 디렉토리
    data_dir = Path(__file__).parent.parent.parent / "data" / "pilot"
    output_path = Path(__file__).parent.parent.parent / "data" / "pilot" / "judge_results_summary_window.json"
    
    # LLM 클라이언트 초기화
    print("🔧 LLM 클라이언트 초기화...")
    llm_client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.3,  # 평가는 일관성 있게
        max_tokens=300    # 간단한 JSON 응답
    )
    print(f"   모델: gpt-4o-mini, temperature: 0.3")
    print()
    
    # Judge 실행
    judge = QualityJudge(llm_client)
    
    # 옵션: 각 클래스별 몇 개만 샘플링 (비용 절감)
    # None이면 전체 평가
    sample_per_class = None  # 전체 평가 (각 클래스별 5개, 총 30개)
    
    if sample_per_class:
        print(f"📊 샘플링: 각 클래스별 {sample_per_class}개 (총 {sample_per_class * 6}개)")
    else:
        print(f"📊 전체 평가: 30개 세션 (각 클래스별 5개)")
    print("💰 예상 비용: ~$0.40 (요약 생성 + Judge)")
    print()     
    
    judge.judge_all(data_dir, sample_per_class=sample_per_class)
    
    # 결과 요약
    judge.print_summary()
    
    # 결과 저장
    judge.save_results(output_path)
    
    print("\n" + "=" * 80)
    print("✅ LLM Judge completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
