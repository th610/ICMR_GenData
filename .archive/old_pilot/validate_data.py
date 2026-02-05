"""
Phase 6: Data Validation and Quality Check

이 스크립트는 생성된 파일럿 데이터 30개 세션을 검증합니다:
1. 구조 검증 (필수 필드, 데이터 타입)
2. 내용 검증 (턴 수 범위, 마지막 턴 speaker)
3. 통계 출력 (클래스별 개수, 턴 수 분포)
4. 샘플 출력 (각 클래스별 1개씩)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DataValidator:
    """파일럿 데이터 검증 클래스"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(dict)
        
    def validate_all(self) -> bool:
        """모든 검증 수행"""
        print("=" * 80)
        print("Phase 6: Pilot Data Validation")
        print("=" * 80)
        print()
        
        # 클래스별 파일 검증
        classes = ["normal", "v1", "v2", "v3", "v4", "v5"]
        all_sessions = []
        
        for cls in classes:
            filepath = self.data_dir / f"{cls}.json"
            sessions = self.validate_class_file(cls, filepath)
            if sessions:
                all_sessions.extend(sessions)
        
        # 전체 통계
        self.print_overall_stats(all_sessions)
        
        # 샘플 출력
        self.print_samples(all_sessions)
        
        # 에러 및 경고 출력
        self.print_errors_and_warnings()
        
        return len(self.errors) == 0
    
    def validate_class_file(self, class_name: str, filepath: Path) -> List[Dict]:
        """클래스별 JSON 파일 검증"""
        print(f"\n{'─' * 80}")
        print(f"Validating: {class_name.upper()}")
        print(f"{'─' * 80}")
        
        # 1. 파일 존재 여부
        if not filepath.exists():
            self.errors.append(f"{class_name}: File not found - {filepath}")
            print(f"❌ File not found: {filepath}")
            return []
        
        # 2. JSON 로드
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"{class_name}: JSON parsing error - {e}")
            print(f"❌ JSON parsing error: {e}")
            return []
        
        print(f"✅ Loaded {len(sessions)} sessions")
        
        # 3. 각 세션 검증
        turn_counts = []
        for i, session in enumerate(sessions):
            errors = self.validate_session(class_name, i, session)
            if errors:
                self.errors.extend(errors)
            else:
                turn_counts.append(len(session.get('dialog', [])))
        
        # 4. 클래스별 통계
        if turn_counts:
            self.stats[class_name] = {
                'count': len(sessions),
                'turn_counts': turn_counts,
                'avg_turns': sum(turn_counts) / len(turn_counts),
                'min_turns': min(turn_counts),
                'max_turns': max(turn_counts)
            }
            
            print(f"   Sessions: {len(sessions)}")
            print(f"   Turns - Avg: {self.stats[class_name]['avg_turns']:.1f}, "
                  f"Min: {self.stats[class_name]['min_turns']}, "
                  f"Max: {self.stats[class_name]['max_turns']}")
        
        return sessions
    
    def validate_session(self, class_name: str, index: int, session: Dict) -> List[str]:
        """개별 세션 검증"""
        errors = []
        session_id = session.get('session_id', f'{class_name}_{index}')
        
        # 1. 필수 필드 존재 여부
        required_fields = ['situation', 'dialog', 'primary_label', 'session_id', 
                          'generation_method', 'violation_turn_index', 'violation_reason']
        
        for field in required_fields:
            if field not in session:
                errors.append(f"{session_id}: Missing required field '{field}'")
        
        # 2. dialog 검증
        if 'dialog' in session:
            dialog = session['dialog']
            
            # dialog가 리스트인지
            if not isinstance(dialog, list):
                errors.append(f"{session_id}: 'dialog' must be a list")
            elif len(dialog) == 0:
                errors.append(f"{session_id}: 'dialog' is empty")
            else:
                # 턴 수 범위 검증
                turn_count = len(dialog)
                
                # V1-V3는 12-22턴, V4-V5는 12-16턴 (여유 있게)
                if class_name in ['normal', 'v1', 'v2', 'v3']:
                    if not (10 <= turn_count <= 25):
                        self.warnings.append(
                            f"{session_id}: Turn count {turn_count} outside expected range [10-25]"
                        )
                elif class_name in ['v4', 'v5']:
                    if not (10 <= turn_count <= 18):
                        self.warnings.append(
                            f"{session_id}: Turn count {turn_count} outside expected range [10-18]"
                        )
                
                # 마지막 턴이 supporter인지
                last_turn = dialog[-1]
                if 'speaker' in last_turn:
                    if last_turn['speaker'] != 'supporter':
                        errors.append(
                            f"{session_id}: Last turn speaker is '{last_turn['speaker']}', "
                            f"expected 'supporter'"
                        )
                else:
                    errors.append(f"{session_id}: Last turn missing 'speaker' field")
                
                # 각 턴에 speaker와 content가 있는지
                for turn_idx, turn in enumerate(dialog):
                    if 'speaker' not in turn:
                        errors.append(f"{session_id}: Turn {turn_idx} missing 'speaker'")
                    if 'content' not in turn:
                        errors.append(f"{session_id}: Turn {turn_idx} missing 'content'")
        
        # 3. primary_label 검증
        if 'primary_label' in session:
            expected_label = class_name.upper() if class_name != 'normal' else 'Normal'
            actual_label = session['primary_label']
            if actual_label != expected_label:
                errors.append(
                    f"{session_id}: primary_label is '{actual_label}', "
                    f"expected '{expected_label}'"
                )
        
        # 4. violation_turn_index 검증
        if 'violation_turn_index' in session:
            vti = session['violation_turn_index']
            if class_name == 'normal':
                if vti is not None:
                    errors.append(f"{session_id}: Normal should have violation_turn_index=null")
            else:
                if vti is None:
                    errors.append(f"{session_id}: Violation class should have violation_turn_index")
                elif 'dialog' in session and isinstance(vti, int):
                    if not (0 <= vti < len(session['dialog'])):
                        errors.append(
                            f"{session_id}: violation_turn_index {vti} out of range [0-{len(session['dialog'])-1}]"
                        )
        
        # 5. violation_reason 검증
        if 'violation_reason' in session:
            vr = session['violation_reason']
            if class_name == 'normal':
                if vr is not None:
                    errors.append(f"{session_id}: Normal should have violation_reason=null")
            else:
                if vr is None or vr == "":
                    errors.append(f"{session_id}: Violation class should have violation_reason")
        
        return errors
    
    def print_overall_stats(self, all_sessions: List[Dict]):
        """전체 통계 출력"""
        print("\n" + "=" * 80)
        print("Overall Statistics")
        print("=" * 80)
        
        total = len(all_sessions)
        print(f"\n📊 Total Sessions: {total}")
        
        # 클래스별 통계 테이블
        print(f"\n{'Class':<10} {'Count':<8} {'Avg Turns':<12} {'Min':<6} {'Max':<6}")
        print("─" * 50)
        
        for class_name in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
            if class_name in self.stats:
                s = self.stats[class_name]
                print(f"{class_name.upper():<10} {s['count']:<8} "
                      f"{s['avg_turns']:<12.1f} {s['min_turns']:<6} {s['max_turns']:<6}")
        
        # 전체 턴 수 분포
        all_turns = []
        for s in self.stats.values():
            all_turns.extend(s['turn_counts'])
        
        if all_turns:
            print(f"\n📈 Overall Turn Distribution:")
            print(f"   Average: {sum(all_turns) / len(all_turns):.1f}")
            print(f"   Min: {min(all_turns)}")
            print(f"   Max: {max(all_turns)}")
    
    def print_samples(self, all_sessions: List[Dict]):
        """클래스별 샘플 1개씩 출력"""
        print("\n" + "=" * 80)
        print("Sample Sessions (First session from each class)")
        print("=" * 80)
        
        classes = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
        
        for cls in classes:
            # 해당 클래스의 첫 번째 세션 찾기
            sample = None
            for session in all_sessions:
                if session.get('primary_label', '').lower() == cls or \
                   session.get('primary_label', '') == cls.upper():
                    sample = session
                    break
            
            if sample:
                print(f"\n{'─' * 80}")
                print(f"{cls.upper()} Sample")
                print(f"{'─' * 80}")
                print(f"Session ID: {sample.get('session_id', 'N/A')}")
                print(f"Situation: {sample.get('situation', 'N/A')[:100]}...")
                print(f"Turns: {len(sample.get('dialog', []))}")
                print(f"Violation Turn Index: {sample.get('violation_turn_index', 'N/A')}")
                print(f"Violation Reason: {sample.get('violation_reason', 'N/A')[:80] if sample.get('violation_reason') else 'null'}...")
                
                # 마지막 2턴 출력
                if 'dialog' in sample and len(sample['dialog']) >= 2:
                    print(f"\nLast 2 turns:")
                    for turn in sample['dialog'][-2:]:
                        speaker = turn.get('speaker', 'unknown')
                        content = turn.get('content', '')[:80]
                        print(f"  [{speaker}] {content}...")
    
    def print_errors_and_warnings(self):
        """에러 및 경고 출력"""
        print("\n" + "=" * 80)
        print("Validation Results")
        print("=" * 80)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # 최대 10개만
                print(f"   - {error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more errors")
        else:
            print("\n✅ No errors found!")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # 최대 10개만
                print(f"   - {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more warnings")
        else:
            print("\n✅ No warnings!")
        
        print("\n" + "=" * 80)
        if len(self.errors) == 0:
            print("✅ Validation PASSED - All 30 pilot sessions are valid!")
        else:
            print("❌ Validation FAILED - Please fix the errors above")
        print("=" * 80)


def main():
    """메인 함수"""
    # 데이터 디렉토리
    data_dir = Path(__file__).parent.parent.parent / "data" / "pilot"
    
    # 검증 실행
    validator = DataValidator(data_dir)
    success = validator.validate_all()
    
    # 종료 코드
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
