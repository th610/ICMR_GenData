"""
Agent Logger

Role: 전체 파이프라인 추적 및 디버깅 정보 기록

Logs:
    - 입력 컨텍스트
    - 생성된 후보들
    - 위반 감지 결과
    - Controller 결정
    - 최종 출력
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class AgentLogger:
    """에이전트 실행 로그 기록"""
    
    def __init__(self, log_dir: str = "logs/agent", enable_file_log: bool = True):
        """
        Args:
            log_dir: 로그 파일 저장 디렉토리
            enable_file_log: 파일 로그 활성화
        """
        self.log_dir = Path(log_dir)
        self.enable_file_log = enable_file_log
        
        if enable_file_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_logs = []
    
    def log_turn(self, 
                 turn_id: int,
                 context: Dict[str, Any],
                 candidates: List[Dict[str, Any]],
                 detections: List[Dict[str, Any]],
                 controller_result: Dict[str, Any],
                 user_input: Optional[str] = None) -> None:
        """
        단일 턴 로그 기록
        
        Args:
            turn_id: 턴 번호
            context: ContextBuilder 출력
            candidates: CandidateGenerator 출력
            detections: ViolationDetector 출력
            controller_result: Controller 출력
            user_input: 사용자 입력 (옵션)
        """
        log_entry = {
            "turn_id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "context": {
                "total_turns": context.get("total_turns"),
                "has_summary": context.get("state_summary") is not None,
                "recent_turns_count": len(context.get("recent_turns", []))
            },
            "candidates": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "length": len(c["text"])
                }
                for c in candidates
            ],
            "detections": [
                {
                    "candidate_id": d["candidate_id"],
                    "top_violation": d["top_violation"],
                    "confidence": d["confidence"],
                    "severity": d["severity"],
                    "violations": d["violations"]
                }
                for d in detections
            ],
            "controller_decision": {
                "selected_candidate_id": controller_result["selected_candidate_id"],
                "intervention_level": controller_result["intervention_level"],
                "intervention_details": controller_result["intervention_details"]
            },
            "final_response": controller_result["final_response"]
        }
        
        self.session_logs.append(log_entry)
        
        # Console output
        self._print_turn_summary(log_entry)
        
        # File output
        if self.enable_file_log:
            self._save_turn_log(log_entry)
    
    def _print_turn_summary(self, log_entry: Dict[str, Any]) -> None:
        """콘솔 출력"""
        print(f"\n{'='*60}")
        print(f"Turn {log_entry['turn_id']}")
        print(f"{'='*60}")
        
        if log_entry["user_input"]:
            print(f"\nUser: {log_entry['user_input']}")
        
        print(f"\n[Candidates Generated: {len(log_entry['candidates'])}]")
        for i, (cand, detect) in enumerate(zip(log_entry['candidates'], log_entry['detections'])):
            violation_status = f"{detect['top_violation']} (conf: {detect['confidence']:.2%})"
            print(f"  {i}. {violation_status}")
            print(f"     {cand['text'][:80]}...")
        
        decision = log_entry["controller_decision"]
        print(f"\n[Controller Decision]")
        print(f"  Selected: Candidate #{decision['selected_candidate_id']}")
        print(f"  Intervention: {decision['intervention_level']}")
        print(f"  Reason: {decision['intervention_details'].get('reason', 'N/A')}")
        
        print(f"\n[Final Response]")
        print(f"  {log_entry['final_response']}")
        print(f"{'='*60}\n")
    
    def _save_turn_log(self, log_entry: Dict[str, Any]) -> None:
        """파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"turn_{log_entry['turn_id']}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def save_session_summary(self, session_id: str) -> str:
        """전체 세션 요약 저장"""
        if not self.session_logs:
            return ""
        
        summary = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "total_turns": len(self.session_logs),
            "turns": self.session_logs,
            "statistics": self._compute_statistics()
        }
        
        if self.enable_file_log:
            filename = self.log_dir / f"session_{session_id}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return str(filename)
        
        return ""
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """세션 통계"""
        if not self.session_logs:
            return {}
        
        intervention_counts = {"none": 0, "soft": 0, "hard": 0}
        violation_counts = {"normal": 0, "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0}
        
        for log in self.session_logs:
            # Intervention level
            level = log["controller_decision"]["intervention_level"]
            intervention_counts[level] = intervention_counts.get(level, 0) + 1
            
            # Violations detected
            for detect in log["detections"]:
                violation = detect["top_violation"]
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return {
            "intervention_distribution": intervention_counts,
            "violation_distribution": violation_counts,
            "avg_candidates_per_turn": sum(len(log["candidates"]) for log in self.session_logs) / len(self.session_logs)
        }
    
    def get_debug_panel(self, turn_id: int) -> str:
        """특정 턴의 디버그 정보 텍스트로 반환"""
        if turn_id >= len(self.session_logs):
            return "Turn not found"
        
        log = self.session_logs[turn_id]
        
        lines = []
        lines.append(f"=== Debug Panel: Turn {turn_id} ===\n")
        
        lines.append("[Context]")
        lines.append(f"  Total turns: {log['context']['total_turns']}")
        lines.append(f"  Has summary: {log['context']['has_summary']}")
        lines.append(f"  Recent window: {log['context']['recent_turns_count']} turns\n")
        
        lines.append("[Candidates & Detections]")
        for cand, detect in zip(log['candidates'], log['detections']):
            lines.append(f"  Candidate {cand['id']}:")
            lines.append(f"    Text: {cand['text']}")
            lines.append(f"    Violation: {detect['top_violation']} (severity: {detect['severity']})")
            lines.append(f"    Confidence: {detect['confidence']:.2%}")
            lines.append(f"    All violations: {detect['violations']}\n")
        
        lines.append("[Controller Decision]")
        decision = log['controller_decision']
        lines.append(f"  Selected: Candidate #{decision['selected_candidate_id']}")
        lines.append(f"  Intervention: {decision['intervention_level']}")
        lines.append(f"  Details: {decision['intervention_details']}\n")
        
        lines.append("[Final Output]")
        lines.append(f"  {log['final_response']}")
        
        return "\n".join(lines)
