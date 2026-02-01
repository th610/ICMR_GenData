"""
Dialogue Agent - 전체 파이프라인 통합

Architecture:
    Context Builder → Candidate Generator → Violation Detector → Controller → Final Response
"""
from typing import List, Dict, Any, Optional
from .context_builder import ContextBuilder
from .candidate_generator import CandidateGenerator
from .violation_detector import ViolationDetector
from .controller import Controller
from .logger import AgentLogger


class DialogueAgent:
    """Violation-aware 대화 에이전트"""
    
    def __init__(self,
                 context_builder: ContextBuilder,
                 candidate_generator: CandidateGenerator,
                 violation_detector: ViolationDetector,
                 controller: Controller,
                 logger: Optional[AgentLogger] = None):
        """
        Args:
            context_builder: 컨텍스트 압축 모듈
            candidate_generator: 후보 생성 모듈
            violation_detector: 위반 감지 모듈
            controller: 제어 정책 모듈
            logger: 로깅 모듈 (옵션)
        """
        self.context_builder = context_builder
        self.candidate_generator = candidate_generator
        self.violation_detector = violation_detector
        self.controller = controller
        self.logger = logger or AgentLogger(enable_file_log=False)
        
        # Session state
        self.dialog_history = []
        self.turn_count = 0
        self.situation = None
    
    def start_session(self, situation: Optional[str] = None) -> None:
        """
        새 세션 시작
        
        Args:
            situation: 초기 상황 설명
        """
        self.dialog_history = []
        self.turn_count = 0
        self.situation = situation
        
        print(f"{'='*60}")
        print("Violation-aware Dialogue Agent Started")
        print(f"{'='*60}")
        if situation:
            print(f"\n[Initial Situation]\n{situation}\n")
    
    def process_turn(self, user_input: str) -> str:
        """
        1턴 처리 파이프라인
        
        Args:
            user_input: 사용자 입력 (seeker 발화)
        
        Returns:
            최종 응답 (supporter 발화)
        """
        # 사용자 입력 추가
        self.dialog_history.append({
            "speaker": "seeker",
            "text": user_input
        })
        
        # [1] Context Builder
        context = self.context_builder.build_context(
            self.dialog_history,
            self.situation
        )
        
        # [2] Candidate Generator
        candidates = self.candidate_generator.generate_candidates(context)
        
        # [3] Violation Detector
        detections = self.violation_detector.detect_batch(context, candidates)
        
        # [4] Controller
        result = self.controller.select_response(candidates, detections, context)
        
        final_response = result["final_response"]
        
        # 응답 히스토리에 추가
        self.dialog_history.append({
            "speaker": "supporter",
            "text": final_response
        })
        
        # [5] Logger
        self.logger.log_turn(
            turn_id=self.turn_count,
            context=context,
            candidates=candidates,
            detections=detections,
            controller_result=result,
            user_input=user_input
        )
        
        self.turn_count += 1
        
        return final_response
    
    def get_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        return self.dialog_history.copy()
    
    def end_session(self, session_id: Optional[str] = None) -> str:
        """
        세션 종료 및 로그 저장
        
        Args:
            session_id: 세션 ID
        
        Returns:
            저장된 로그 파일 경로
        """
        if session_id is None:
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_path = self.logger.save_session_summary(session_id)
        
        print(f"\n{'='*60}")
        print("Session Ended")
        print(f"Total turns: {self.turn_count}")
        if log_path:
            print(f"Log saved: {log_path}")
        print(f"{'='*60}\n")
        
        return log_path
    
    def interactive_chat(self) -> None:
        """대화형 인터페이스"""
        print("\nInteractive Chat Mode")
        print("Type 'quit' to exit, 'history' to see conversation\n")
        
        while True:
            try:
                user_input = input("Seeker: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.lower() == 'history':
                    self._print_history()
                    continue
                
                response = self.process_turn(user_input)
                print(f"Supporter: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
        
        self.end_session()
    
    def _print_history(self) -> None:
        """히스토리 출력"""
        print(f"\n{'='*60}")
        print("Conversation History")
        print(f"{'='*60}\n")
        
        for i, turn in enumerate(self.dialog_history, 1):
            speaker = turn["speaker"].capitalize()
            print(f"{i}. {speaker}: {turn['text']}")
        
        print(f"\n{'='*60}\n")
