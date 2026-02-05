"""
Dialogue Agent - 전체 파이프라인 통합

Architecture:
    1. Context Builder: WINDOW + TARGET 구성 (512 token reverse truncation)
    2. Generator: 후보 응답 1개 생성
    3. Violation Detector (BERT): label ∈ {Normal, V1..V5}
    4. Policy Layer (Controller): 위반 타입별 행동 결정
    5. Rewriter/Regenerator: 위반 수정
    6. Re-check: 수정된 응답 재검사
    7. Fail-safe: 반복 실패 시 안전 응답
    
Priority: V5 > V4 > V2 > V1 > V3 > Normal
Max retry: 1-2회
"""
from typing import List, Dict, Any, Optional
from .context_builder import ContextBuilder
from .candidate_generator import CandidateGenerator
from .violation_detector import ViolationDetector
from .controller import ControllerV2
from .logger import AgentLogger


class DialogueAgent:
    """Violation-aware 대화 에이전트"""
    
    def __init__(self,
                 context_builder: ContextBuilder,
                 candidate_generator: CandidateGenerator,
                 violation_detector: ViolationDetector,
                 controller: ControllerV2,
                 logger: Optional[AgentLogger] = None,
                 max_retries: int = 2):
        """
        Args:
            context_builder: 컨텍스트 압축 모듈 (WINDOW + TARGET)
            candidate_generator: 후보 생성 모듈 (LLM/규칙 기반)
            violation_detector: 위반 감지 모듈 (BERT Judge)
            controller: 제어 정책 모듈 (Policy Layer)
            logger: 로깅 모듈 (옵션)
            max_retries: 최대 재시도 횟수 (default: 2)
        """
        self.context_builder = context_builder
        self.candidate_generator = candidate_generator
        self.violation_detector = violation_detector
        self.controller = controller
        self.logger = logger or AgentLogger(enable_file_log=False)
        self.max_retries = max_retries
        
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
        self.logger.log_session_start(situation)
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        메인 파이프라인: 사용자 입력 → 안전한 응답 생성
        
        Pipeline:
            1. Context Builder: WINDOW 구성
            2. Generator: 초기 후보 생성
            3. Loop (max_retries):
                a. Detector: 위반 검사
                b. Controller: 정책 결정
                c. Rewriter: 필요시 수정
                d. Re-check: 수정 후 재검사
            4. Fail-safe: 최종 안전장치
        
        Args:
            user_input: Seeker 입력
            
        Returns:
            {
                'response': str,
                'label': str,
                'confidence': float,
                'retry_count': int,
                'violation_history': List[str]
            }
        """
        self.turn_count += 1
        self.logger.log_turn_start(self.turn_count, user_input)
        
        # Step 1: Context Building
        context = self.context_builder.build(
            history=self.dialog_history,
            current_input=user_input,
            situation=self.situation
        )
        self.logger.log_context(context)
        
        # Step 2: Initial Generation
        candidate = self.candidate_generator.generate(context)
        self.logger.log_candidate(candidate, attempt=0)
        
        # Step 3: Violation Loop
        violation_history = []
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            # Step 3a: Detect Violation
            detection = self.violation_detector.detect(
                window=context['window'],
                candidate=candidate
            )
            self.logger.log_detection(detection, attempt=attempt)
            
            # Step 3b: Policy Decision
            action = self.controller.decide(detection)
            self.logger.log_action(action, attempt=attempt)
            
            # Normal이면 통과
            if action['type'] == 'accept':
                self._update_history(user_input, candidate)
                return {
                    'response': candidate,
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'retry_count': retry_count,
                    'violation_history': violation_history
                }
            
            # 위반 발견
            violation_history.append(detection['label'])
            
            # 최대 재시도 도달
            if attempt >= self.max_retries:
                break
            
            # Step 3c: Rewrite
            retry_count += 1
            candidate = self.controller.rewrite(
                original=candidate,
                violation_type=detection['label'],
                context=context
            )
            self.logger.log_rewrite(candidate, attempt=attempt + 1)
        
        # Step 4: Fail-safe
        safe_response = self.controller.failsafe(
            violation_type=violation_history[-1] if violation_history else 'Unknown',
            context=context
        )
        self.logger.log_failsafe(safe_response)
        
        self._update_history(user_input, safe_response)
        
        return {
            'response': safe_response,
            'label': 'FAILSAFE',
            'confidence': 1.0,
            'retry_count': retry_count,
            'violation_history': violation_history
        }
    
    def _update_history(self, user_input: str, response: str) -> None:
        """대화 히스토리 업데이트"""
        self.dialog_history.append({'speaker': 'seeker', 'text': user_input})
        self.dialog_history.append({'speaker': 'supporter', 'text': response})
    
    def get_history(self) -> List[Dict[str, str]]:
        """현재 대화 히스토리 반환"""
        return self.dialog_history.copy()
    
    def end_session(self) -> Dict[str, Any]:
        """세션 종료 및 요약 반환"""
        summary = {
            'total_turns': self.turn_count,
            'situation': self.situation,
            'dialog_length': len(self.dialog_history)
        }
        self.logger.log_session_end(summary)
        return summary
