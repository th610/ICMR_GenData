"""
Violation-aware Dialogue Agent

Architecture:
    Context Builder → Candidate Generator → Violation Detector → Controller → Final Responder
    
Components:
    - ContextBuilder: 대화 히스토리 압축 (summary + recent turns)
    - CandidateGenerator: K개 후보 응답 생성
    - ViolationDetector: 위반 패턴 감지
    - Controller: 선택/개입 정책
    - DialogueAgent: 전체 파이프라인 통합
"""

from .step1_context_builder import ContextBuilder
from .step2_candidate_generator import CandidateGenerator
from .step3_violation_detector import ViolationDetector
from .step4_controller import ControllerV2
from .step5_logger import AgentLogger
from .step6_agent import DialogueAgent

__all__ = [
    'ContextBuilder',
    'CandidateGenerator',
    'ViolationDetector',
    'ControllerV2',
    'DialogueAgent',
    'AgentLogger',
]
