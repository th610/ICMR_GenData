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

from .context_builder import ContextBuilder
from .candidate_generator import CandidateGenerator
from .violation_detector import ViolationDetector
from .controller import ControllerV2
from .agent import DialogueAgent
from .logger import AgentLogger

__all__ = [
    'ContextBuilder',
    'CandidateGenerator',
    'ViolationDetector',
    'ControllerV2',
    'DialogueAgent',
    'AgentLogger',
]
