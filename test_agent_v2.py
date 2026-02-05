"""
에이전트 V2 사용 예시
"""
from src.agent.agent import DialogueAgent
from src.agent.context_builder import ContextBuilder
from src.agent.candidate_generator import CandidateGenerator
from src.agent.violation_detector import ViolationDetector
from src.agent.controller import ControllerV2
from src.agent.logger import AgentLogger


def main():
    """
    에이전트 구조 테스트
    
    TODO:
        - Context Builder 구현 (512 token reverse truncation)
        - Candidate Generator 구현 (LLM 또는 규칙 기반)
        - Violation Detector 구현 (BERT Judge 모델 로드)
        - Controller V2 완성 (LLM 연동)
    """
    
    print("="*60)
    print("Agent V2 Architecture")
    print("="*60)
    print("""
Pipeline:
    1. Context Builder: WINDOW + TARGET (512 token)
    2. Generator: 후보 응답 생성
    3. Loop (max 2회):
        a. Detector: BERT Judge 검사
        b. Controller: 정책 결정
        c. Rewriter: 위반 수정
        d. Re-check: 재검사
    4. Fail-safe: 안전 응답

Priority: V5 > V4 > V2 > V1 > V3 > Normal
    """)
    
    # TODO: 실제 모듈 구현 후 테스트
    print("\n⚠️  모듈 구현 필요:")
    print("  - ContextBuilder.build()")
    print("  - CandidateGenerator.generate()")
    print("  - ViolationDetector.detect()")
    print("  - ControllerV2.rewrite() with LLM")
    
    print("\n✅ 구조 정의 완료:")
    print("  - agent_v2.py: 메인 파이프라인")
    print("  - controller_v2.py: Policy Layer")
    print("  - Failsafe templates 정의")
    print("  - Rewrite prompts 정의")


if __name__ == '__main__':
    main()
