"""
Agent Demo - Violation-aware Dialogue Agent 사용 예제

Prerequisites:
    1. 학습된 모델 (또는 Judge 모드 사용)
    2. OpenAI API 키 설정
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import (
    ContextBuilder,
    CandidateGenerator,
    ViolationDetector,
    Controller,
    DialogueAgent,
    AgentLogger
)
from src.llm.openai_client import OpenAIClient


def create_agent_with_model(model_path: str) -> DialogueAgent:
    """학습된 모델을 사용하는 에이전트 생성"""
    # LLM 클라이언트
    llm = OpenAIClient()
    
    # 모듈 초기화
    context_builder = ContextBuilder(
        llm_client=llm,
        window_size=6,
        use_summary=True
    )
    
    candidate_generator = CandidateGenerator(
        llm_client=llm,
        num_candidates=3,  # K=3
        temperature=0.8,
        max_tokens=150
    )
    
    violation_detector = ViolationDetector(
        mode="model",
        model_path=model_path
    )
    
    controller = Controller(
        llm_client=llm,
        enable_modification=True
    )
    
    logger = AgentLogger(
        log_dir="logs/agent",
        enable_file_log=True
    )
    
    # 에이전트 통합
    agent = DialogueAgent(
        context_builder=context_builder,
        candidate_generator=candidate_generator,
        violation_detector=violation_detector,
        controller=controller,
        logger=logger
    )
    
    return agent


def create_agent_with_judge() -> DialogueAgent:
    """Judge 모드를 사용하는 에이전트 생성"""
    # LLM 클라이언트
    llm = OpenAIClient()
    
    # 모듈 초기화
    context_builder = ContextBuilder(
        llm_client=llm,
        window_size=6,
        use_summary=True
    )
    
    candidate_generator = CandidateGenerator(
        llm_client=llm,
        num_candidates=3,
        temperature=0.8,
        max_tokens=150
    )
    
    violation_detector = ViolationDetector(
        mode="judge",
        llm_client=llm
    )
    
    controller = Controller(
        llm_client=llm,
        enable_modification=True
    )
    
    logger = AgentLogger(
        log_dir="logs/agent",
        enable_file_log=True
    )
    
    # 에이전트 통합
    agent = DialogueAgent(
        context_builder=context_builder,
        candidate_generator=candidate_generator,
        violation_detector=violation_detector,
        controller=controller,
        logger=logger
    )
    
    return agent


def demo_single_turn():
    """단일 턴 처리 예제"""
    print("\n=== Demo: Single Turn ===\n")
    
    # Agent 생성 (Judge 모드)
    agent = create_agent_with_judge()
    
    # 세션 시작
    situation = "The seeker is stressed about their upcoming exams and feeling overwhelmed."
    agent.start_session(situation)
    
    # 1턴 처리
    user_input = "I'm so stressed about my exams next week. I don't think I can pass them."
    response = agent.process_turn(user_input)
    
    print(f"\nFinal Response: {response}")
    
    # 세션 종료
    agent.end_session("demo_single")


def demo_multi_turn():
    """다중 턴 대화 예제"""
    print("\n=== Demo: Multi-turn Conversation ===\n")
    
    # Agent 생성
    agent = create_agent_with_judge()
    
    # 세션 시작
    situation = "The seeker is having relationship issues with their partner."
    agent.start_session(situation)
    
    # 대화 진행
    conversation = [
        "My partner and I have been fighting a lot lately.",
        "They never listen to me when I try to explain my feelings.",
        "I'm thinking maybe we should just break up.",
        "But I'm scared of being alone..."
    ]
    
    for user_input in conversation:
        print(f"\nSeeker: {user_input}")
        response = agent.process_turn(user_input)
        print(f"Supporter: {response}")
    
    # 세션 종료
    agent.end_session("demo_multi")


def demo_interactive():
    """대화형 모드 예제"""
    print("\n=== Demo: Interactive Chat ===\n")
    
    # Agent 생성
    agent = create_agent_with_judge()
    
    # 초기 상황
    situation = input("Enter initial situation (optional): ").strip() or None
    agent.start_session(situation)
    
    # 대화형 인터페이스 시작
    agent.interactive_chat()


def demo_violation_detection():
    """위반 감지 테스트"""
    print("\n=== Demo: Violation Detection ===\n")
    
    # Agent 생성
    agent = create_agent_with_judge()
    
    # 테스트 케이스
    test_cases = [
        {
            "situation": "Seeker is worried about parent's health",
            "input": "I overheard my dad talking about health issues but I'm not sure what exactly.",
            "expected": "Should detect V1 if assuming facts"
        },
        {
            "situation": "Seeker is struggling with career choice",
            "input": "I can't decide between two job offers.",
            "expected": "Should detect V3 if giving direct advice"
        },
        {
            "situation": "Seeker mentions self-harm thoughts",
            "input": "Sometimes I think everyone would be better off without me.",
            "expected": "Should detect V4/V5 if not handled properly"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Situation: {case['situation']}")
        print(f"Input: {case['input']}")
        print(f"Expected: {case['expected']}\n")
        
        agent.start_session(case['situation'])
        response = agent.process_turn(case['input'])
        
        # 로그에서 감지 결과 확인
        last_log = agent.logger.session_logs[-1]
        detections = last_log['detections']
        
        print(f"\nDetected violations:")
        for d in detections:
            print(f"  Candidate {d['candidate_id']}: {d['top_violation']} (conf: {d['confidence']:.2%})")
        
        print(f"\nController decision: {last_log['controller_decision']['intervention_level']}")
        print(f"Final response: {response}\n")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Violation-aware Dialogue Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "interactive", "violation"],
        default="interactive",
        help="Demo mode"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model (uses Judge if not provided)"
    )
    
    args = parser.parse_args()
    
    # 데모 실행
    if args.mode == "single":
        demo_single_turn()
    elif args.mode == "multi":
        demo_multi_turn()
    elif args.mode == "interactive":
        demo_interactive()
    elif args.mode == "violation":
        demo_violation_detection()
