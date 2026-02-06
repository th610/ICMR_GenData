"""
간단 Agent 테스트 - 골드 데이터 1개로 Turn 4 생성
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.step1_context_builder import ContextBuilder
from src.agent.step2_candidate_generator import CandidateGenerator  
from src.agent.step3_violation_detector import ViolationDetector
from src.llm.openai_client import OpenAIClient


def load_one_sample():
    """골드 데이터 1개 로드"""
    gold_path = Path(__file__).parent.parent.parent / "data" / "final" / "test_gold_300.json"
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['samples'][0]


def prepare_agent_input(sample):
    """Agent 입력 형식으로 변환"""
    # Turn 1-3 까지만 (Turn 4 제거)
    prefix = []
    for turn in sample['prefix_dialog']:
        prefix.append({
            'speaker': turn['speaker'],
            'content': turn['content']
        })
        # supporter turn 3개까지
        if turn['speaker'] == 'supporter' and len([t for t in prefix if t['speaker'] == 'supporter']) >= 3:
            break
    
    # 마지막 seeker 발화
    last_seeker = [t for t in prefix if t['speaker'] == 'seeker'][-1]
    
    # 이전 대화 기록
    history = []
    for turn in prefix:
        if turn == last_seeker:
            break
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    
    return {
        'situation': sample['situation'],
        'history': history,
        'user_input': last_seeker['content'],
        'gold_label': sample['label'],
        'gold_response': sample['generated_dialog'][0]['content']
    }


def main():
    print("="*80)
    print("Simple Agent Test - Generate Turn 4 from Gold Data")
    print("="*80)
    
    # 1. 샘플 로드
    sample = load_one_sample()
    test_input = prepare_agent_input(sample)
    
    print(f"\n[Situation]")
    print(test_input['situation'])
    
    print(f"\n[History] ({len(test_input['history'])} turns)")
    for turn in test_input['history']:
        speaker = "🔵" if turn['speaker'] == 'seeker' else "🟢"
        print(f"{speaker} {turn['speaker']}: {turn['text'][:80]}...")
    
    print(f"\n[User Input]")
    print(f"🔵 seeker: {test_input['user_input']}")
    
    print(f"\n[Gold Label] {test_input['gold_label']}")
    print(f"[Gold Response]")
    print(f"🟢 supporter: {test_input['gold_response'][:100]}...")
    
    print("\n" + "="*80)
    print("TODO: Initialize Agent and Generate Response")
    print("="*80)
    
    # TODO: Agent 초기화 및 실행
    # client = OpenAIClient()
    # context_builder = ContextBuilder(client)
    # generator = CandidateGenerator(client)
    # detector = ViolationDetector(mode="model", model_path="models/best_model")
    
    # context = context_builder.build_context(test_input['history'], test_input['situation'])
    # candidates = generator.generate_candidates(context)
    # results = detector.detect_batch(context, candidates)
    
    print("\nModules needed:")
    print("✅ 01_context_builder.py - IMPLEMENTED")
    print("✅ 02_candidate_generator.py - IMPLEMENTED")
    print("✅ 03_violation_detector.py - IMPLEMENTED")
    print("⏳ Integration test - PENDING")


if __name__ == "__main__":
    main()
