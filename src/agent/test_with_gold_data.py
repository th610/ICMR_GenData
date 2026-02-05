"""
Agent 테스트 - 골드 데이터로 Turn 4 생성 테스트

Process:
    1. test_gold_300.json에서 샘플 로드
    2. Turn 1-3만 사용 (Turn 4 제거)
    3. Agent로 Turn 4 생성
    4. 생성된 응답의 label과 골드 label 비교
"""
import json
from pathlib import Path


def load_gold_samples(n_samples: int = 10):
    """골드 데이터 로드"""
    gold_path = Path(__file__).parent.parent.parent / "data" / "final" / "test_gold_300.json"
    
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['samples'][:n_samples]


def extract_prefix_dialog(sample):
    """Turn 4를 제외한 prefix 추출"""
    # prefix_dialog는 Turn 3까지만 포함
    dialog = sample['prefix_dialog']
    
    # Last turn이 supporter가 아니면 조정 필요
    # ESConv 구조: seeker -> supporter -> seeker -> supporter (Turn 4)
    # 우리는 Turn 3 (seeker)까지만 사용
    
    prefix = []
    for turn in dialog:
        prefix.append({
            'speaker': turn['speaker'],
            'content': turn['content']
        })
        # supporter turn 3개까지만 (Turn 1,2,3)
        if turn['speaker'] == 'supporter' and len([t for t in prefix if t['speaker'] == 'supporter']) >= 3:
            break
    
    return prefix


def prepare_test_input(sample):
    """Agent 입력 형식으로 변환"""
    situation = sample['situation']
    prefix = extract_prefix_dialog(sample)
    
    # 마지막 seeker 발화가 입력
    last_seeker = [t for t in prefix if t['speaker'] == 'seeker'][-1]
    user_input = last_seeker['content']
    
    # 이전 대화 기록 (마지막 seeker 제외)
    history = []
    for turn in prefix:
        if turn == last_seeker:
            break
        history.append({
            'speaker': turn['speaker'],
            'content': turn['content']
        })
    
    return {
        'situation': situation,
        'history': history,
        'user_input': user_input,
        'gold_label': sample['label'],
        'gold_response': sample['generated_dialog'][0]['content'],  # Turn 4
        'esconv_id': sample['esconv_session_id']
    }


def main():
    """메인 테스트"""
    print("="*80)
    print("Agent Test with Gold Data - Turn 4 Generation")
    print("="*80)
    
    # 1. 샘플 로드
    samples = load_gold_samples(n_samples=5)
    print(f"\n1. Loaded {len(samples)} gold samples")
    
    # 2. 테스트 입력 준비
    test_cases = []
    for sample in samples:
        test_input = prepare_test_input(sample)
        test_cases.append(test_input)
    
    print(f"2. Prepared {len(test_cases)} test cases")
    
    # 3. 샘플 출력 (구조 확인)
    print("\n" + "="*80)
    print("Sample Test Case:")
    print("="*80)
    sample_case = test_cases[0]
    
    print(f"\n[Situation]")
    print(sample_case['situation'])
    
    print(f"\n[Dialog History] ({len(sample_case['history'])} turns)")
    for i, turn in enumerate(sample_case['history'], 1):
        speaker = "🔵 SEEKER" if turn['speaker'] == 'seeker' else "🟢 SUPPORTER"
        print(f"{speaker} (Turn {i}): {turn['content'][:100]}...")
    
    print(f"\n[User Input] (Turn {len(sample_case['history'])+1})")
    print(f"🔵 SEEKER: {sample_case['user_input']}")
    
    print(f"\n[Gold Label] {sample_case['gold_label']}")
    print(f"[Gold Response]")
    print(f"🟢 SUPPORTER: {sample_case['gold_response']}")
    
    print("\n" + "="*80)
    print("TODO: Implement Agent modules to generate Turn 4")
    print("="*80)
    
    # 4. Agent 테스트 (TODO)
    # from .agent import DialogueAgent
    # agent = DialogueAgent(...)
    # result = agent.generate_response(...)
    
    # 저장
    output_path = Path(__file__).parent / "test_cases_gold.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest cases saved to: {output_path}")


if __name__ == "__main__":
    main()
