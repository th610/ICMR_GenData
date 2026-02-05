import json
import random

# V3 데이터 로드
v3_data = json.load(open('.archive/2026-02-02_generated/generated/v3_200.json', encoding='utf-8'))

# test에서 사용된 샘플 찾기 (random seed 42)
random.seed(42)
test_samples = random.sample(v3_data, 10)

print('='*70)
print('V10 테스트에 사용된 V3 샘플 (10개)')
print('='*70)
print()

for idx, sample in enumerate(test_samples, 1):
    print(f'[V3 샘플 {idx}]')
    print(f'session_id: {sample["session_id"]}')
    print(f'situation: {sample["situation"][:60]}...')
    
    prefix_len = sample['prefix_length']
    insert_len = sample['insertion_length']
    violation_turn = sample['violation_turn_index']
    
    print(f'prefix_length: {prefix_len}, insertion_length: {insert_len}')
    print(f'dialog 총 {len(sample["dialog"])}턴, 위반: turn {violation_turn}')
    print()
    
    # INSERTION 영역만 출력 (seeker 요청 + supporter 위반)
    insert_start = prefix_len
    insert_end = insert_start + insert_len
    
    print('  INSERTION 영역:')
    for i in range(insert_start, insert_end):
        turn = sample['dialog'][i]
        marker = " ← 위반" if i == violation_turn else ""
        print(f'    Turn {i}: [{turn["speaker"]}]{marker}')
        content = turn['content']
        # 길면 축약
        if len(content) > 150:
            content = content[:150] + "..."
        print(f'      {content}')
        print()
    
    # Violation reason
    print(f'  violation_reason:')
    print(f'    {sample.get("violation_reason", "N/A")[:200]}...')
    print()
    print('-'*70)
    print()
