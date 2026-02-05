"""
test_gold_300.json에서 Turn 4를 제거한 테스트 데이터 생성
"""
import json
from pathlib import Path

# 경로 설정
root = Path(__file__).parent.parent.parent
input_path = root / "data" / "final" / "test_gold_300.json"
output_path = root / "src" / "agent" / "test_gold_300_prefix.json"

# 데이터 로드
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data['samples'])} samples")

# Turn 4 제거 (prefix_dialog는 Turn 3까지만, generated_dialog는 Turn 4)
for sample in data['samples']:
    # prefix_dialog에서 마지막 supporter turn (Turn 4) 제거
    # 구조: seeker, supporter, supporter, seeker, supporter, supporter, seeker, seeker, supporter, supporter, supporter, supporter (Turn 4)
    # Turn 3까지만 유지: 처음 7개 턴 (seeker 3개 + supporter 3개 + seeker 1개)
    sample['prefix_dialog'] = sample['prefix_dialog'][:7]
    
    # generated_dialog도 백업 (나중에 비교용)
    if 'generated_dialog' in sample and len(sample['generated_dialog']) > 0:
        sample['gold_turn4'] = sample['generated_dialog'][0]['content']
    if 'label' in sample:
        sample['gold_label'] = sample['label']

print(f"Processed: prefix_dialog limited to 7 turns (Turn 1-3)")

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")
print(f"\nSample structure:")
sample = data['samples'][0]
print(f"  - prefix_dialog: {len(sample['prefix_dialog'])} turns")
if 'gold_turn4' in sample:
    print(f"  - gold_turn4: {sample['gold_turn4'][:50]}...")
if 'gold_label' in sample:
    print(f"  - gold_label: {sample['gold_label']}")
