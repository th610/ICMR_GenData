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

# generated_dialog를 prefix_dialog에 합친 후 마지막 턴 제거
for sample in data['samples']:
    # prefix_dialog + generated_dialog 합치기
    full_dialog = sample['prefix_dialog'].copy()
    if 'generated_dialog' in sample:
        for turn in sample['generated_dialog']:
            full_dialog.append({
                'speaker': turn['speaker'],
                'content': turn['content']
            })
    
    # 마지막 턴을 gold_turn4로 백업 (에이전트가 생성해야 할 target)
    if len(full_dialog) > 0:
        last_turn = full_dialog[-1]
        sample['gold_turn4'] = last_turn['content']
        sample['gold_speaker'] = last_turn['speaker']
        
        # 마지막 턴 제거 (에이전트 입력용)
        sample['prefix_dialog'] = full_dialog[:-1]
    
    # label 백업
    if 'label' in sample:
        sample['gold_label'] = sample['label']

print(f"Processed: Merged prefix + generated, removed last turn")

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")
print(f"\nSample structure:")
sample = data['samples'][0]
print(f"  - prefix_dialog: {len(sample['prefix_dialog'])} turns (merged, last turn removed)")
print(f"  - Last turn speaker: {sample['prefix_dialog'][-1]['speaker']}")
if 'gold_turn4' in sample:
    print(f"  - gold_turn4 ({sample['gold_speaker']}): {sample['gold_turn4'][:60]}...")
if 'gold_label' in sample:
    print(f"  - gold_label: {sample['gold_label']}")
