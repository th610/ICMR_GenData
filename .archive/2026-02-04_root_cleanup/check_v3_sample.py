import json

v3_data = json.load(open('.archive/2026-02-02_generated/generated/v3_200.json', encoding='utf-8'))
sample = v3_data[0]

print('='*60)
print('V3 샘플 분석')
print('='*60)
print(f'Situation: {sample["situation"]}')
print()

prefix_len = sample['prefix_length']
insert_start = prefix_len
insert_end = insert_start + sample['insertion_length']

print(f'삽입 구간: turn {insert_start} ~ {insert_end-1}')
print(f'위반 턴: {sample["violation_turn_index"]}')
print()

# 삽입 전 2턴 + 삽입 4턴 + 이후 1턴
start_idx = max(0, insert_start - 2)
end_idx = min(len(sample['dialog']), insert_end + 1)

for i, turn in enumerate(sample['dialog'][start_idx:end_idx]):
    turn_idx = start_idx + i
    marker = " ← 위반!" if turn_idx == sample['violation_turn_index'] else ""
    print(f'Turn {turn_idx}: [{turn["speaker"]}]{marker}')
    print(f'  {turn["content"]}')
    print()
