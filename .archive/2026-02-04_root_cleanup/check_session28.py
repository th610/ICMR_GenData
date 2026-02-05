import json

v7 = json.load(open('test_judge_v7_100.json', encoding='utf-8'))
v8 = json.load(open('test_judge_v8_100.json', encoding='utf-8'))

# Session 28 (V5 case)
r7 = v7[28]
r8 = v8[28]

print('=== Session 28 (V5 케이스) ===\n')
print(f'V7: {r7["label"].upper()}, 윈도우 {r7["window_length"]}턴')
print(f'V8: {r8["label"].upper()}, 윈도우 {r8["window_length"]}턴\n')

print('전체 대화 길이:', len(r7['dialog']), '턴')
print('V8 SUMMARY:', r8.get('summary', '(없음)')[:200])
print()

# "bridge" 검색
for i, turn in enumerate(r7['dialog'], 1):
    if 'bridge' in turn['content'].lower() or 'jump' in turn['content'].lower():
        print(f'[턴 {i}] {turn["speaker"]}: {turn["content"].strip()[:100]}')

print('\n=== V8이 실제로 본 대화 (recent_dialog = 8-14턴) ===\n')
recent_dialog = r8['dialog'][-7:]
for i, turn in enumerate(recent_dialog, 8):
    print(f'[턴 {i}] {turn["speaker"]}: {turn["content"].strip()[:100]}')
