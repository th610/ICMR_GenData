import json

v10 = json.load(open('test_judge_v10_100.json', encoding='utf-8'))
r = v10[28]

print('=== Session 28 STATE MEMORY 생성 문제 ===\n')
print('Label:', r['label'])
print()
print('STATE MEMORY:')
print(r.get('state_memory', '없음'))
print()
print('Dialog 중 bridge 포함 턴:')
for i, t in enumerate(r['dialog']):
    if 'bridge' in t['content'].lower():
        print(f'{i+1}. {t["speaker"]}: {t["content"].strip()[:150]}')
print()

# Check which turns were used for STATE MEMORY generation
print(f'전체 대화: {len(r["dialog"])}턴')
print(f'STATE MEMORY 생성에 사용된 턴: 1-{len(r["dialog"])-7}턴 (마지막 7턴 제외)')
print(f'bridge가 있는 턴 8은 STATE MEMORY 생성 범위에 {"포함" if 8 <= len(r["dialog"])-7 else "미포함"}')
