import json

esconv = json.load(open('ESConv.json', encoding='utf-8'))
v1 = json.load(open('.archive/2026-02-02_generated/generated/v1_240.json', encoding='utf-8'))

sample = v1[0]
idx = sample['source_esconv_idx']

print('='*60)
print('ESConv 매칭 확인')
print('='*60)
print(f'V1 session_id: {sample["session_id"]}')
print(f'source_esconv_idx: {idx}')
print()

print('ESConv[163]:')
print(f'  situation: {esconv[idx]["situation"]}')
print(f'  dialog 길이: {len(esconv[idx]["dialog"])}턴')
print()

print('V1 샘플:')
print(f'  situation: {sample["situation"]}')
print(f'  dialog 길이: {len(sample["dialog"])}턴')
print(f'  prefix_length: {sample["prefix_length"]}')
print(f'  insertion_length: {sample["insertion_length"]}')
print()

# Prefix 비교
print('='*60)
print('Prefix 비교 (첫 3턴)')
print('='*60)

print('ESConv[163]:')
for i, turn in enumerate(esconv[idx]['dialog'][:3]):
    print(f'  Turn {i}: [{turn["speaker"]}] {turn["content"][:80]}...')

print()
print('V1 샘플:')
for i, turn in enumerate(sample['dialog'][:3]):
    print(f'  Turn {i}: [{turn["speaker"]}] {turn["content"][:80]}...')

print()

# ESConv에서 같은 situation 찾기
print('='*60)
print('V1 situation으로 ESConv 검색')
print('='*60)

v1_situation = sample["situation"]
found = False
for i, sess in enumerate(esconv):
    if sess['situation'] == v1_situation:
        print(f'✅ 발견! ESConv[{i}]')
        print(f'  situation: {sess["situation"]}')
        print(f'  dialog 길이: {len(sess["dialog"])}턴')
        found = True
        
        # Prefix 비교
        print()
        print('  첫 3턴:')
        for j, turn in enumerate(sess['dialog'][:3]):
            print(f'    Turn {j}: [{turn["speaker"]}] {turn["content"][:60]}...')
        break

if not found:
    print('❌ 못 찾음 - situation이 생성된 것일 수도 있음')
