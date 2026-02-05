import json

# V1 데이터 확인
v1 = json.load(open('.archive/2026-02-02_generated/generated/v1_240.json', encoding='utf-8'))

print('V1 첫 샘플 필드:')
for k in v1[0].keys():
    print(f'  - {k}')

print()

if 'session_id' in v1[0]:
    print(f'session_id 있음: {v1[0]["session_id"]}')
else:
    print('session_id 없음')
    print()
    print('situation으로 ESConv 매칭 가능')
    print(f'situation: {v1[0]["situation"]}')
