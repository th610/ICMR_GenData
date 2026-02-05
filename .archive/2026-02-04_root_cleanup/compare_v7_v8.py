import json

v7 = json.load(open('test_judge_v7_100.json', encoding='utf-8'))
v8 = json.load(open('test_judge_v8_100.json', encoding='utf-8'))

v7_violations = [r for r in v7 if r['label'] != 'normal']

print(f'V7 위반 케이스 {len(v7_violations)}개:\n')

for r in v7_violations:
    print(f'Session {r["session_id"]}: {r["label"].upper()} (윈도우: {r["window_length"]}턴)')
    print(f'  이유: {r["reason"][:100]}...')
    
    # 같은 세션의 V8 결과 찾기
    v8_result = next((v for v in v8 if v['session_id'] == f'sample_{r["session_id"]+1}'), None)
    if v8_result:
        print(f'  V8 결과: {v8_result["label"].upper()}')
        print(f'  V8 이유: {v8_result["reason"][:100]}...')
    print()
