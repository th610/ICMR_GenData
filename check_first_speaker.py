import json

# ESConv 원본 확인
esconv = json.load(open('ESConv.json', encoding='utf-8'))

first_speakers = [s['dialog'][0]['speaker'] for s in esconv]
seeker_first = sum(1 for sp in first_speakers if sp == 'seeker')
supporter_first = sum(1 for sp in first_speakers if sp == 'supporter')

print("=== ESConv 원본 ===")
print(f"Seeker first: {seeker_first} ({seeker_first*100/len(esconv):.1f}%)")
print(f"Supporter first: {supporter_first} ({supporter_first*100/len(esconv):.1f}%)")
print(f"Total: {len(esconv)}")

# V1/V2/V3 확인
for vtype in ['v1', 'v2', 'v3']:
    data = json.load(open(f'data/pilot/{vtype}.json', encoding='utf-8'))
    first_speakers = [s['dialog'][0]['speaker'] for s in data]
    seeker_first = sum(1 for sp in first_speakers if sp == 'seeker')
    supporter_first = sum(1 for sp in first_speakers if sp == 'supporter')
    
    print(f"\n=== {vtype.upper()} ===")
    print(f"Seeker first: {seeker_first}")
    print(f"Supporter first: {supporter_first}")
    print(f"Total: {len(data)}")
    
    if supporter_first > 0:
        print("\n  Supporter로 시작하는 세션:")
        for s in data:
            if s['dialog'][0]['speaker'] == 'supporter':
                print(f"    - {s['session_id']}: {s['dialog'][0]['content'][:60]}...")
