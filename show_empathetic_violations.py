import json

data = json.load(open('data/external/empathetic_judged.json'))
violations = [r for r in data['results'] if r['judge_label'] != 'normal']

print(f"Total violations: {len(violations)}\n")
print("="*60)

for v in violations:
    print(f"[{v['judge_label'].upper()}] Context: {v['context']}")
    print(f"Response: {v['last_response']}")
    print(f"Reason: {v['judge_reason']}")
    print("-"*60 + "\n")
