import json

data = json.load(open('data/pilot/judge_results_summary_window.json', encoding='utf-8'))
r = data['results']

print("=== NORMAL Judgments ===")
for x in r['normal']:
    match = "✅" if x['matches'] else "❌"
    print(f"{match} {x['session_id']}: {x['expected_label']} → {x['predicted_label']}")
    if not x['matches']:
        print(f"   Reason: {x['reason'][:80]}...")

print("\n=== V1 Judgments ===")
for x in r['v1']:
    match = "✅" if x['matches'] else "❌"
    print(f"{match} {x['session_id']}: {x['expected_label']} → {x['predicted_label']}")
    if not x['matches']:
        print(f"   Reason: {x['reason'][:80]}...")

print("\n=== V3 Judgments ===")
for x in r['v3']:
    match = "✅" if x['matches'] else "❌"
    print(f"{match} {x['session_id']}: {x['expected_label']} → {x['predicted_label']}")
    print(f"   Last Supporter: {x['dialog'][-1]['content'][:100]}...")
    print(f"   Reason: {x['reason'][:100]}...")
