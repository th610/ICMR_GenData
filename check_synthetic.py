import json

# Load synthetic data
with open('data/sessions_synth_50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*60)
print("합성 데이터 위반 주입 샘플 확인 (처음 3개)")
print("="*60)

for i, session in enumerate(data[:3]):
    inj = session.get('injected_violation', {})
    
    print(f"\n[샘플 {i+1}]")
    print(f"Session ID: {session.get('session_id')}")
    print(f"Violation Type: {inj.get('type')}")
    print(f"Turn ID: {inj.get('turn_id')} (전체 대화에서 {inj.get('turn_id')}번째 턴)")
    print(f"Supporter Utterance Index: {inj.get('supporter_utterance_index')} ({inj.get('supporter_utterance_index')}번째 supporter 응답)")
    
    print(f"\n[원본 응답]")
    original = inj.get('original_text', '')
    print(original if len(original) < 200 else original[:200] + "...")
    
    print(f"\n[리라이트 응답]")
    rewritten = inj.get('rewritten_text', '')
    print(rewritten if len(rewritten) < 200 else rewritten[:200] + "...")
    
    print(f"\n[변경 이유]")
    print(inj.get('rationale', ''))
    print("-" * 60)

# 위반 유형별 개수
print("\n" + "="*60)
print("전체 위반 분포")
print("="*60)
violation_counts = {}
for session in data:
    v_type = session.get('injected_violation', {}).get('type', 'unknown')
    violation_counts[v_type] = violation_counts.get(v_type, 0) + 1

for v_type in sorted(violation_counts.keys()):
    print(f"  {v_type}: {violation_counts[v_type]}개")
