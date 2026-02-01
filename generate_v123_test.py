"""
V1/V2/V3 각 5개씩 생성 테스트 (4턴 고정 구조)
"""
from src.generation.insertion_generator import generate_v1_v3_with_insertion
from src.utils import load_json, save_json

print("="*60)
print("V1/V2/V3 Generation Test (4-turn fixed structure)")
print("="*60)

esconv = load_json('ESConv.json')

for violation_type in ['V1', 'V2', 'V3']:
    print(f"\n{'='*60}")
    print(f"Generating {violation_type} (5 sessions)")
    print(f"{'='*60}")
    
    result = generate_v1_v3_with_insertion(
        esconv, 
        violation_type, 
        5,  # 5 sessions
        42 + ord(violation_type[1]),  # Different seed for each
        'gpt-4o-mini', 
        0.7
    )
    
    output_file = f'data/pilot/{violation_type.lower()}.json'
    save_json(result, output_file)
    
    print(f"\n{'─'*60}")
    print(f"Summary for {violation_type}:")
    print(f"{'─'*60}")
    for i, d in enumerate(result):
        prefix_len = d.get('prefix_length', 0)
        insertion_len = d.get('insertion_length', 0)
        total_len = len(d['dialog'])
        violation_idx = d['violation_turn_index']
        
        print(f"  [{i+1}] Session {d['session_id']}:")
        print(f"      Prefix: {prefix_len} turns")
        print(f"      Insertion: {insertion_len} turns")
        print(f"      Total: {total_len} turns")
        print(f"      Violation at: Turn {violation_idx}")
        
        # 마지막 4턴 (삽입부) 간략히 출력
        print(f"      Insertion preview:")
        for j, t in enumerate(d['dialog'][-4:], start=total_len-4):
            speaker = t['speaker'][:4].upper()
            content = t['content'][:60] + '...' if len(t['content']) > 60 else t['content']
            print(f"        [T{j}] {speaker}: {content}")
        print()
    
    print(f"✅ Saved {len(result)} {violation_type} sessions to {output_file}\n")

print(f"\n{'='*60}")
print("Generation Complete!")
print(f"{'='*60}")
print("Generated files:")
print("  - data/pilot/v1.json (5 sessions)")
print("  - data/pilot/v2.json (5 sessions)")
print("  - data/pilot/v3.json (5 sessions)")
