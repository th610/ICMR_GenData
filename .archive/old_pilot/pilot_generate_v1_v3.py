"""
STEP 2 (Revised): Generate V1-V3 with Insertion Strategy
ESConv prefix (12-20 turns) + 4-6 turn insertion
"""
import yaml
from pathlib import Path
from src.utils import load_json, save_json
from src.generation.insertion_generator import (
    generate_v1_v3_with_insertion,
    validate_insertion_session
)


def main():
    print("=" * 60)
    print("STEP 2: Generate V1-V3 with Insertion Strategy")
    print("=" * 60)
    
    # Load config
    with open('configs/pilot.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    num_per_class = config['v1_v3']['num_per_class']
    seed = config['seed']
    model = config['llm']['model']
    temperature = config['llm']['temperature']
    
    print(f"\nLoading ESConv...")
    esconv_sessions = load_json(config['paths']['raw_data'])
    print(f"  Total ESConv sessions: {len(esconv_sessions)}")
    
    # Generate V1, V2, V3
    for violation_type in ['V1', 'V2', 'V3']:
        print(f"\n{'-'*60}")
        print(f"Generating {violation_type} sessions (Insertion Strategy)...")
        print(f"  ESConv prefix: 12-20 turns (random)")
        print(f"  Insertion: 4-6 turns")
        print(f"  Violation: Last Supporter turn only")
        print(f"  Model: {model}, Temperature: {temperature}")
        print(f"{'-'*60}")
        
        sessions = generate_v1_v3_with_insertion(
            esconv_sessions,
            violation_type,
            num_per_class,
            seed + ord(violation_type[1]),  # 각 클래스 다른 시드
            model,
            temperature
        )
        
        # Validate
        print(f"\nValidating sessions...")
        for session in sessions:
            session_id = session['session_id']
            num_turns = len(session['dialog'])
            prefix_len = session.get('prefix_length', 'N/A')
            last_speaker = session['dialog'][-1]['speaker']
            is_valid = validate_insertion_session(session, violation_type)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {session_id}: prefix={prefix_len}, total={num_turns}, last={last_speaker}")
        
        # Save
        output_path = Path(config['paths'][violation_type.lower()])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(sessions)} sessions to {output_path}...")
        save_json(sessions, str(output_path))
        print(f"  Saved: {output_path}")
        
        # Summary
        avg_length = sum(len(s['dialog']) for s in sessions) / len(sessions) if sessions else 0
        avg_prefix = sum(s.get('prefix_length', 0) for s in sessions) / len(sessions) if sessions else 0
        avg_insertion = avg_length - avg_prefix
        print(f"  Avg prefix: {avg_prefix:.1f} turns")
        print(f"  Avg insertion: {avg_insertion:.1f} turns")
        print(f"  Avg total: {avg_length:.1f} turns")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  V1 sessions: {num_per_class} (saved to {config['paths']['v1']})")
    print(f"  V2 sessions: {num_per_class} (saved to {config['paths']['v2']})")
    print(f"  V3 sessions: {num_per_class} (saved to {config['paths']['v3']})")
    print(f"  Model: {model}, Temperature: {temperature}")
    print(f"  Strategy: ESConv prefix + Insertion")
    print(f"  Note: V1/V2/V3 use different ESConv source sessions")
    print(f"\n✅ Step 2 완료!")


if __name__ == '__main__':
    main()
