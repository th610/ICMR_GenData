"""
STEP 3: Generate V4-V5 Sessions (Pilot)
완전 신규 멀티턴 대화 생성
"""
import yaml
from pathlib import Path
from src.utils import save_json
from src.generation.v4_v5_generator import prepare_v4_v5_sessions, validate_v4_v5_session


def main():
    print("=" * 60)
    print("STEP 3: Generate V4-V5 Sessions (Pilot)")
    print("=" * 60)
    
    # Load config
    with open('configs/pilot.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    num_per_class = config['v4_v5']['num_per_class']
    model = config['llm']['model']
    temperature = config['llm']['temperature']
    
    # Generate V4, V5
    for violation_type in ['V4', 'V5']:
        print(f"\n{'-'*60}")
        print(f"Generating {violation_type} sessions...")
        print(f"  Method: Full multiturn generation")
        print(f"  Model: {model}")
        print(f"{'-'*60}")
        
        sessions = prepare_v4_v5_sessions(
            violation_type,
            num_per_class,
            model,
            temperature
        )
        
        # Validate
        print(f"\nValidating sessions...")
        for session in sessions:
            session_id = session['session_id']
            num_turns = len(session['dialog'])
            last_speaker = session['dialog'][-1]['speaker']
            is_valid = validate_v4_v5_session(session, violation_type)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {session_id}: {num_turns} turns, last={last_speaker}")
        
        # Save
        output_path = Path(config['paths'][violation_type.lower()])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(sessions)} sessions to {output_path}...")
        save_json(sessions, str(output_path))
        print(f"  Saved: {output_path}")
        
        # Summary
        avg_length = sum(len(s['dialog']) for s in sessions) / len(sessions)
        print(f"  Avg length: {avg_length:.1f} turns")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  V4 sessions: {num_per_class} (saved to {config['paths']['v4']})")
    print(f"  V5 sessions: {num_per_class} (saved to {config['paths']['v5']})")
    print(f"  Model: {model}, Temperature: {temperature}")
    print(f"  Method: Full multiturn generation")
    print(f"\n✅ Step 3 완료!")


if __name__ == "__main__":
    main()
