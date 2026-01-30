"""
STEP 2: Generate V1-V3 Sessions (Pilot)
ESConv 세션의 마지막 turn을 LLM으로 rewrite
"""
import yaml
from pathlib import Path
from src.utils import load_json, save_json
from src.generation.v1_v3_rewriter import prepare_v1_v3_sessions, validate_v1_v3_session


def main():
    print("=" * 60)
    print("STEP 2: Generate V1-V3 Sessions (Pilot)")
    print("=" * 60)
    
    # Load config
    with open('configs/pilot.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    num_per_class = config['v1_v3']['num_per_class']
    length_range = tuple(config['v1_v3']['target_length'])
    seed = config['seed']
    model = config['llm']['model']
    temperature = config['llm']['temperature']
    
    print(f"\nLoading ESConv...")
    esconv_sessions = load_json(config['paths']['raw_data'])
    print(f"  Total ESConv sessions: {len(esconv_sessions)}")
    
    # Generate V1, V2, V3
    for violation_type in ['V1', 'V2', 'V3']:
        print(f"\n{'-'*60}")
        print(f"Generating {violation_type} sessions...")
        print(f"  Target length: {length_range[0]}-{length_range[1]} turns")
        print(f"  Last turn: Rewrite with LLM ({model})")
        print(f"{'-'*60}")
        
        sessions = prepare_v1_v3_sessions(
            esconv_sessions,
            violation_type,
            num_per_class,
            length_range,
            seed,
            model,
            temperature
        )
        
        # Validate
        print(f"\nValidating sessions...")
        for session in sessions:
            session_id = session['session_id']
            num_turns = len(session['dialog'])
            last_speaker = session['dialog'][-1]['speaker']
            is_valid = validate_v1_v3_session(session, violation_type)
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
    print(f"  V1 sessions: {num_per_class} (saved to {config['paths']['v1']})")
    print(f"  V2 sessions: {num_per_class} (saved to {config['paths']['v2']})")
    print(f"  V3 sessions: {num_per_class} (saved to {config['paths']['v3']})")
    print(f"  Model: {model}, Temperature: {temperature}")
    print(f"  Sampling: Random (seed={seed})")
    print(f"\n✅ Step 2 완료!")


if __name__ == "__main__":
    main()
