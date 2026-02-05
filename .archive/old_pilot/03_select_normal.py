"""
Step 1: Prepare Normal sessions from ESConv
- Sample 5 sessions
- Cut to 15~20 turns
- Last turn = Supporter
- Save to data/pilot/normal/
"""
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_json, save_json, load_yaml
from src.generation.normal_cutter import prepare_normal_sessions, validate_normal_session


def main():
    print(f"\n{'='*60}")
    print("STEP 1: Prepare Normal Sessions (Pilot)")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml('configs/pilot.yaml')
    
    # Load ESConv
    print("Loading ESConv...")
    esconv_path = config['normal']['source']
    esconv_sessions = load_json(esconv_path)
    print(f"  Total ESConv sessions: {len(esconv_sessions)}")
    
    # Prepare Normal sessions
    num_sessions = config['normal']['num_sessions']
    target_length = tuple(config['normal']['target_length'])
    seed = config['seed']
    
    print(f"\nPreparing {num_sessions} Normal sessions...")
    print(f"  Target length: {target_length[0]}-{target_length[1]} turns")
    print(f"  Last turn: Supporter (required)")
    
    normal_sessions = prepare_normal_sessions(
        esconv_sessions,
        num_sessions,
        target_length,
        seed
    )
    
    # Validate
    print(f"\nValidating sessions...")
    all_valid = True
    for i, session in enumerate(normal_sessions):
        valid = validate_normal_session(session)
        if not valid:
            print(f"  ❌ Session {i} failed validation")
            all_valid = False
        else:
            dialog_len = len(session['dialog'])
            last_speaker = session['dialog'][-1]['speaker']
            print(f"  ✅ {session['session_id']}: {dialog_len} turns, last={last_speaker}")
    
    if not all_valid:
        print("\n❌ Validation failed!")
        return
    
    # Save as single JSON file
    output_path = Path(config['paths']['normal'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(normal_sessions)} sessions to {output_path}...")
    save_json(normal_sessions, str(output_path))
    print(f"  Saved: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total sessions: {len(normal_sessions)}")
    print(f"  Avg length: {sum(len(s['dialog']) for s in normal_sessions) / len(normal_sessions):.1f} turns")
    print(f"  Sampling: Random (seed={seed})")
    print(f"  Output: {output_path}")
    print(f"\n✅ Step 1 완료!")


if __name__ == "__main__":
    main()
