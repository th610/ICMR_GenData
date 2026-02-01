"""
Generate V4-V5 Sessions (Large Scale with Checkpoints)
150개씩 생성, 50개마다 체크포인트 저장
"""
from pathlib import Path
from src.utils import save_json
from src.generation.v4_v5_generator import generate_v4_v5_session, validate_v4_v5_session


def generate_with_checkpoints(violation_type, total_count, checkpoint_interval=50, start_idx=0, resume_from_checkpoint=None):
    """
    Generate sessions with checkpoint saving
    
    Args:
        violation_type: 'V4' or 'V5'
        total_count: Total sessions to generate
        checkpoint_interval: Save every N sessions
        start_idx: Start from this index (default 0)
        resume_from_checkpoint: Path to checkpoint file to resume from
    """
    output_dir = Path("data/pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing sessions if resuming
    if resume_from_checkpoint:
        from src.utils import load_json
        sessions = load_json(resume_from_checkpoint)
        print(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")
        print(f"   Loaded {len(sessions)} existing sessions")
    else:
        sessions = []
    
    error_count = 0
    
    print(f"\n{'='*80}")
    print(f"Generating {violation_type}: {total_count} sessions (starting from {start_idx})")
    print(f"Checkpoint: every {checkpoint_interval} sessions")
    print(f"{'='*80}\n")
    
    for i in range(start_idx, total_count):
        try:
            print(f"[{i+1}/{total_count}] Generating {violation_type}_{i:04d}...", end=" ")
            
            session = generate_v4_v5_session(
                violation_type,
                i,
                model="gpt-4o-mini",
                temperature=0.7
            )
            
            # Validate
            is_valid = validate_v4_v5_session(session, violation_type)
            num_turns = len(session['dialog'])
            
            if is_valid:
                sessions.append(session)
                print(f"✅ {num_turns} turns")
            else:
                print(f"❌ Invalid")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            error_count += 1
            continue
        
        # Checkpoint save
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"{violation_type.lower()}_checkpoint_{i+1:04d}.json"
            save_json(sessions, str(checkpoint_path))
            
            avg_length = sum(len(s['dialog']) for s in sessions) / len(sessions) if sessions else 0
            
            print(f"\n{'─'*80}")
            print(f"📍 Checkpoint {i+1}/{total_count}")
            print(f"   Generated: {len(sessions)}, Errors: {error_count}")
            print(f"   Avg length: {avg_length:.1f} turns")
            print(f"   💾 Saved: {checkpoint_path.name}")
            print(f"{'─'*80}\n")
    
    # Final save
    final_path = output_dir / f"{violation_type.lower()}_full_{total_count}.json"
    save_json(sessions, str(final_path))
    
    print(f"\n{'='*80}")
    print(f"{violation_type} Generation Complete")
    print(f"{'='*80}")
    print(f"  Total generated: {len(sessions)}/{total_count}")
    print(f"  Errors: {error_count}")
    
    if sessions:
        avg_length = sum(len(s['dialog']) for s in sessions) / len(sessions)
        print(f"  Avg length: {avg_length:.1f} turns")
    
    print(f"  💾 Final: {final_path}")
    print(f"{'='*80}\n")
    
    return sessions


def main():
    print("=" * 80)
    print("V4-V5 Large Scale Generation (150개씩)")
    print("=" * 80)
    
    # Generate V4 (resume from checkpoint 50)
    v4_sessions = generate_with_checkpoints(
        'V4', 
        total_count=150, 
        checkpoint_interval=50,
        start_idx=50,
        resume_from_checkpoint='data/pilot/v4_checkpoint_0050.json'
    )
    
    # Generate V5
    v5_sessions = generate_with_checkpoints('V5', total_count=150, checkpoint_interval=50)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"  V4: {len(v4_sessions)}/150 sessions")
    print(f"  V5: {len(v5_sessions)}/150 sessions")
    print("\n✅ All Done!")


if __name__ == "__main__":
    main()
