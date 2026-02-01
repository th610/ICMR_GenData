"""
Generate V1-V3 Large Scale (from Normal pool)
- Normal pool에서 샘플링
- V1: 240개, V2: 160개, V3: 200개
- Normal: 400개 그대로 유지
- 50개마다 체크포인트 저장
- 로그 파일 생성
"""
import random
from pathlib import Path
from datetime import datetime
from src.utils import load_json, save_json
from src.generation.insertion_generator import generate_v1_v3_with_insertion


def setup_logging(log_dir="logs"):
    """로그 파일 설정"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"generate_v1_v3_{timestamp}.log"
    
    return log_file


def log_and_print(message, log_file):
    """화면 출력 + 로그 파일 저장"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def main():
    # Setup
    log_file = setup_logging()
    output_dir = Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_and_print("=" * 80, log_file)
    log_and_print(f"V1-V3 Large Scale Generation", log_file)
    log_and_print(f"Started: {timestamp}", log_file)
    log_and_print("=" * 80, log_file)
    
    # Load Normal pool
    normal_pool_path = "data/esconv_judged/normal/normal_sessions.json"
    log_and_print(f"\n📂 Loading Normal pool: {normal_pool_path}", log_file)
    normal_pool = load_json(normal_pool_path)
    log_and_print(f"   Total Normal sessions: {len(normal_pool)}", log_file)
    
    # Split Normal pool
    random.seed(42)
    shuffled = random.sample(normal_pool, len(normal_pool))
    
    normal_for_label = shuffled[:400]
    normal_for_v1 = shuffled[400:640]  # 240
    normal_for_v2 = shuffled[640:800]  # 160
    normal_for_v3 = shuffled[800:1000] # 200
    
    log_and_print(f"\n📊 Normal pool split:", log_file)
    log_and_print(f"   Normal label: {len(normal_for_label)}", log_file)
    log_and_print(f"   V1 source: {len(normal_for_v1)}", log_file)
    log_and_print(f"   V2 source: {len(normal_for_v2)}", log_file)
    log_and_print(f"   V3 source: {len(normal_for_v3)}", log_file)
    log_and_print(f"   Unused: {len(normal_pool) - 1000}", log_file)
    
    # Save Normal sessions
    normal_output = output_dir / "normal_400.json"
    save_json(normal_for_label, str(normal_output))
    log_and_print(f"\n💾 Saved Normal: {normal_output}", log_file)
    
    # Generate V1, V2, V3
    configs = [
        ('V1', normal_for_v1, 240),
        ('V2', normal_for_v2, 160),
        ('V3', normal_for_v3, 200)
    ]
    
    for violation_type, source_pool, target_count in configs:
        log_and_print(f"\n{'='*80}", log_file)
        log_and_print(f"Generating {violation_type}: {target_count} sessions", log_file)
        log_and_print(f"Source pool: {len(source_pool)} Normal sessions", log_file)
        log_and_print(f"Checkpoint: every 50 sessions", log_file)
        log_and_print(f"{'='*80}\n", log_file)
        
        # Check for existing checkpoint
        generated = []
        start_idx = 0
        checkpoint_interval = 50
        
        # Find latest checkpoint (search backwards from target)
        latest_checkpoint = None
        for num in range(target_count, 0, -1):
            if num % checkpoint_interval == 0:
                checkpoint_path = output_dir / f"{violation_type.lower()}_checkpoint_{num:04d}.json"
                if checkpoint_path.exists():
                    latest_checkpoint = checkpoint_path
                    break
        
        if latest_checkpoint:
            log_and_print(f"📂 Found checkpoint: {latest_checkpoint.name}", log_file)
            generated = load_json(str(latest_checkpoint))
            start_idx = len(generated)
            log_and_print(f"   Resuming from index {start_idx} ({len(generated)} sessions loaded)", log_file)
            log_and_print("", log_file)
        
        for i in range(start_idx, target_count):
            try:
                msg = f"[{i+1}/{target_count}] Generating {violation_type}_{i:04d}..."
                log_and_print(msg, log_file)
                
                # Generate one session
                sessions = generate_v1_v3_with_insertion(
                    esconv_sessions=source_pool,
                    violation_type=violation_type,
                    num_sessions=1,
                    seed=42 + i,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                
                if sessions:
                    session = sessions[0]
                    session['session_id'] = f"{violation_type.lower()}_{i:04d}"
                    generated.append(session)
                    
                    num_turns = len(session['dialog'])
                    msg = f"   ✅ {num_turns} turns"
                    log_and_print(msg, log_file)
                else:
                    msg = f"   ❌ Generation failed"
                    log_and_print(msg, log_file)
                    
            except Exception as e:
                msg = f"   ❌ Error: {e}"
                log_and_print(msg, log_file)
                continue
            
            # Checkpoint save
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = output_dir / f"{violation_type.lower()}_checkpoint_{i+1:04d}.json"
                save_json(generated, str(checkpoint_path))
                
                avg_length = sum(len(s['dialog']) for s in generated) / len(generated) if generated else 0
                
                log_and_print(f"\n{'─'*80}", log_file)
                log_and_print(f"📍 Checkpoint {i+1}/{target_count}", log_file)
                log_and_print(f"   Generated: {len(generated)}", log_file)
                log_and_print(f"   Avg length: {avg_length:.1f} turns", log_file)
                log_and_print(f"   💾 Saved: {checkpoint_path.name}", log_file)
                log_and_print(f"{'─'*80}\n", log_file)
        
        # Final save
        final_path = output_dir / f"{violation_type.lower()}_{target_count}.json"
        save_json(generated, str(final_path))
        
        log_and_print(f"\n{'='*80}", log_file)
        log_and_print(f"{violation_type} Generation Complete", log_file)
        log_and_print(f"{'='*80}", log_file)
        log_and_print(f"   Total: {len(generated)}/{target_count}", log_file)
        
        if generated:
            avg_length = sum(len(s['dialog']) for s in generated) / len(generated)
            log_and_print(f"   Avg length: {avg_length:.1f} turns", log_file)
        
        log_and_print(f"   💾 Final: {final_path}", log_file)
        log_and_print(f"{'='*80}\n", log_file)
    
    # Final summary
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_and_print("\n" + "=" * 80, log_file)
    log_and_print("All Generation Complete", log_file)
    log_and_print("=" * 80, log_file)
    log_and_print(f"Finished: {end_timestamp}", log_file)
    log_and_print(f"Log saved: {log_file}", log_file)
    log_and_print("\n✅ Done!", log_file)


if __name__ == "__main__":
    main()
