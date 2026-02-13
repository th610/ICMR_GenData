"""
Step 1: Assign ESConv Sessions to Labels
=========================================
1300개의 고유한 ESConv 세션을 6개의 레이블에 할당합니다:
- Normal: 525 sessions
- V1: 200 sessions
- V2: 200 sessions
- V3: 200 sessions
- V4: 100 sessions
- V5: 75 sessions (separate prefix pool)

Output: generation/outputs/assigned/ESConv_*_assigned.json
"""
import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    # Load prefixes
    with open("ESConv_normal_prefixes.json", 'r', encoding='utf-8') as f:
        normal_prefixes = json.load(f)
    
    with open("ESConv_v5_prefixes.json", 'r', encoding='utf-8') as f:
        v5_prefixes = json.load(f)
    
    print(f"Available normal prefixes: {len(normal_prefixes)}")
    print(f"Available V5 prefixes: {len(v5_prefixes)}")
    
    # Sample 1225 unique sessions from normal_prefixes
    random.seed(42)
    sampled_1225 = random.sample(normal_prefixes, 1225)
    
    # Split into labels
    splits = {
        "Normal": sampled_1225[0:525],
        "V1": sampled_1225[525:725],
        "V2": sampled_1225[725:925],
        "V3": sampled_1225[925:1125],
        "V4": sampled_1225[1125:1225],
        "V5": v5_prefixes
    }
    
    # Verify uniqueness
    print("\n" + "="*70)
    print("DISTRIBUTION (NO OVERLAP)")
    print("="*70)
    
    all_sessions = set()
    for label, prefixes in splits.items():
        sessions = {p["esconv_session_id"] for p in prefixes}
        all_sessions.update(sessions)
        print(f"{label:8s}: {len(prefixes):4d} sessions")
    
    print("-"*70)
    print(f"{'TOTAL':8s}: {sum(len(p) for p in splits.values()):4d} samples")
    print(f"Unique sessions: {len(all_sessions)}")
    
    # Save assignments
    output_dir = Path("generation/outputs/assigned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for label, prefixes in splits.items():
        if label == "Normal":
            filename = output_dir / "ESConv_normal_assigned.json"
        else:
            filename = output_dir / f"ESConv_{label.lower()}_assigned.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prefixes, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {filename}")
    
    print("\n✓ Session assignment complete!")

if __name__ == "__main__":
    main()
