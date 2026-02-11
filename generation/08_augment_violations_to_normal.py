"""
Step 8: Augment Data - Violation → Normal
===========================================
위반 샘플 555개의 Turn 4를 Normal로 재생성하여 데이터 증강

Input:  data/final/train_1000.json (445 Normal + 555 Violations)
Output: data/final/train_1600_augmented.json (1000 original + 600 augmented)

Augmentation Strategy:
- 원본 1000개 유지
- 위반 555개 → Normal Turn 4 생성 → 555개 추가
- Session ID: augmented_1301 ~ augmented_1855
- augmentation_meta로 원본 추적
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts_augmentation import (
    AUGMENTATION_SYSTEM,
    build_augmentation_prompt
)


def main():
    input_file = Path("data/final/train_1000.json")
    output_file = Path("data/final/train_1600_augmented.json")
    
    print("="*70)
    print("Data Augmentation: Violation → Normal")
    print("="*70)
    
    # Load train data
    with open(input_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    samples = train_data['samples']
    
    # Filter violation samples only
    violation_samples = [s for s in samples if s['primary_label'] != 'Normal']
    
    print(f"\nOriginal samples: {len(samples)}")
    print(f"Violation samples to augment: {len(violation_samples)}")
    
    # Show distribution
    dist = Counter([s['primary_label'] for s in violation_samples])
    print("\nViolation distribution:")
    for label in ['V1', 'V2', 'V3', 'V4', 'V5']:
        print(f"  {label}: {dist[label]}")
    
    # Initialize OpenAI client
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.8,  # Slight variation for diversity
        max_tokens=300,
        timeout=60,
        max_retries=2
    )
    
    # Augment violation samples
    augmented_samples = []
    augmented_id_start = 1301
    
    print(f"\n{'='*70}")
    print("Starting Augmentation...")
    print(f"{'='*70}\n")
    
    for i, sample in enumerate(violation_samples):
        violation_type = sample['primary_label']
        
        print(f"[{i+1}/{len(violation_samples)}] {violation_type} → Normal (Session: {sample.get('esconv_session_id', 'unknown')})")
        
        # Extract data
        situation = sample.get('situation', '')
        prefix_dialog = sample.get('prefix_dialog', [])
        generated_dialog = sample.get('generated_dialog', [])
        
        if len(generated_dialog) < 4:
            print(f"  ✗ Skip: generated_dialog has only {len(generated_dialog)} turns")
            continue
        
        # Build prompt
        try:
            user_prompt = build_augmentation_prompt(
                violation_type=violation_type,
                situation=situation,
                prefix_dialog=prefix_dialog,
                generated_turns_123=generated_dialog[:3]  # Turn 0-2
            )
            
            # Call GPT
            result = client.call(
                system_prompt=AUGMENTATION_SYSTEM,
                user_prompt=user_prompt,
                retry_message="Return ONLY valid JSON: {\"supporter_response\": \"...\"}"
            )
            
            new_turn4_text = result.get('supporter_response', '')
            
            if not new_turn4_text:
                print(f"  ✗ Error: Empty response from GPT")
                continue
            
            # Build augmented sample
            augmented_sample = {
                "esconv_session_id": f"augmented_{augmented_id_start + i}",
                "situation": situation,
                "prefix_dialog": prefix_dialog,
                "generated_dialog": [
                    generated_dialog[0],  # Turn 0 (seeker)
                    generated_dialog[1],  # Turn 1 (supporter)
                    generated_dialog[2],  # Turn 2 (seeker)
                    {
                        "speaker": "supporter",
                        "text": new_turn4_text,
                        "annotation_text": "Normal (augmented from violation)"
                    }
                ],
                "primary_label": "Normal",
                "generation_method": "augmentation_violation_to_normal",
                "augmentation_meta": {
                    "source": "train_violation_to_normal",
                    "original_session_id": sample.get('esconv_session_id', ''),
                    "original_label": violation_type,
                    "original_turn4": generated_dialog[3].get('text', ''),
                    "augmented_turn4": new_turn4_text,
                    "augmentation_date": datetime.now().strftime("%Y-%m-%d")
                }
            }
            
            # Copy optional fields
            if 'violation_turn_index' in sample:
                augmented_sample['augmentation_meta']['original_violation_turn_index'] = sample['violation_turn_index']
            if 'violation_reason' in sample:
                augmented_sample['augmentation_meta']['original_violation_reason'] = sample['violation_reason']
            
            augmented_samples.append(augmented_sample)
            print(f"  ✓ Success (new Turn 4: {new_turn4_text[:60]}...)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            continue
        
        # Progress report
        if (i + 1) % 50 == 0:
            print(f"\nPROGRESS: {i+1}/{len(violation_samples)} - Success: {len(augmented_samples)}\n")
    
    # Combine original + augmented
    final_samples = samples + augmented_samples  # 1000 + ~555
    
    # Update metadata
    final_data = {
        "metadata": {
            "total_samples": len(final_samples),
            "original_samples": len(samples),
            "augmented_samples": len(augmented_samples),
            "split": "train_augmented",
            "distribution": dict(Counter([s['primary_label'] for s in final_samples])),
            "augmentation_info": {
                "method": "violation_to_normal_turn4_replacement",
                "violation_sources": dict(Counter([
                    s['augmentation_meta']['original_label'] 
                    for s in final_samples 
                    if 'augmentation_meta' in s
                ])),
                "augmented_session_id_range": f"augmented_1301 ~ augmented_{augmented_id_start + len(augmented_samples) - 1}"
            }
        },
        "samples": final_samples
    }
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("Augmentation Complete!")
    print(f"{'='*70}")
    print(f"\nFinal Distribution:")
    for label, count in sorted(final_data['metadata']['distribution'].items()):
        pct = count / len(final_samples) * 100
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nOriginal: {len(samples)}")
    print(f"Augmented: {len(augmented_samples)}")
    print(f"Total: {len(final_samples)}")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
