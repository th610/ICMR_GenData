"""
V5 재생성 스크립트 (올바른 구조로)
=================================
V5를 prefix_dialog + generated_dialog 구조로 생성합니다.

테스트: python generation/02_generate_v5_fixed.py --test
전체 생성: python generation/02_generate_v5_fixed.py
"""
import json
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm import prompts

def generate_v5(client, test_mode=False):
    """Generate V5 samples with correct structure"""
    input_file = Path("generation/outputs/assigned/ESConv_v5_prefixes.json")
    output_file = Path("generation/outputs/generated/generated_V5_new.json")
    
    # Load V5 prefixes
    with open(input_file, 'r', encoding='utf-8') as f:
        v5_prefixes = json.load(f)
    
    if test_mode:
        v5_prefixes = v5_prefixes[:1]  # 1개만 테스트
        print("\n🧪 TEST MODE: Generating 1 sample only\n")
    
    print(f"\n{'='*70}")
    print(f"Generating V5: {len(v5_prefixes)} samples")
    print(f"{'='*70}")
    
    results = []
    for i, prefix_data in enumerate(v5_prefixes):
        print(f"[{i+1}/{len(v5_prefixes)}] Session {prefix_data['esconv_session_id']}")
        
        # Build V5 prompt
        user_prompt = prompts.build_v5_prompt(
            situation=prefix_data['situation'],
            prefix_dialog=prefix_data['dialog'],
            prefix_length=len(prefix_data['dialog'])
        )
        
        try:
            result = client.call(
                system_prompt=prompts.V5_SYSTEM,
                user_prompt=user_prompt,
                retry_message=prompts.RETRY_MESSAGE
            )
            
            if test_mode:
                print(f"\n  API Response keys: {list(result.keys())}")
                print(f"  API Response: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}...")
            
            # Build sample with CORRECT structure
            sample = {
                "esconv_session_id": prefix_data["esconv_session_id"],
                "situation": prefix_data["situation"],
                "prefix_dialog": prefix_data["dialog"],
                "generated_dialog": result.get("dialog", []),  # API 직접 반환
                "primary_label": "V5",
                "generation_method": "esconv_prefix_with_insertion",
                "violation_turn_index": 3,
                "violation_reason": result.get("violation_reason", "")
            }
            
            results.append(sample)
            print(f"  ✓ Success")
            
            if test_mode:
                print("\n" + "="*70)
                print("✅ TEST SAMPLE STRUCTURE:")
                print("="*70)
                print(f"Keys: {list(sample.keys())}")
                print(f"\nprefix_dialog: {len(sample['prefix_dialog'])} turns")
                print(f"generated_dialog: {len(sample['generated_dialog'])} turns")
                print(f"primary_label: {sample['primary_label']}")
                print(f"\nSample data:")
                print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
    
    # Save results
    metadata = {
        "label": "V5",
        "total_samples": len(results),
        "distribution": {"V5": len(results)}
    }
    
    output_data = {
        "metadata": metadata,
        "samples": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(results)}/{len(v5_prefixes)} samples")
    print(f"Saved: {output_file}")
    
    return len(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test mode: generate only 1 sample')
    args = parser.parse_args()
    
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=800,
        timeout=60,
        max_retries=1
    )
    
    generated = generate_v5(client, test_mode=args.test)
    
    if args.test:
        print("\n" + "="*70)
        print("🧪 TEST COMPLETE - Check structure above")
        print("If correct, run without --test flag to generate all 75 samples")
        print("="*70)
    else:
        print(f"\n{'='*70}")
        print(f"✓ V5 generation complete: {generated} samples")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
