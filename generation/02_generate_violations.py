"""
Step 2: Generate Violation Samples (V1-V5)
==========================================
GPT-4o-mini를 사용하여 각 violation 유형별 샘플을 생성합니다.

- V1 (200): Lack of empathy
- V2 (200): Excessive self-disclosure
- V3 (200): Judgment or criticism
- V4 (100): Offering premature advice
- V5 (75): Abrupt topic change

Output: generation/outputs/generated/generated_V*.json
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm import prompts

def generate_label(label, count, client):
    """Generate samples for a specific violation label"""
    input_file = Path(f"generation/outputs/assigned/ESConv_{label.lower()}_assigned.json")
    output_file = Path(f"generation/outputs/generated/generated_{label}.json")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        prefixes = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Generating {label}: {len(prefixes)} samples")
    print(f"{'='*70}")
    
    # Select appropriate prompt
    prompt_mapping = {
        "V1": (prompts.V1_SYSTEM, prompts.build_v1_prompt),
        "V2": (prompts.V2_SYSTEM, prompts.build_v2_prompt),
        "V3": (prompts.V3_SYSTEM, prompts.build_v3_prompt),
        "V4": (prompts.V4_SYSTEM, prompts.build_v4_prompt),
        "V5": (prompts.V5_SYSTEM, prompts.build_v5_prompt),
    }
    
    system_prompt, prompt_builder = prompt_mapping[label]
    
    results = []
    for i, prefix_data in enumerate(prefixes):
        print(f"[{i+1}/{count}] Session {prefix_data['esconv_session_id']}")
        
        # Build prompt based on label
        if label == "V5":
            user_prompt = prompt_builder(
                situation=prefix_data['situation'],
                prefix_conversation=prefix_data['conversation']
            )
        else:
            user_prompt = prompt_builder(
                situation=prefix_data['situation'],
                prefix_dialog=prefix_data['dialog'],
                prefix_length=len(prefix_data['dialog']) - 1
            )
        
        try:
            result = client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                retry_message=prompts.RETRY_MESSAGE
            )
            
            # Build sample
            sample = {
                "esconv_session_id": prefix_data["esconv_session_id"],
                "situation": prefix_data["situation"],
                "primary_label": label,
                "generation_method": "esconv_prefix_with_insertion",
            }
            
            if label == "V5":
                sample["prefix_conversation"] = prefix_data["conversation"]
                sample["generated_turn"] = result.get("generated_turn", "")
            else:
                sample["prefix_dialog"] = prefix_data["dialog"]
                sample["generated_dialog"] = result["dialog"]
                sample["violation_turn_index"] = 3
            
            if "violation_reason" in result:
                sample["violation_reason"] = result["violation_reason"]
            
            results.append(sample)
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
        
        if (i + 1) % 20 == 0:
            print(f"PROGRESS: {i+1}/{count} - Success: {len(results)}")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(results)}/{count} samples")
    print(f"Saved: {output_file}")
    
    return len(results)

def main():
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=800,
        timeout=60,
        max_retries=1
    )
    
    labels = [
        ("V1", 200),
        ("V2", 200),
        ("V3", 200),
        ("V4", 100),
        ("V5", 75),
    ]
    
    total_generated = 0
    for label, count in labels:
        generated = generate_label(label, count, client)
        total_generated += generated
    
    print(f"\n{'='*70}")
    print(f"✓ All violations generated: {total_generated} samples")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
