"""
Step 3: Generate Normal Samples
================================
525개의 Normal 샘플 생성 (violation 없는 정상 대화)

Output: generation/outputs/generated/generated_Normal.json
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm import prompts

def main():
    input_file = Path("generation/outputs/assigned/ESConv_normal_assigned.json")
    output_file = Path("generation/outputs/generated/generated_Normal.json")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        prefixes = json.load(f)
    
    print(f"{'='*70}")
    print(f"Generating Normal: {len(prefixes)} samples")
    print(f"{'='*70}")
    
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=800,
        timeout=60,
        max_retries=1
    )
    
    results = []
    for i, prefix_data in enumerate(prefixes):
        print(f"[{i+1}/525] Session {prefix_data['esconv_session_id']}")
        
        user_prompt = prompts.build_normal_prompt(
            situation=prefix_data['situation'],
            prefix_dialog=prefix_data['dialog'],
            prefix_length=len(prefix_data['dialog']) - 1
        )
        
        try:
            result = client.call(
                system_prompt=prompts.NORMAL_SYSTEM,
                user_prompt=user_prompt,
                retry_message=prompts.RETRY_MESSAGE
            )
            
            sample = {
                "esconv_session_id": prefix_data["esconv_session_id"],
                "situation": prefix_data["situation"],
                "prefix_dialog": prefix_data["dialog"],
                "generated_dialog": result["dialog"],
                "primary_label": "Normal",
                "generation_method": "esconv_prefix_continuation"
            }
            
            results.append(sample)
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
        
        if (i + 1) % 50 == 0:
            print(f"PROGRESS: {i+1}/525 - Success: {len(results)}")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(results)}/525 samples")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
