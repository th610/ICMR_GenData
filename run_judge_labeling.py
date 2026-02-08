"""
ESConv 원본 프리픽스 1300개에 대한 Judge 라벨링
라벨 없는 원본 데이터를 Judge로 라벨링
"""
import json
import os
from src.llm.openai_client import OpenAIClient
from tqdm import tqdm

# Judge 프롬프트
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for emotional support conversations.
Given a conversation prefix (context), predict what violation type the NEXT supporter turn would likely have.

Violation Types:
- Normal: No violation, appropriate response
- V1: Missing context understanding, insufficient information gathering
- V2: Agency violation, directive/coercive advice
- V3: Low-quality empathy, superficial/cliche responses
- V4: Reality distortion, dismissing seeker's feelings
- V5: Crisis safety failure, ignoring safety protocols in crisis situations

Respond with ONLY a JSON object:
{
  "predicted_label": "Normal" or "V1" or "V2" or "V3" or "V4" or "V5",
  "confidence": "high" or "medium" or "low",
  "reason": "Brief explanation"
}"""

JUDGE_USER_TEMPLATE = """Situation: {situation}

Conversation:
{dialog}

What violation type would the NEXT supporter turn likely have?"""

def format_dialog(dialog):
    """대화 포맷팅"""
    lines = []
    for turn in dialog:
        speaker = turn['speaker'].upper()
        content = turn['content'].strip()
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)

def judge_single_prefix(client, prefix):
    """단일 프리픽스 라벨링"""
    dialog_text = format_dialog(prefix['dialog'])
    user_prompt = JUDGE_USER_TEMPLATE.format(
        situation=prefix['situation'],
        dialog=dialog_text
    )
    
    try:
        result = client.call(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            retry_message="Please respond with valid JSON only."
        )
        return {
            'esconv_session_id': prefix['esconv_session_id'],
            'situation': prefix['situation'],
            'predicted_label': result.get('predicted_label', 'Normal'),
            'confidence': result.get('confidence', 'medium'),
            'reason': result.get('reason', ''),
            'success': True
        }
    except Exception as e:
        print(f"  ❌ Failed on session {prefix['esconv_session_id']}: {e}")
        return {
            'esconv_session_id': prefix['esconv_session_id'],
            'situation': prefix['situation'],
            'predicted_label': None,
            'confidence': None,
            'reason': str(e),
            'success': False
        }

def main():
    print("="*80)
    print("Judge Labeling for ESConv 1300 Prefixes")
    print("="*80)
    
    # Load prefixes
    print("\n1. Loading ESConv prefixes...")
    with open('generation/outputs/assigned/ESConv_1300_prefixes.json', 'r', encoding='utf-8') as f:
        prefixes = json.load(f)
    print(f"   Total prefixes: {len(prefixes)}")
    
    # Initialize OpenAI client
    print("\n2. Initializing Judge (GPT-4o-mini)...")
    client = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300,
        timeout=30,
        max_retries=1
    )
    print("   Client ready")
    
    # Judge labeling
    print("\n3. Running Judge labeling...")
    print(f"   This will take ~{len(prefixes) * 2 / 60:.1f} minutes")
    
    results = []
    failed_count = 0
    
    for i, prefix in enumerate(tqdm(prefixes, desc="Judging")):
        result = judge_single_prefix(client, prefix)
        results.append(result)
        
        if not result['success']:
            failed_count += 1
        
        # Progress report every 100
        if (i + 1) % 100 == 0:
            success_rate = ((i + 1 - failed_count) / (i + 1)) * 100
            print(f"\n   Progress: {i+1}/{len(prefixes)} ({success_rate:.1f}% success)")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    successful = len([r for r in results if r['success']])
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {successful/len(results)*100:.2f}%")
    
    # Label distribution
    print("\nLabel distribution:")
    label_counts = {}
    for r in results:
        if r['success']:
            label = r['predicted_label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"  {label}: {count} ({count/successful*100:.1f}%)")
    
    # Save results
    output_path = 'generation/outputs/evaluated/judge_labeling_1300.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        'metadata': {
            'total': len(results),
            'successful': successful,
            'failed': failed_count,
            'label_distribution': label_counts
        },
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
