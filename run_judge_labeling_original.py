"""
ESConv 원본 프리픽스 1300개에 대한 Judge 라벨링 (원래 프롬프트 사용)
src/llm/prompts_judge.py의 JUDGE_SYSTEM과 build_judge_prompt 사용
"""
import json
import os
from dotenv import load_dotenv
from src.llm.openai_client import OpenAIClient
from src.llm.prompts_judge import JUDGE_SYSTEM, build_judge_prompt
from tqdm import tqdm

# Load .env file
load_dotenv(override=True)

def judge_single_prefix(client, prefix):
    """단일 프리픽스 라벨링"""
    dialog = prefix['dialog']
    
    # 마지막 턴이 supporter인지 확인
    if not dialog or dialog[-1]['speaker'] != 'supporter':
        # 마지막이 supporter가 아니면 스킵
        return {
            'esconv_session_id': prefix['esconv_session_id'],
            'situation': prefix['situation'],
            'predicted_label': None,
            'confidence': None,
            'reason': 'Last turn is not supporter',
            'success': False
        }
    
    try:
        # prompts_judge.py의 프롬프트 사용
        user_prompt = build_judge_prompt(dialog)
        
        result = client.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            retry_message="Please respond with valid JSON only."
        )
        
        return {
            'esconv_session_id': prefix['esconv_session_id'],
            'situation': prefix['situation'],
            'predicted_label': result.get('label', 'Normal'),
            'confidence': result.get('confidence', 'medium'),
            'reason': result.get('reason', ''),
            'evidence': result.get('evidence', ''),
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
            'evidence': None,
            'success': False
        }

def main():
    output_path = 'generation/outputs/evaluated/judge_labeling_1300_original.json'
    
    print("="*80)
    print("Judge Labeling for ESConv 1300 Prefixes (with Auto-save)")
    print("Using Original prompts_judge.py")
    print("="*80)
    
    # Load prefixes
    print("\n1. Loading ESConv prefixes...")
    with open('generation/outputs/assigned/ESConv_1300_prefixes.json', 'r', encoding='utf-8') as f:
        prefixes = json.load(f)
    print(f"   Total prefixes: {len(prefixes)}")
    
    # Load existing results (if any)
    results = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                results = existing.get('results', [])
                processed_ids = {r['esconv_session_id'] for r in results}
            print(f"\n✅ Loaded {len(results)} existing results")
            print(f"   Resuming from session {len(results)}")
        except:
            print("\n⚠️  Could not load existing results, starting fresh")
    
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
    print("   Using prompts_judge.py (V7 version)")
    
    # Judge labeling
    remaining = len(prefixes) - len(results)
    print(f"\n3. Running Judge labeling...")
    print(f"   Remaining: {remaining}/{len(prefixes)}")
    print(f"   Auto-save every 100 items")
    
    failed_count = sum(1 for r in results if not r['success'])
    
    for i, prefix in enumerate(tqdm(prefixes, desc="Judging", initial=len(results), total=len(prefixes))):
        # Skip if already processed
        if prefix['esconv_session_id'] in processed_ids:
            continue
        
        result = judge_single_prefix(client, prefix)
        results.append(result)
        processed_ids.add(prefix['esconv_session_id'])
        
        if not result['success']:
            failed_count += 1
        
        # Auto-save every 100
        if len(results) % 100 == 0:
            _save_results(results, failed_count, output_path)
            success_rate = (len(results) - failed_count) / len(results) * 100
            print(f"\n   💾 Auto-saved: {len(results)}/{len(prefixes)} ({success_rate:.1f}% success)")
    
    # Final summary
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
    
    # Final save
    _save_results(results, failed_count, output_path)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)

def _save_results(results, failed_count, output_path):
    """중간 저장 함수"""
    successful = len([r for r in results if r['success']])
    label_counts = {}
    for r in results:
        if r['success']:
            label = r['predicted_label']
            label_counts[label] = label_counts.get(label, 0) + 1
    
    output = {
        'metadata': {
            'total': len(results),
            'successful': successful,
            'failed': failed_count,
            'prompt_version': 'prompts_judge.py (V7)',
            'label_distribution': label_counts
        },
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
