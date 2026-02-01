"""Quick V3 Judge Evaluation"""
import json
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt

# Load V3 data
v3_data = json.load(open('data/pilot/v3.json', encoding='utf-8'))

# Initialize client
client = OpenAIClient('gpt-4o-mini', 0.3, 2000)

print("=" * 60)
print("V3 Judge Evaluation (New Prompt)")
print("=" * 60 + "\n")

matches = 0
for i, session in enumerate(v3_data):
    # Build dialog text
    dialog_text = "\n".join([f"[{t['speaker'].upper()}] {t['content']}" for t in session['dialog']])
    
    # Judge
    prompt = build_judge_prompt(dialog_text)
    result = client.call(JUDGE_SYSTEM, prompt, "")
    # Parse if string, otherwise use directly
    if isinstance(result, str):
        result = json.loads(result)
    judge_label = result['label']
    
    is_match = judge_label == 'V3'
    matches += is_match
    
    status = "✅" if is_match else "❌"
    print(f"  [{i}] {status} V3 → {judge_label}")
    if not is_match:
        print(f"      Reason: {result.get('reason', 'N/A')[:80]}...")

print(f"\n{'='*60}")
print(f"V3 Accuracy: {matches}/{len(v3_data)} ({matches*100/len(v3_data):.1f}%)")
print(f"{'='*60}")
