"""
Show actual examples from both datasets side by side
"""
import json
import random

print("="*80)
print("DATASET COMPARISON - ACTUAL EXAMPLES")
print("="*80)

# OUR DATA - ESConv based
print("\n" + "="*80)
print("OUR DATA (ESConv-based) - Short dialogue responses")
print("="*80)

datasets = [
    ('Normal', 'data/generated/normal_400.json'),
    ('V1 (Empathy Violation)', 'data/generated/v1_240.json'),
    ('V2 (Fact-Checking)', 'data/generated/v2_160.json'),
    ('V3 (Advice Violation)', 'data/generated/v3_200.json'),
]

for label, filepath in datasets:
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        # Pick a random sample
        sample = random.choice(data)
        
        print(f"\n{'─'*80}")
        print(f"[{label}]")
        print(f"{'─'*80}")
        
        # Get last few turns
        dialog = sample.get('dialog', [])
        
        print("\n[CONTEXT - Last 3 turns]")
        recent_turns = dialog[-4:-1] if len(dialog) > 3 else dialog[:-1]
        for turn in recent_turns:
            speaker = turn.get('speaker', 'unknown')
            content = turn.get('content', turn.get('text', ''))
            print(f"{speaker}: {content}")
        
        print(f"\n[LAST RESPONSE - {len(dialog[-1].get('content', dialog[-1].get('text', '')))} chars]")
        last_turn = dialog[-1]
        response = last_turn.get('content', last_turn.get('text', ''))
        print(f"supporter: {response}")
        
        # Judge result if available
        if sample.get('judge_label'):
            print(f"\n[Judge Result: {sample['judge_label']}]")
            if sample.get('judge_reason'):
                reason = sample['judge_reason'][:150]
                print(f"Reason: {reason}...")
        
    except Exception as e:
        print(f"{label}: Error - {e}")

# COUNSEL-CHAT
print("\n\n" + "="*80)
print("COUNSEL-CHAT - Professional counselor responses (LONG)")
print("="*80)

try:
    with open('data/external/counsel_chat.json', encoding='utf-8') as f:
        counsel = json.load(f)
    
    # Show 3 random examples
    samples = random.sample(counsel[:50], 3)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'─'*80}")
        print(f"[Example {i}]")
        print(f"{'─'*80}")
        
        print(f"\n[QUESTION - {len(sample['question'])} chars]")
        print(sample['question'])
        if sample.get('question_text'):
            print(sample['question_text'])
        
        answer_len = len(sample['answer'])
        print(f"\n[ANSWER - {answer_len} chars]")
        
        # Show full answer if short, or truncated if long
        if answer_len <= 300:
            print(sample['answer'])
        else:
            # Show first 150 and last 150 chars
            print(sample['answer'][:200])
            print(f"\n... ({answer_len - 400} chars omitted) ...\n")
            print(sample['answer'][-200:])
        
        print(f"\n[Topics: {', '.join(sample.get('topics', ['none']))}]")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("KEY DIFFERENCES")
print("="*80)
print("""
OUR DATA (ESConv):
  ✓ Short, conversational responses (50-150 chars avg)
  ✓ Multi-turn context visible
  ✓ Real-time dialogue feel
  ✗ Sometimes too brief to show full counseling approach
  
COUNSEL-CHAT:
  ✓ Comprehensive, thoughtful responses (600+ chars avg)
  ✓ Professional counseling style
  ✓ Multiple perspectives and explanations
  ✗ No dialogue context (single Q&A pair)
  ✗ More essay-like than conversational
  
JUDGE BEHAVIOR:
  - Short responses → Violations clear and obvious
  - Long responses → Violations diluted by good parts
  - Length bias likely affecting detection!
""")
