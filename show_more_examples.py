import json
import pandas as pd

# Load results
api_results = json.load(open('data/external/empathetic_judged.json', encoding='utf-8'))
model_results = json.load(open('data/external/empathetic_model_judged.json', encoding='utf-8'))
df = pd.read_parquet('data/external/empathetic_train.parquet')

print("="*80)
print("MORE DISAGREEMENT EXAMPLES")
print("="*80)

# Get disagreement cases
cases = []
for api_result in api_results['results']:
    conv_id = api_result['conv_id']
    model_result = next((r for r in model_results['results'] if r['conv_id'] == conv_id), None)
    
    if model_result and api_result['judge_label'] != model_result['model_label']:
        cases.append({
            'conv_id': conv_id,
            'context': api_result['context'],
            'api_label': api_result['judge_label'],
            'api_reason': api_result['judge_reason'],
            'model_label': model_result['model_label'],
            'model_confidence': model_result['model_confidence'],
            'model_probs': model_result['probs']
        })

# Show cases where API said V2 but model said Normal
api_v2_cases = [c for c in cases if c['api_label'] == 'v2']

print(f"\nAPI detected V2 violations: {len(api_v2_cases)} cases")
print("Let's see those:")

for idx, case in enumerate(api_v2_cases[:3], 1):
    conv_id = case['conv_id']
    conv_rows = df[df['conv_id'] == conv_id].sort_values('utterance_idx')
    
    print(f"\n{'='*80}")
    print(f"API V2 CASE {idx}: {conv_id} (Context: {case['context']})")
    print(f"Model: {case['model_label'].upper()} (conf: {case['model_confidence']:.3f})")
    print(f"{'='*80}")
    
    print(f"\n[FULL DIALOG]")
    for _, row in conv_rows.iterrows():
        speaker = "Seeker" if row['speaker_idx'] == 0 else "Supporter"
        print(f"{speaker}: {row['utterance']}")
    
    print(f"\n[API REASONING - V2]")
    print(f"{case['api_reason']}")
    
    print(f"\n[MODEL PROBS]")
    print(f"Normal: {case['model_probs']['normal']:.3f}, V1: {case['model_probs']['v1']:.3f}, V2: {case['model_probs']['v2']:.3f}, V3: {case['model_probs']['v3']:.3f}")
    print("-"*80)

# Show cases where Model said V1 with high confidence
model_v1_high = [c for c in cases if c['model_label'] == 'v1' and c['model_confidence'] > 0.5 and c['api_label'] == 'normal']

print(f"\n\n{'='*80}")
print(f"MODEL HIGH CONFIDENCE V1 (but API said Normal): {len(model_v1_high)} cases")
print("="*80)

for idx, case in enumerate(model_v1_high[:3], 1):
    conv_id = case['conv_id']
    conv_rows = df[df['conv_id'] == conv_id].sort_values('utterance_idx')
    
    print(f"\n{'='*80}")
    print(f"MODEL V1 CASE {idx}: {conv_id} (Context: {case['context']})")
    print(f"Model: V1 (conf: {case['model_confidence']:.3f}), API: NORMAL")
    print(f"{'='*80}")
    
    print(f"\n[FULL DIALOG]")
    for _, row in conv_rows.iterrows():
        speaker = "Seeker" if row['speaker_idx'] == 0 else "Supporter"
        print(f"{speaker}: {row['utterance']}")
    
    print(f"\n[WHY API SAID NORMAL]")
    print(f"{case['api_reason']}")
    
    print(f"\n[MODEL PROBS]")
    print(f"Normal: {case['model_probs']['normal']:.3f}, V1: {case['model_probs']['v1']:.3f}")
    print("-"*80)
