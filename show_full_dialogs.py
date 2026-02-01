import json
import pandas as pd

# Load API and Model results
api_results = json.load(open('data/external/empathetic_judged.json', encoding='utf-8'))
model_results = json.load(open('data/external/empathetic_model_judged.json', encoding='utf-8'))

# Load original data
df = pd.read_parquet('data/external/empathetic_train.parquet')

print("="*80)
print("VERIFICATION: Same Dataset?")
print("="*80)
print(f"API results: {len(api_results['results'])} conversations")
print(f"Model results: {len(model_results['results'])} conversations")
print(f"Same conv_ids? {api_results['results'][0]['conv_id'] == model_results['results'][0]['conv_id']}")
print(f"First 5 conv_ids match: {[api_results['results'][i]['conv_id'] == model_results['results'][i]['conv_id'] for i in range(5)]}")

print("\n" + "="*80)
print("FULL DIALOG EXAMPLES")
print("="*80)

# Get some disagreement cases
disagreements = []
for api_result in api_results['results']:
    conv_id = api_result['conv_id']
    model_result = next((r for r in model_results['results'] if r['conv_id'] == conv_id), None)
    
    if model_result and api_result['judge_label'] != model_result['model_label']:
        disagreements.append({
            'conv_id': conv_id,
            'context': api_result['context'],
            'api_label': api_result['judge_label'],
            'api_reason': api_result['judge_reason'],
            'model_label': model_result['model_label'],
            'model_confidence': model_result['model_confidence']
        })

# Show 3 full dialogs
selected = [
    disagreements[0],  # Model=V1, API=Normal
    disagreements[3],  # Model=V2, API=Normal
    next((d for d in disagreements if d['api_label'] != 'normal'), None)  # API detected one
]

for idx, case in enumerate(selected[:3], 1):
    if not case:
        continue
        
    conv_id = case['conv_id']
    
    # Get full dialog from parquet
    conv_rows = df[df['conv_id'] == conv_id].sort_values('utterance_idx')
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {idx}: {conv_id}")
    print(f"Context: {case['context']}")
    print(f"API Label: {case['api_label'].upper()}")
    print(f"Model Label: {case['model_label'].upper()} (confidence: {case['model_confidence']:.3f})")
    print(f"{'='*80}")
    
    print(f"\n[FULL DIALOG]")
    for _, row in conv_rows.iterrows():
        speaker = "Seeker" if row['speaker_idx'] == 0 else "Supporter"
        print(f"{speaker}: {row['utterance']}")
    
    print(f"\n[SITUATION (used for model)]")
    print(f"An emotional support conversation about {case['context']}.")
    
    print(f"\n[API REASONING]")
    print(f"{case['api_reason']}")
    
    print(f"\n[LAST RESPONSE ONLY]")
    last_utterance = conv_rows.iloc[-1]['utterance']
    print(f"{last_utterance}")
    
    print("\n" + "-"*80)
