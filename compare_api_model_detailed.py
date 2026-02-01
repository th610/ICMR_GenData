import json

# Load both results
api_results = json.load(open('data/external/empathetic_judged.json', encoding='utf-8'))
model_results = json.load(open('data/external/empathetic_model_judged.json', encoding='utf-8'))

print("="*80)
print("API vs Model Comparison on Same 99 Samples")
print("="*80)

print("\nOverall Statistics:")
print(f"API:   Normal={api_results['violation_counts']['normal']}, V1={api_results['violation_counts']['v1']}, V2={api_results['violation_counts']['v2']} → {api_results['violation_rate']:.1f}% violations")
print(f"Model: Normal={model_results['violation_counts']['normal']}, V1={model_results['violation_counts']['v1']}, V2={model_results['violation_counts']['v2']}, V3={model_results['violation_counts']['v3']} → {model_results['violation_rate']:.1f}% violations")

# Find disagreements
disagreements = []
agreements = []

for api_result in api_results['results']:
    conv_id = api_result['conv_id']
    model_result = next((r for r in model_results['results'] if r['conv_id'] == conv_id), None)
    
    if model_result:
        api_label = api_result['judge_label']
        model_label = model_result['model_label']
        
        if api_label != model_label:
            disagreements.append({
                'conv_id': conv_id,
                'context': api_result['context'],
                'response': api_result['last_response'],
                'api_label': api_label,
                'api_reason': api_result['judge_reason'],
                'model_label': model_label,
                'model_confidence': model_result['model_confidence'],
                'model_probs': model_result['probs']
            })
        else:
            agreements.append({
                'conv_id': conv_id,
                'label': api_label,
                'response': api_result['last_response']
            })

print(f"\nAgreements: {len(agreements)}/{len(api_results['results'])} ({len(agreements)/len(api_results['results'])*100:.1f}%)")
print(f"Disagreements: {len(disagreements)}/{len(api_results['results'])} ({len(disagreements)/len(api_results['results'])*100:.1f}%)")

print("\n" + "="*80)
print("DISAGREEMENT EXAMPLES (First 10)")
print("="*80)

for i, d in enumerate(disagreements[:10], 1):
    print(f"\n[Example {i}] Conv: {d['conv_id']}, Context: {d['context']}")
    print(f"Response: {d['response'][:150]}...")
    print(f"  API:   {d['api_label'].upper()}")
    print(f"         Reason: {d['api_reason'][:100]}...")
    print(f"  Model: {d['model_label'].upper()} (confidence: {d['model_confidence']:.3f})")
    print(f"         Probs: normal={d['model_probs']['normal']:.3f}, v1={d['model_probs']['v1']:.3f}, v2={d['model_probs']['v2']:.3f}, v3={d['model_probs']['v3']:.3f}")
    print("-"*80)

print("\n" + "="*80)
print("AGREEMENT EXAMPLES (First 5 Normal)")
print("="*80)

normal_agreements = [a for a in agreements if a['label'] == 'normal'][:5]
for i, a in enumerate(normal_agreements, 1):
    print(f"\n[Example {i}] Conv: {a['conv_id']}")
    print(f"Response: {a['response'][:150]}...")
    print(f"  Both agree: NORMAL ✓")
    print("-"*80)

print("\n" + "="*80)
print("MODEL DETECTED BUT API MISSED (Model=V1/V2/V3, API=Normal)")
print("="*80)

model_caught = [d for d in disagreements if d['api_label'] == 'normal' and d['model_label'] != 'normal'][:5]
for i, d in enumerate(model_caught, 1):
    print(f"\n[Example {i}] Conv: {d['conv_id']}, Context: {d['context']}")
    print(f"Response: {d['response']}")
    print(f"  API:   NORMAL (missed it)")
    print(f"         API Reason: {d['api_reason']}")
    print(f"  Model: {d['model_label'].upper()} (confidence: {d['model_confidence']:.3f})")
    print(f"         Model Probs: v1={d['model_probs']['v1']:.3f}, v2={d['model_probs']['v2']:.3f}, v3={d['model_probs']['v3']:.3f}")
    print("-"*80)

print("\n" + "="*80)
print("API DETECTED BUT MODEL MISSED (API=V1/V2, Model=Normal)")
print("="*80)

api_caught = [d for d in disagreements if d['model_label'] == 'normal' and d['api_label'] != 'normal'][:5]
for i, d in enumerate(api_caught, 1):
    print(f"\n[Example {i}] Conv: {d['conv_id']}, Context: {d['context']}")
    print(f"Response: {d['response']}")
    print(f"  API:   {d['api_label'].upper()}")
    print(f"         API Reason: {d['api_reason']}")
    print(f"  Model: NORMAL (missed it, confidence: {d['model_confidence']:.3f})")
    print(f"         Model Probs: normal={d['model_probs']['normal']:.3f}, v1={d['model_probs']['v1']:.3f}, v2={d['model_probs']['v2']:.3f}")
    print("-"*80)
