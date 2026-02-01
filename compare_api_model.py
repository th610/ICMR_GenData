import json

# Load both results
api_results = json.load(open('data/external/empathetic_judged.json'))
model_results = json.load(open('data/external/empathetic_model_judged.json'))

print("="*60)
print("API vs Model Comparison")
print("="*60)

print("\nAPI Results (100 samples):")
print(f"  Normal: {api_results['violation_counts']['normal']}")
print(f"  V1: {api_results['violation_counts']['v1']}")
print(f"  V2: {api_results['violation_counts']['v2']}")
print(f"  Total violations: {api_results['violation_rate']:.1f}%")

print("\nModel Results (995 samples):")
print(f"  Normal: {model_results['violation_counts']['normal']}")
print(f"  V1: {model_results['violation_counts']['v1']}")
print(f"  V2: {model_results['violation_counts']['v2']}")
print(f"  Total violations: {model_results['violation_rate']:.1f}%")

# Check same conversations from API results
print("\n" + "="*60)
print("Checking same conversations that API found violations:")
print("="*60)

api_violations = [r for r in api_results['results'] if r['judge_label'] != 'normal']
print(f"\nAPI found {len(api_violations)} violations:")

for v in api_violations:
    # Find same conversation in model results
    model_result = next((r for r in model_results['results'] if r['conv_id'] == v['conv_id']), None)
    
    if model_result:
        print(f"\n[{v['conv_id']}] Context: {v['context']}")
        print(f"Response: {v['last_response'][:100]}...")
        print(f"  API Label: {v['judge_label'].upper()}")
        print(f"  Model Label: {model_result['model_label'].upper()}")
        print(f"  Model Confidence: {model_result['model_confidence']:.3f}")
        print(f"  Model Probs: v1={model_result['probs']['v1']:.3f}, v2={model_result['probs']['v2']:.3f}")
