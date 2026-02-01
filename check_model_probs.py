import json

d = json.load(open('data/external/empathetic_model_judged.json', encoding='utf-8'))
samples = d['results'][:10]

print("Sample probability distributions:")
print("="*60)
for i, s in enumerate(samples):
    print(f"\nConv {i+1} ({s['conv_id']}):")
    print(f"  Confidence: {s['model_confidence']:.4f}")
    print(f"  Label: {s['model_label']}")
    print(f"  Probs: normal={s['probs']['normal']:.4f}, v1={s['probs']['v1']:.4f}, v2={s['probs']['v2']:.4f}, v3={s['probs']['v3']:.4f}")
    print(f"  Last response: {s['last_response'][:80]}...")
