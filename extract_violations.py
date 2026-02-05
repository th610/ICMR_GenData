"""
모델이 위반으로 예측한 케이스만 추출 (원본 대화 포함)
"""
import json

# Load model predictions
with open('generation/outputs/evaluated/model_predictions_1300.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)

# Load original ESConv prefixes
with open('generation/outputs/assigned/ESConv_1300_prefixes.json', 'r', encoding='utf-8') as f:
    esconv_data = json.load(f)

# Create session_id to esconv mapping
esconv_map = {item['esconv_session_id']: item for item in esconv_data}

# Filter violations only
violations = []
for result in model_data['results']:
    if result['predicted_label'] != 'Normal':
        session_id = result['esconv_session_id']
        esconv_item = esconv_map[session_id]
        
        violations.append({
            'esconv_session_id': session_id,
            'situation': result['situation'],
            'dialog': esconv_item['dialog'],  # 원본 대화 포함
            'model_prediction': {
                'label': result['predicted_label'],
                'label_id': result['predicted_id']
            }
        })

# Summary
label_counts = {}
for v in violations:
    label = v['model_prediction']['label']
    label_counts[label] = label_counts.get(label, 0) + 1

output = {
    'metadata': {
        'total_violations': len(violations),
        'source': 'model_predictions_1300.json',
        'violation_distribution': label_counts
    },
    'violations': violations
}

# Save
output_path = 'generation/outputs/evaluated/model_violations_with_dialog.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("="*60)
print("모델 위반 예측 추출 완료")
print("="*60)
print(f"Total violations: {len(violations)}")
print(f"\nViolation distribution:")
for label, count in sorted(label_counts.items()):
    pct = count / len(violations) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")
print(f"\n✅ Saved to: {output_path}")
