"""
Test model on our own training data to verify it works
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def session_to_text(session):
    """Convert session to text input (same as training)"""
    situation = session.get('situation', '')
    dialog = session.get('dialog', [])
    
    # Format dialog
    dialog_lines = []
    for turn in dialog:
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', turn.get('text', ''))
        dialog_lines.append(f"{speaker}: {content}")
    
    dialog_text = '\n'.join(dialog_lines)
    
    # Combine
    text = f"[SITUATION]\n{situation}\n\n[DIALOG]\n{dialog_text}"
    
    return text


# Load model
print("Loading model...")
model_path = "models/violation_classifier/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load test data
print("Loading test data...")
with open('data/final/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Test on first 10 samples
print("\nTesting on first 10 samples from test set:")
print("="*60)

id2label = {0: "normal", 1: "v1", 2: "v2", 3: "v3", 4: "v4", 5: "v5"}

correct = 0
for i, session in enumerate(test_data[:10]):
    text = session_to_text(session)
    true_label = session['label']
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()
    
    pred_label = id2label[pred_idx]
    is_correct = pred_label == true_label
    correct += is_correct
    
    print(f"\nSample {i+1}:")
    print(f"  True: {true_label.upper()}")
    print(f"  Pred: {pred_label.upper()} {'✓' if is_correct else '✗'}")
    print(f"  Confidence: {probs[0][pred_idx]:.4f}")
    print(f"  Probs: normal={probs[0][0]:.4f}, v1={probs[0][1]:.4f}, v2={probs[0][2]:.4f}")

print(f"\n{'='*60}")
print(f"Accuracy on these 10 samples: {correct}/10 = {correct/10*100:.1f}%")
