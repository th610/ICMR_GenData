"""
Systematic debugging of the model

Test 1: Tokenizer
Test 2: Model loading
Test 3: Model file integrity
Test 4: Full pipeline
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

print("="*60)
print("MODEL DEBUG TEST")
print("="*60)

model_path = "models/violation_classifier/best_model"

# ============================================================
# TEST 1: Tokenizer
# ============================================================
print("\n[TEST 1] Tokenizer Loading...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer loaded successfully")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    
    # Test tokenization
    test_text = "seeker: I'm feeling sad\nsupporter: I understand how you feel"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"  Test tokenization: {tokens['input_ids'].shape}")
    print(f"  Sample tokens: {tokens['input_ids'][0][:10].tolist()}")
except Exception as e:
    print(f"✗ Tokenizer error: {e}")
    exit(1)

# ============================================================
# TEST 2: Model Loading
# ============================================================
print("\n[TEST 2] Model Loading...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Num labels: {model.config.num_labels}")
    print(f"  Hidden size: {model.config.hidden_size}")
except Exception as e:
    print(f"✗ Model loading error: {e}")
    exit(1)

# ============================================================
# TEST 3: Model Parameters Check
# ============================================================
print("\n[TEST 3] Model Parameters...")
try:
    # Check if parameters have variation
    param_values = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            param_values.extend(param.flatten()[:100].tolist())
            if len(param_values) >= 1000:
                break
    
    param_std = np.std(param_values)
    param_mean = np.mean(param_values)
    
    print(f"  Parameter mean: {param_mean:.6f}")
    print(f"  Parameter std: {param_std:.6f}")
    
    if param_std < 0.001:
        print("  ✗ WARNING: Parameters have very low variation!")
    else:
        print("  ✓ Parameters look normal")
        
except Exception as e:
    print(f"✗ Parameter check error: {e}")

# ============================================================
# TEST 4: Forward Pass with Different Inputs
# ============================================================
print("\n[TEST 4] Forward Pass Test...")
model.eval()

test_cases = [
    "You should just leave him.",
    "I understand how you're feeling. That must be really difficult.",
    "Have you considered talking to a professional?",
]

print("\nTesting 3 different inputs:")
for i, text in enumerate(test_cases, 1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    print(f"\nInput {i}: {text[:50]}...")
    print(f"  Logits: {logits[0].tolist()}")
    print(f"  Probs: {probs[0].tolist()}")
    print(f"  Max prob: {probs[0].max():.4f}")
    print(f"  Min prob: {probs[0].min():.4f}")

# ============================================================
# TEST 5: Test with EXACT training data format
# ============================================================
print("\n[TEST 5] Testing with training data format...")

# Load one sample from training data
with open('data/final/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Get samples with different labels
samples_by_label = {}
for session in test_data:
    label = session['label']
    if label not in samples_by_label:
        samples_by_label[label] = session

print(f"\nFound labels: {list(samples_by_label.keys())}")

for label, session in list(samples_by_label.items())[:3]:
    # Format exactly as in training
    situation = session.get('situation', '')
    dialog = session.get('dialog', [])
    
    dialog_lines = []
    for turn in dialog:
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', turn.get('text', ''))
        dialog_lines.append(f"{speaker}: {content}")
    
    dialog_text = '\n'.join(dialog_lines)
    text = f"[SITUATION]\n{situation}\n\n[DIALOG]\n{dialog_text}"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits).item()
    
    id2label = {0: "normal", 1: "v1", 2: "v2", 3: "v3", 4: "v4", 5: "v5"}
    
    print(f"\nTrue label: {label.upper()}")
    print(f"  Input length: {len(text)} chars")
    print(f"  Token count: {inputs['input_ids'].shape[1]}")
    print(f"  Logits: {logits[0].tolist()}")
    print(f"  Predicted: {id2label[pred_idx].upper()}")
    print(f"  Confidence: {probs[0][pred_idx]:.4f}")
    print(f"  All probs: {[f'{p:.4f}' for p in probs[0].tolist()]}")

print("\n" + "="*60)
print("DEBUG TEST COMPLETE")
print("="*60)
