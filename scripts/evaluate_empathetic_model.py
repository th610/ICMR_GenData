"""
Evaluate EmpatheticDialogues with our trained classifier model

Much faster than Judge API!
"""
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path="models/violation_classifier/best_model"):
    """Load trained classifier"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return tokenizer, model, device


def predict_batch(texts, tokenizer, model, device, batch_size=8):  # Reduced from 32
    """Predict violations in batch"""
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        # Convert to labels
        id2label = {0: "normal", 1: "v1", 2: "v2", 3: "v3", 4: "v4", 5: "v5"}
        
        for pred, prob in zip(preds, probs):
            label = id2label[pred.item()]
            confidence = prob[pred].item()
            predictions.append({
                "label": label,
                "confidence": confidence,
                "probs": {id2label[i]: p.item() for i, p in enumerate(prob)}
            })
    
    return predictions


def evaluate_empathetic_with_model(max_samples=None,
                                   output_path="data/external/empathetic_model_judged.json"):
    """Evaluate with classifier model"""
    
    # Load data
    print("Loading EmpatheticDialogues data...")
    df = pd.read_parquet('data/external/empathetic_train.parquet')
    
    # Group by conversation
    print("Grouping conversations...")
    conversations = {}
    for _, row in df.iterrows():
        conv_id = row['conv_id']
        if conv_id not in conversations:
            conversations[conv_id] = {
                'conv_id': conv_id,
                'context': row['context'],
                'utterances': []
            }
        conversations[conv_id]['utterances'].append({
            'utterance_idx': row['utterance_idx'],
            'speaker_idx': row['speaker_idx'],
            'utterance': row['utterance']
        })
    
    # Sample conversations
    conv_list = list(conversations.items())
    if max_samples:
        conv_list = conv_list[:max_samples]
    
    samples = []
    for conv_id, conv_data in conv_list:
        # Sort utterances
        utterances = sorted(conv_data['utterances'], key=lambda x: x['utterance_idx'])
        
        if len(utterances) < 3:
            continue
        
        # Build full dialog (all turns, not just last 6)
        dialog_parts = []
        for utt in utterances:
            speaker = "seeker" if utt['speaker_idx'] == 0 else "supporter"
            dialog_parts.append(f"{speaker}: {utt['utterance']}")
        
        dialog_text = "\n".join(dialog_parts)
        
        # Format like training data
        situation = f"An emotional support conversation about {conv_data['context']}."
        full_text = f"[SITUATION]\n{situation}\n\n[DIALOG]\n{dialog_text}"
        
        samples.append({
            "conv_id": conv_id,
            "context": conv_data['context'],
            "num_turns": len(utterances),
            "last_response": utterances[-1]['utterance'],
            "dialog_text": full_text
        })
    
    print(f"Evaluating {len(samples)} conversations with model...")
    
    # Load model
    tokenizer, model, device = load_model()
    
    # Prepare texts
    texts = [s['dialog_text'] for s in samples]
    
    # Predict in batches
    print("Running predictions...")
    predictions = predict_batch(texts, tokenizer, model, device, batch_size=8)  # Reduced batch size
    
    # Combine results
    violation_counts = {
        "normal": 0,
        "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0
    }
    
    results = []
    for sample, pred in zip(samples, predictions):
        label = pred['label']
        violation_counts[label] += 1
        
        results.append({
            "conv_id": sample['conv_id'],
            "context": sample['context'],
            "num_turns": sample['num_turns'],
            "last_response": sample['last_response'],
            "model_label": label,
            "model_confidence": pred['confidence'],
            "probs": pred['probs']
        })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "empathetic_dialogues",
            "method": "trained_classifier_model",
            "total_samples": len(results),
            "violation_counts": violation_counts,
            "violation_rate": (len(results) - violation_counts['normal']) / len(results) * 100 if results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("EmpatheticDialogues Model Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"\nViolation Distribution:")
    for label, count in violation_counts.items():
        pct = count / len(results) * 100 if results else 0
        print(f"  {label.upper()}: {count} ({pct:.1f}%)")
    
    violation_rate = (len(results) - violation_counts['normal']) / len(results) * 100 if results else 0
    print(f"\nTotal Violations: {violation_rate:.1f}%")
    print(f"Saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Max conversations to evaluate (default: all)")
    args = parser.parse_args()
    
    evaluate_empathetic_with_model(args.max_samples)
