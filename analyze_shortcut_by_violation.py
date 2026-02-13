"""
Analyze shortcut patterns by violation type from 1000-sample model.
"""
import json
import torch
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

# Import custom modules
import sys
sys.path.append('models')
from data_utils import DataConfig, setup_tokenizer
from model import ViolationClassifier

def format_input_text(sample, mode='full'):
    """Format input text based on mode"""
    # Build prefix
    prefix_parts = []
    for turn in sample['prefix_dialog']:
        speaker = turn['speaker']
        content = turn['content'].strip()
        
        if speaker == 'seeker':
            prefix_parts.append(f"{DataConfig.SEEKER_TOKEN} {content}")
        else:
            prefix_parts.append(f"{DataConfig.SUPPORTER_TOKEN} {content}")
    
    prefix_text = " ".join(prefix_parts)
    
    # Build generated dialog (Turn 1-3)
    gen_parts = []
    for i, turn in enumerate(sample['generated_dialog'][:3]):  # Only first 3
        speaker = turn['speaker']
        content = turn['content'].strip()
        
        if speaker == 'seeker':
            gen_parts.append(f"{DataConfig.SEEKER_TOKEN} {content}")
        else:
            gen_parts.append(f"{DataConfig.SUPPORTER_TOKEN} {content}")
    
    context_text = " ".join(gen_parts)
    
    # TARGET (4th turn, always supporter)
    target_turn = sample['generated_dialog'][3]
    target_text = f"{DataConfig.SUPPORTER_TARGET_TOKEN} {target_turn['content'].strip()}"
    
    # Combine based on mode
    if mode == 'full':
        return f"{prefix_text} {context_text} {target_text}"
    elif mode == 'response_only':
        return target_text
    elif mode == 'context_only':
        return f"{prefix_text} {context_text}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

def evaluate_by_violation(model_path):
    """Evaluate model by violation type"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    num_added = setup_tokenizer(tokenizer)
    print(f"  Added {num_added} special tokens")
    
    # Load model
    print("Loading model...")
    model = ViolationClassifier(
        model_name="roberta-base",
        num_labels=len(DataConfig.LABELS),
        pooling="cls"
    )
    
    # Resize embeddings to match tokenizer
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    with open('experiments/test_gold_300_prefix.json', 'r') as f:
        data = json.load(f)
    samples = data['samples']
    
    # Group by label
    by_label = {}
    for sample in samples:
        label = sample['primary_label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)
    
    print(f"  Total: {len(samples)} samples")
    print(f"  Distribution: {', '.join([f'{k}:{len(v)}' for k, v in sorted(by_label.items())])}")
    
    # Evaluate each mode
    modes = ['full', 'response_only', 'context_only']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*70}")
        
        mode_results = {}
        
        for label in sorted(by_label.keys()):
            label_samples = by_label[label]
            
            true_labels = []
            pred_labels = []
            
            for sample in label_samples:
                # Format input
                input_text = format_input_text(sample, mode=mode)
                
                # Tokenize
                encoding = tokenizer(
                    input_text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Predict
                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    logits = model(input_ids, attention_mask)
                    pred = torch.argmax(logits, dim=1).item()
                
                true_labels.append(DataConfig.LABEL2ID[sample['primary_label']])
                pred_labels.append(pred)
            
            # Calculate metrics
            correct = sum([t == p for t, p in zip(true_labels, pred_labels)])
            accuracy = correct / len(true_labels)
            
            # Count predictions
            pred_counts = {}
            for p in pred_labels:
                pred_label = DataConfig.ID2LABEL[p]
                pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
            
            mode_results[label] = {
                'total': len(label_samples),
                'correct': correct,
                'accuracy': accuracy,
                'predictions': pred_counts
            }
            
            print(f"\n{label} ({len(label_samples)} samples):")
            print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(label_samples)})")
            print(f"  Predictions: {pred_counts}")
        
        results[mode] = mode_results
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Accuracy by Violation Type")
    print(f"{'='*70}")
    print(f"{'Label':<10} {'Full':>10} {'Response':>10} {'Context':>10}")
    print("-" * 70)
    for label in sorted(by_label.keys()):
        full_acc = results['full'][label]['accuracy']
        resp_acc = results['response_only'][label]['accuracy']
        ctx_acc = results['context_only'][label]['accuracy']
        print(f"{label:<10} {full_acc:>9.1%} {resp_acc:>9.1%} {ctx_acc:>9.1%}")
    
    # Save results
    output_file = 'shortcut_by_violation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/outputs/best_model_edited.pt'
    print(f"Model: {model_path}\n")
    evaluate_by_violation(model_path)
