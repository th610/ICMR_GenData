#!/usr/bin/env python3
"""
Step 7: Evaluate Violation Detector

Loads trained model and evaluates on test set.
Outputs detailed metrics including per-label precision/recall/F1.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from typing import Dict, List
import yaml

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix


class ViolationDataset(Dataset):
    """Dataset for violation detection."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Serialize sample
        text = serialize_sample(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels
        labels = sample.get('labels', {})
        label_vector = [
            labels.get('V1', 0),
            labels.get('V2', 0),
            labels.get('V3', 0),
            labels.get('V4', 0),
            labels.get('V5', 0),
        ]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_vector, dtype=torch.float),
            'sample_id': f"{sample['session_id']}_{sample['turn_id']}"
        }


def serialize_sample(sample: Dict) -> str:
    """Convert sample to text format for model input."""
    
    # 1. Situation
    situation = sample.get('meta', {}).get('situation', '')
    
    # 2. Summary
    summary_bullets = sample.get('summary', [])
    summary = '\n'.join(f"- {b}" for b in summary_bullets) if summary_bullets else "(No summary)"
    
    # 3. Context
    context_turns = sample.get('context_turns', [])
    context_lines = []
    for turn in context_turns:
        speaker = turn.get('speaker', 'unknown')
        text = turn.get('text', '')
        context_lines.append(f"{speaker}: {text}")
    context = '\n'.join(context_lines) if context_lines else "(No context)"
    
    # 4. Target response
    response = sample.get('response', '')
    
    # Combine
    return f"""[SITUATION]
{situation}

[SUMMARY]
{summary}

[CONTEXT]
{context}

[RESPONSE]
{response}"""


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute detailed multi-label classification metrics.
    
    Args:
        y_true: Ground truth labels (N, 5)
        y_pred: Predicted probabilities (N, 5)
        threshold: Classification threshold
    
    Returns:
        Dictionary with micro/macro/per-label metrics
    """
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Overall metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true.flatten(), y_pred_binary.flatten(), average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true.flatten(), y_pred_binary.flatten(), average='macro', zero_division=0
    )
    
    # Per-label metrics
    violation_names = ['V1', 'V2', 'V3', 'V4', 'V5']
    per_label = {}
    
    for i, name in enumerate(violation_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], y_pred_binary[:, i], average='binary', zero_division=0
        )
        
        # Support is the number of true positives in ground truth
        true_support = int(y_true[:, i].sum())
        
        per_label[name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': true_support,
            'predicted_positives': int(y_pred_binary[:, i].sum()),
            'true_positives': int(((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1)).sum()),
        }
    
    # Confusion matrices
    cm = multilabel_confusion_matrix(y_true, y_pred_binary)
    
    return {
        'micro': {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1': float(f1_micro),
        },
        'macro': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro),
        },
        'per_label': per_label,
        'confusion_matrices': cm.tolist(),
        'threshold': threshold,
        'num_samples': len(y_true),
    }


def main(args):
    print("=" * 60)
    print("STEP 7: Evaluate Violation Detector")
    print("=" * 60)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_config = config['training']
    
    # Load test data
    test_path = Path(args.input_dir) / "labeled_turns_test.jsonl"
    print(f"\nLoading test data from: {test_path}")
    
    test_samples = []
    with open(test_path) as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    print(f"  Test samples: {len(test_samples)}")
    
    # Check label distribution
    label_counts = {f'V{i+1}': 0 for i in range(5)}
    for sample in test_samples:
        labels = sample.get('labels', {})
        for k, v in labels.items():
            if v == 1:
                label_counts[k] += 1
    
    print(f"\nTest label distribution:")
    for k, v in label_counts.items():
        print(f"  {k}: {v}")
    
    # Load model
    model_path = Path(args.model_dir) / "final_model"
    print(f"\nLoading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    
    # Create dataset
    max_length = model_config.get('max_length', 512)
    test_dataset = ViolationDataset(test_samples, tokenizer, max_length)
    
    # Evaluate
    print(f"\nRunning inference on {len(test_dataset)} samples...")
    
    all_predictions = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            batch = test_dataset[i]
            
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels'].numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()[0]
            probs = 1 / (1 + np.exp(-logits))  # Sigmoid
            
            all_predictions.append(probs)
            all_labels.append(labels)
            all_sample_ids.append(batch['sample_id'])
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_dataset)}")
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print("\nInference complete!")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_detailed_metrics(all_labels, all_predictions, threshold=0.5)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'model_path': str(model_path),
        'test_path': str(test_path),
        'num_samples': len(test_samples),
        'label_distribution': label_counts,
    }
    
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save predictions
    predictions_data = []
    for sample_id, pred, label in zip(all_sample_ids, all_predictions, all_labels):
        predictions_data.append({
            'sample_id': sample_id,
            'predictions': {
                'V1': float(pred[0]),
                'V2': float(pred[1]),
                'V3': float(pred[2]),
                'V4': float(pred[3]),
                'V5': float(pred[4]),
            },
            'labels': {
                'V1': int(label[0]),
                'V2': int(label[1]),
                'V3': int(label[2]),
                'V4': int(label[3]),
                'V5': int(label[4]),
            },
        })
    
    predictions_path = output_dir / "test_predictions.jsonl"
    with open(predictions_path, 'w') as f:
        for item in predictions_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Predictions saved to: {predictions_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Micro - P: {metrics['micro']['precision']:.4f}, R: {metrics['micro']['recall']:.4f}, F1: {metrics['micro']['f1']:.4f}")
    print(f"  Macro - P: {metrics['macro']['precision']:.4f}, R: {metrics['macro']['recall']:.4f}, F1: {metrics['macro']['f1']:.4f}")
    
    print(f"\nPer-Label Metrics:")
    print(f"{'Label':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10} {'Predicted':<10} {'TP':<10}")
    print("-" * 80)
    for label, stats in metrics['per_label'].items():
        print(f"{label:<8} {stats['precision']:<12.4f} {stats['recall']:<12.4f} {stats['f1']:<12.4f} {stats['support']:<10} {stats['predicted_positives']:<10} {stats['true_positives']:<10}")
    
    print("\n" + "=" * 60)
    print("✅ Step 7 complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Evaluate Violation Detector")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with labeled test.jsonl")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory with trained model")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for results")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    
    args = parser.parse_args()
    main(args)
