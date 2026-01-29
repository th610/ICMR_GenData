"""
Step 6: Train violation detector model.

Input: labeled_turns_{train/val/test}.jsonl
Output: models/detector/ (checkpoints + training logs)
"""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_jsonl, load_yaml


def serialize_sample(sample):
    """
    Serialize turn sample into text input.
    
    Format:
    [SITUATION]
    {emotion_type} related to {problem_type}
    
    [SUMMARY]
    - bullet1
    - bullet2
    
    [CONTEXT]
    speaker: text
    speaker: text
    
    [RESPONSE]
    response text
    """
    meta = sample.get('meta', {})
    situation = f"{meta.get('emotion_type', 'unknown')} related to {meta.get('problem_type', 'unknown')}"
    
    # Summary
    summary_bullets = sample.get('summary', [])
    summary_text = '\n'.join([f"- {b}" for b in summary_bullets])
    
    # Context
    context_turns = sample.get('context_turns', [])
    context_text = '\n'.join([f"{t.get('speaker', 'unknown')}: {t.get('text', '')}" 
                              for t in context_turns])
    
    # Response
    response = sample.get('response', '')
    
    # Combine
    text = f"""[SITUATION]
{situation}

[SUMMARY]
{summary_text}

[CONTEXT]
{context_text}

[RESPONSE]
{response}"""
    
    return text


class ViolationDataset(Dataset):
    """Dataset for violation detection."""
    
    def __init__(self, samples, tokenizer, max_length=512, label_names=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_names or ['V1', 'V2', 'V3', 'V4', 'V5']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Serialize text
        text = serialize_sample(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Extract labels (multi-label: 0 or 1 for each V1~V5)
        labels_dict = sample.get('labels', {})
        labels = [float(labels_dict.get(v, 0)) for v in self.label_names]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


def compute_metrics(eval_pred):
    """Compute metrics for multi-label classification."""
    predictions, labels = eval_pred
    
    # Sigmoid + threshold
    probs = 1 / (1 + np.exp(-predictions))  # Sigmoid
    preds = (probs > 0.5).astype(int)
    
    # Micro/Macro F1
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Per-label F1
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    metrics = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'V1_f1': per_label_f1[0] if len(per_label_f1) > 0 else 0,
        'V2_f1': per_label_f1[1] if len(per_label_f1) > 1 else 0,
        'V3_f1': per_label_f1[2] if len(per_label_f1) > 2 else 0,
        'V4_f1': per_label_f1[3] if len(per_label_f1) > 3 else 0,
        'V5_f1': per_label_f1[4] if len(per_label_f1) > 4 else 0,
    }
    
    return metrics


def main(args):
    print(f"\n{'='*60}")
    print("STEP 6: Train Violation Detector")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_yaml(args.config)
    
    # Load data
    print(f"Loading training data from: {args.input_dir}")
    train_samples = load_jsonl(str(Path(args.input_dir) / "labeled_turns_train.jsonl"))
    val_samples = load_jsonl(str(Path(args.input_dir) / "labeled_turns_val.jsonl"))
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    
    # Initialize tokenizer
    model_name = config['training']['model_name']
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    label_names = config['labels']
    max_length = config['training']['max_length']
    
    print(f"Creating datasets (max_length={max_length})...")
    train_dataset = ViolationDataset(train_samples, tokenizer, max_length, label_names)
    val_dataset = ViolationDataset(val_samples, tokenizer, max_length, label_names)
    
    # Initialize model
    print(f"\nInitializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification"
    )
    
    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config['training']['num_epochs']),
        per_device_train_batch_size=int(config['training']['batch_size']),
        per_device_eval_batch_size=int(config['training']['batch_size']),
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=int(config['training']['warmup_steps']),
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard for PoC
    )
    
    # Trainer
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Output dir: {output_dir}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\nStarting training...")
    print("(This may take several minutes depending on hardware)\n")
    
    train_result = trainer.train()
    
    # Save model
    print(f"\nSaving final model to: {output_dir}")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    # Save metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")
    
    # Final evaluation on val
    print("\nFinal validation evaluation:")
    val_metrics = trainer.evaluate()
    
    print(f"\n{'='*60}")
    print("Training Results")
    print(f"{'='*60}")
    print(f"Micro F1: {val_metrics.get('eval_micro_f1', 0):.4f}")
    print(f"Macro F1: {val_metrics.get('eval_macro_f1', 0):.4f}")
    print(f"\nPer-label F1:")
    for label in label_names:
        f1_key = f"eval_{label}_f1"
        print(f"  {label}: {val_metrics.get(f1_key, 0):.4f}")
    print(f"{'='*60}\n")
    
    print("✅ Step 6 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/labeled",
                        help="Directory with labeled JSONL files")
    parser.add_argument("--output_dir", type=str, default="models/detector",
                        help="Output directory for model checkpoints")
    parser.add_argument("--config", type=str, default="configs/poc.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    main(args)
