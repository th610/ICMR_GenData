"""
Train RoBERTa classifier on violation detection dataset

Input: data/final/{train,valid,test}.json
Output: models/violation_classifier/ + results/
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from pathlib import Path
from datetime import datetime
import sys


# Setup logging to file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return self.terminal.isatty()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout


# Label mapping
LABEL_MAP = {
    'normal': 0,
    'v1': 1,
    'v2': 2,
    'v3': 3,
    'v4': 4,
    'v5': 5
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def session_to_text(session):
    """Convert session to text input"""
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


class ViolationDataset(Dataset):
    """Dataset for violation detection"""
    
    def __init__(self, filepath, tokenizer, max_length=512):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        session = self.data[idx]
        
        # Convert to text
        text = session_to_text(session)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        label = LABEL_MAP[session['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Macro F1
    f1_macro = f1_score(labels, predictions, average='macro')
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }


def evaluate_model(trainer, test_dataset, output_dir):
    """Evaluate model and save detailed results"""
    
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)
    
    # Predict
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[ID2LABEL[i] for i in range(6)],
        digits=4
    )
    
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("True ↓     ", " ".join([f"{ID2LABEL[i]:8s}" for i in range(6)]))
    for i, row in enumerate(cm):
        print(f"{ID2LABEL[i]:8s}  ", " ".join([f"{val:8d}" for val in row]))
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':8s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>10s}")
    print("-" * 60)
    for i in range(6):
        print(f"{ID2LABEL[i]:8s}  {precision[i]:10.4f}  {recall[i]:10.4f}  {f1[i]:10.4f}  {support[i]:10d}")
    
    # Overall metrics
    overall_acc = (pred_labels == true_labels).mean()
    overall_f1 = f1_score(true_labels, pred_labels, average='macro')
    
    print("\n" + "="*80)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Macro F1-Score:   {overall_f1:.4f}")
    print("="*80)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(overall_acc),
            'f1_macro': float(overall_f1)
        },
        'per_class': {
            ID2LABEL[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i in range(6)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved: {results_file}\n")
    
    return results


def main():
    print("\n" + "="*80)
    print("RoBERTa Violation Classifier Training")
    print("="*80 + "\n")
    
    # Setup
    model_name = "roberta-base"
    output_dir = Path("models/violation_classifier")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ViolationDataset('data/final/train.json', tokenizer)
    valid_dataset = ViolationDataset('data/final/valid.json', tokenizer)
    test_dataset = ViolationDataset('data/final/test.json', tokenizer)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Initialize model
    print(f"\nInitializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save best model
    print("\n" + "="*80)
    print("Saving Best Model")
    print("="*80)
    best_model_path = output_dir / 'best_model'
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))  # ✅ Save tokenizer too!
    print(f"✅ Model saved: {best_model_path}")
    print(f"✅ Tokenizer saved: {best_model_path}\n")
    
    # Evaluate on test set
    results = evaluate_model(trainer, test_dataset, results_dir)
    
    print("\n✅ Training and Evaluation Complete!\n")


if __name__ == "__main__":
    main()
