"""
Training Script for Violation Classifier
=========================================
RoBERTa 기반 대화 위반 탐지 분류기 학습

실행 방법:
    python models/train.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_utils import (
    load_json_data,
    print_data_statistics,
    split_train_valid,
    ViolationDataset,
    calculate_class_weights,
    setup_tokenizer,
    DataConfig
)
from model import ViolationClassifier


# ============================================================================
# Configuration
# ============================================================================

class TrainConfig:
    """학습 설정"""
    # Paths
    SOURCE_DATA_DIR = "data/final"  # 학습 데이터 위치 (수정된 CSV 병합본)
    SOURCE_TRAIN_FILE = "train_1000.json"  # 1000개 (270개 edited: Normal 100 + V1~V5 170)
    SOURCE_TEST_FILE = "test_gold_300.json"  # 300개 (고정, 복사 필요)
    
    SPLIT_DATA_DIR = "models/data"  # Split 결과 저장 위치
    OUTPUT_DIR = "models/outputs"  # 모델 출력 위치
    
    VALID_RATIO = 0.2  # train에서 valid로 분할할 비율 (20% = 200개)
    
    # Model
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 512
    POOLING = "cls"  # "cls" or "mean"
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1
    
    # Reproducibility
    SEED = 42
    
    # Early stopping
    PATIENCE = 3
    METRIC_FOR_BEST = "macro_f1"  # "macro_f1" or "v5_recall"
    
    # Class weight method
    WEIGHT_METHOD = "sqrt"  # "sqrt" (권장) or "inverse"


def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Training & Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, split_name="Validation"):
    """
    평가 및 상세 메트릭 출력
    
    Returns:
        dict: 평가 메트릭
            - loss, accuracy, macro_f1, macro_precision, macro_recall
            - v5_recall (핵심 지표)
            - per_class_f1, confusion_matrix
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics 계산
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(DataConfig.LABELS)), zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(DataConfig.LABELS)))
    
    # Print results
    print(f"\n{'='*80}")
    print(f"[{split_name}] Evaluation Results")
    print(f"{'='*80}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\n{'-'*80}")
    print(f"Per-class Metrics:")
    print(f"{'-'*80}")
    print(f"{'Class':<10} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*80}")
    
    for idx, label in enumerate(DataConfig.LABELS):
        print(f"{label:<10} {support[idx]:<10} {precision[idx]:<12.4f} {recall[idx]:<12.4f} {f1[idx]:<12.4f}")
    
    print(f"\n{'-'*80}")
    print(f"Confusion Matrix:")
    print(f"{'-'*80}")
    print("Predicted →")
    header = "True ↓".ljust(12) + "  ".join([f"{label:>8}" for label in DataConfig.LABELS])
    print(header)
    for idx, label in enumerate(DataConfig.LABELS):
        row = f"{label:<12}" + "  ".join([f"{cm[idx][j]:8d}" for j in range(len(DataConfig.LABELS))])
        print(row)
    print(f"{'='*80}\n")
    
    # V5 Recall (핵심 지표)
    v5_idx = DataConfig.LABEL2ID['V5']
    v5_recall = recall[v5_idx]
    print(f"🎯 V5 Recall (Key Metric): {v5_recall:.4f}\n")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'v5_recall': v5_recall,
        'per_class_f1': f1.tolist(),
        'confusion_matrix': cm.tolist()
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # ========================================================================
    # 0. Setup
    # ========================================================================
    print("\n" + "="*80)
    print("Violation Classifier Training")
    print("="*80 + "\n")
    
    set_seed(TrainConfig.SEED)
    
    # GPU 설정 (비어있는 GPU 1 사용)
    if torch.cuda.is_available():
        gpu_id = 0  # GPU 0번 사용
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    print(f"Model: {TrainConfig.MODEL_NAME}")
    print(f"Max Length: {TrainConfig.MAX_LENGTH}")
    print(f"Pooling: {TrainConfig.POOLING}")
    print(f"Batch Size: {TrainConfig.BATCH_SIZE}")
    print(f"Learning Rate: {TrainConfig.LEARNING_RATE}")
    print(f"Epochs: {TrainConfig.NUM_EPOCHS}")
    print(f"Seed: {TrainConfig.SEED}\n")
    
    # ========================================================================
    # 1. Load Original Data & Split
    # ========================================================================
    print("="*80)
    print("Step 1: Loading Original Data & Splitting Train/Valid")
    print("="*80)
    
    # 원본 데이터 로드
    source_train_path = os.path.join(TrainConfig.SOURCE_DATA_DIR, TrainConfig.SOURCE_TRAIN_FILE)
    source_test_path = os.path.join(TrainConfig.SOURCE_DATA_DIR, TrainConfig.SOURCE_TEST_FILE)
    
    print(f"\nLoading source data from:")
    print(f"  Train: {source_train_path}")
    print(f"  Test:  {source_test_path}")
    
    original_train_data = load_json_data(source_train_path)
    test_data = load_json_data(source_test_path)
    
    print(f"\nOriginal train samples: {len(original_train_data['samples'])}")
    print(f"Test samples: {len(test_data['samples'])}")
    
    # Train/Valid split 수행
    train_data, valid_data = split_train_valid(
        original_train_data, 
        valid_ratio=TrainConfig.VALID_RATIO,
        seed=TrainConfig.SEED
    )
    
    # Split 결과 저장
    os.makedirs(TrainConfig.SPLIT_DATA_DIR, exist_ok=True)
    
    train_save_path = os.path.join(TrainConfig.SPLIT_DATA_DIR, "train_split.json")
    valid_save_path = os.path.join(TrainConfig.SPLIT_DATA_DIR, "valid_split.json")
    test_save_path = os.path.join(TrainConfig.SPLIT_DATA_DIR, "test_split.json")
    
    with open(train_save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(valid_save_path, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)
    with open(test_save_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSplit results saved to {TrainConfig.SPLIT_DATA_DIR}/:")
    print(f"  ✓ train_split.json ({len(train_data['samples'])} samples)")
    print(f"  ✓ valid_split.json ({len(valid_data['samples'])} samples)")
    print(f"  ✓ test_split.json ({len(test_data['samples'])} samples)")
    
    print_data_statistics(train_data, "Train")
    print_data_statistics(valid_data, "Valid")
    print_data_statistics(test_data, "Test Gold")
    
    # ========================================================================
    # 2. Initialize Tokenizer
    # ========================================================================
    print("="*80)
    print("Step 2: Initializing Tokenizer")
    print("="*80)
    tokenizer = RobertaTokenizer.from_pretrained(TrainConfig.MODEL_NAME)
    tokenizer, num_added = setup_tokenizer(tokenizer)
    print(f"Added {num_added} special tokens:")
    print(f"  - {DataConfig.SEEKER_TOKEN}")
    print(f"  - {DataConfig.SUPPORTER_TOKEN}")
    print(f"  - {DataConfig.SUPPORTER_TARGET_TOKEN}\n")
    
    # ========================================================================
    # 3. Create Datasets
    # ========================================================================
    print("="*80)
    print("Step 3: Creating Datasets")
    print("="*80)
    train_dataset = ViolationDataset(train_data, tokenizer, TrainConfig.MAX_LENGTH)
    valid_dataset = ViolationDataset(valid_data, tokenizer, TrainConfig.MAX_LENGTH)
    test_dataset = ViolationDataset(test_data, tokenizer, TrainConfig.MAX_LENGTH, is_test=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # ========================================================================
    # 4. Create DataLoaders
    # ========================================================================
    print("="*80)
    print("Step 4: Creating DataLoaders")
    print("="*80)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print("DataLoaders created.\n")
    
    # ========================================================================
    # 5. Calculate Class Weights
    # ========================================================================
    print("="*80)
    print("Step 5: Calculating Class Weights")
    print("="*80)
    class_weights = calculate_class_weights(train_data, method=TrainConfig.WEIGHT_METHOD)
    class_weights = class_weights.to(device)
    
    # ========================================================================
    # 6. Initialize Model
    # ========================================================================
    print("="*80)
    print("Step 6: Initializing Model")
    print("="*80)
    model = ViolationClassifier(
        model_name=TrainConfig.MODEL_NAME,
        num_labels=len(DataConfig.LABELS),
        dropout=TrainConfig.DROPOUT,
        pooling=TrainConfig.POOLING
    )
    
    # Resize token embeddings for special tokens
    model.roberta.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # ========================================================================
    # 7. Setup Loss & Optimizer
    # ========================================================================
    print("="*80)
    print("Step 7: Setting up Loss & Optimizer")
    print("="*80)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * TrainConfig.NUM_EPOCHS
    warmup_steps = int(total_steps * TrainConfig.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    # ========================================================================
    # 8. Training Loop
    # ========================================================================
    print("="*80)
    print("Step 8: Starting Training")
    print("="*80 + "\n")
    
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs(TrainConfig.OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(TrainConfig.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{TrainConfig.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate on validation set (for early stopping)
        valid_metrics = evaluate(model, valid_loader, criterion, device, split_name="Valid")
        
        # Check for best model
        current_metric = valid_metrics[TrainConfig.METRIC_FOR_BEST]
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(TrainConfig.OUTPUT_DIR, "best_model_v3.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'train_config': {
                    'model_name': TrainConfig.MODEL_NAME,
                    'max_length': TrainConfig.MAX_LENGTH,
                    'pooling': TrainConfig.POOLING,
                    'batch_size': TrainConfig.BATCH_SIZE,
                    'learning_rate': TrainConfig.LEARNING_RATE,
                    'num_epochs': TrainConfig.NUM_EPOCHS
                }
            }, save_path)
            print(f"✅ Best model saved! ({TrainConfig.METRIC_FOR_BEST}: {best_metric:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{TrainConfig.PATIENCE}")
        
        # Early stopping
        if patience_counter >= TrainConfig.PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # 9. Final Evaluation with Best Model
    # ========================================================================
    print(f"\n{'='*80}")
    print("Step 9: Final Evaluation with Best Model")
    print(f"{'='*80}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {TrainConfig.METRIC_FOR_BEST}: {best_metric:.4f}\n")
    
    # Load best model
    checkpoint = torch.load(os.path.join(TrainConfig.OUTPUT_DIR, "best_model_v3.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    final_metrics = evaluate(model, test_loader, criterion, device, split_name="Final Test Gold")
    
    # Save metrics
    metrics_path = os.path.join(TrainConfig.OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'final_metrics': final_metrics,
            'train_config': {
                'model_name': TrainConfig.MODEL_NAME,
                'max_length': TrainConfig.MAX_LENGTH,
                'batch_size': TrainConfig.BATCH_SIZE,
                'learning_rate': TrainConfig.LEARNING_RATE
            }
        }, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Best model saved to: {os.path.join(TrainConfig.OUTPUT_DIR, 'best_model_v3.pt')}")
    print(f"📁 Metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    main()
