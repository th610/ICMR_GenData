"""
Violation Classifier Training Script
=====================================
RoBERTa-base 기반 대화 위반 탐지 분류기 학습

입력 구조:
- prefix (최근 6-8턴) + generated_dialog (4턴)
- 역할 토큰: <SEEKER>, <SUPPORTER>, <SUPPORTER_TARGET>
- 512 토큰 역방향 truncation (TARGET 기준)

레이블:
- Normal, V1, V2, V3, V4, V5 (6-class classification)
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 0. Configuration & Seed Fixing
# ============================================================================

class Config:
    # Paths
    DATA_DIR = "data/final"
    TRAIN_FILE = "train.json"
    TEST_FILE = "test_gold.json"
    OUTPUT_DIR = "models/violation_classifier"
    
    # Model
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 512
    POOLING = "cls"  # "cls" or "mean"
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1
    
    # Window
    WINDOW_SIZE = 8  # prefix에서 최근 몇 턴까지 포함할지
    
    # Class labels
    LABELS = ["Normal", "V1", "V2", "V3", "V4", "V5"]
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
    
    # Special tokens
    SEEKER_TOKEN = "<SEEKER>"
    SUPPORTER_TOKEN = "<SUPPORTER>"
    SUPPORTER_TARGET_TOKEN = "<SUPPORTER_TARGET>"
    
    # Reproducibility
    SEED = 42
    
    # Early stopping
    PATIENCE = 3
    METRIC_FOR_BEST = "macro_f1"  # "macro_f1" or "v5_recall"


def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 1. Data Loading
# ============================================================================

def load_json_data(filepath):
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def print_data_statistics(data, split_name):
    """데이터 통계 출력"""
    print(f"\n{'='*60}")
    print(f"[{split_name}] Data Statistics")
    print(f"{'='*60}")
    
    metadata = data.get('metadata', {})
    distribution = metadata.get('distribution', {})
    total = metadata.get('total_samples', len(data['samples']))
    
    print(f"Total samples: {total}")
    print(f"\nClass distribution:")
    for label in Config.LABELS:
        count = distribution.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
    print(f"{'='*60}\n")


# ============================================================================
# 2. Text Preprocessing & Formatting
# ============================================================================

def format_conversation_with_truncation(prefix_dialog, generated_dialog, tokenizer, max_length=512):
    """
    대화를 포맷팅하고 512 토큰 역방향 truncation 적용
    
    우선순위: TARGET (Turn 4) > Trigger (Turn 3) > Turn 2 > Turn 1 > prefix (최근부터)
    - TARGET/Trigger는 부분 잘림 허용
    - 나머지 턴은 통째로 포함 or drop
    
    Returns:
        formatted_text (str): 최종 입력 텍스트
    """
    # Step 1: generated_dialog 4턴 구조 확인
    if len(generated_dialog) != 4:
        raise ValueError(f"generated_dialog must have exactly 4 turns, got {len(generated_dialog)}")
    
    turn1 = generated_dialog[0]  # seeker
    turn2 = generated_dialog[1]  # supporter
    turn3 = generated_dialog[2]  # seeker (Trigger)
    turn4 = generated_dialog[3]  # supporter (TARGET)
    
    # Step 2: 각 턴을 역할 토큰과 함께 포맷팅
    def format_turn(turn, is_target=False):
        speaker = turn['speaker'].lower()
        content = turn['content'].strip()
        
        if is_target:
            token = Config.SUPPORTER_TARGET_TOKEN
        elif speaker == 'seeker':
            token = Config.SEEKER_TOKEN
        elif speaker == 'supporter':
            token = Config.SUPPORTER_TOKEN
        else:
            token = Config.SUPPORTER_TOKEN
        
        return f"{token} {content}"
    
    target_text = format_turn(turn4, is_target=True)
    trigger_text = format_turn(turn3, is_target=False)
    turn2_text = format_turn(turn2, is_target=False)
    turn1_text = format_turn(turn1, is_target=False)
    
    # Step 3: prefix에서 최근 WINDOW_SIZE 턴만 가져오기
    prefix_turns = []
    if prefix_dialog:
        recent_prefix = prefix_dialog[-Config.WINDOW_SIZE:] if len(prefix_dialog) > Config.WINDOW_SIZE else prefix_dialog
        for turn in recent_prefix:
            prefix_turns.append(format_turn(turn, is_target=False))
    
    # Step 4: 역방향 truncation (뒤에서부터 채우기)
    # 우선 TARGET은 무조건 포함
    parts = [target_text]
    
    # Tokenize해서 길이 체크 (special tokens 고려)
    def get_token_count(text_parts):
        combined = "\n".join(text_parts)
        tokens = tokenizer.encode(combined, add_special_tokens=True)
        return len(tokens)
    
    current_length = get_token_count(parts)
    
    # Trigger 추가 시도
    temp_parts = [trigger_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Trigger 부분 잘림 허용
        available_tokens = max_length - current_length - 10  # 여유분
        trigger_tokens = tokenizer.encode(trigger_text, add_special_tokens=False)
        if available_tokens > 0:
            truncated_trigger = tokenizer.decode(trigger_tokens[-available_tokens:], skip_special_tokens=False)
            parts = [truncated_trigger] + parts
        return "\n".join(parts)
    
    # Turn 2 추가 시도 (통째로만)
    temp_parts = [turn2_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Turn 2는 부분 잘림 불가 → skip
        pass
    
    # Turn 1 추가 시도 (통째로만)
    temp_parts = [turn1_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Turn 1은 부분 잘림 불가 → skip
        pass
    
    # Prefix 추가 (최근 턴부터, 통째로만)
    for prefix_turn in reversed(prefix_turns):
        temp_parts = [prefix_turn] + parts
        if get_token_count(temp_parts) <= max_length:
            parts = temp_parts
            current_length = get_token_count(parts)
        else:
            # 오래된 prefix는 drop
            break
    
    # Step 5: 최종 텍스트 생성
    final_text = "\n".join(parts)
    return final_text


# ============================================================================
# 3. Dataset Class
# ============================================================================

class ViolationDataset(Dataset):
    """대화 위반 탐지 데이터셋"""
    
    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        self.samples = data['samples']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 텍스트 포맷팅 (512 토큰 역방향 truncation)
        prefix_dialog = sample.get('prefix_dialog', [])
        generated_dialog = sample['generated_dialog']
        
        text = format_conversation_with_truncation(
            prefix_dialog, 
            generated_dialog, 
            self.tokenizer, 
            self.max_length
        )
        
        # 2. 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. 레이블
        label = Config.LABEL2ID[sample['primary_label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': sample.get('esconv_session_id', idx)
        }


# ============================================================================
# 4. Model Definition
# ============================================================================

class ViolationClassifier(nn.Module):
    """RoBERTa 기반 위반 분류기"""
    
    def __init__(self, model_name, num_labels, dropout=0.1, pooling="cls"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask):
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pooling
        if self.pooling == "cls":
            # [CLS] token pooling
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling (attention mask 고려)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Dropout + Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


# ============================================================================
# 5. Class Weight Calculation
# ============================================================================

def calculate_class_weights(data, method="sqrt"):
    """
    클래스 가중치 계산
    
    Args:
        data: JSON 데이터
        method: "sqrt" (역제곱근) or "inverse" (역비례)
    
    Returns:
        torch.Tensor: 클래스 가중치 (shape: [num_classes])
    """
    distribution = data['metadata']['distribution']
    total = data['metadata']['total_samples']
    
    weights = []
    print(f"\n{'='*60}")
    print(f"Class Weights Calculation (method={method})")
    print(f"{'='*60}")
    
    for label in Config.LABELS:
        count = distribution[label]
        if method == "sqrt":
            # 역제곱근: sqrt(N_total / N_i)
            weight = np.sqrt(total / count)
        elif method == "inverse":
            # 역비례: N_total / N_i
            weight = total / count
        else:
            weight = 1.0
        
        weights.append(weight)
        print(f"  {label:8s}: count={count:4d}, weight={weight:.2f}")
    
    # 정규화 (평균이 1.0이 되도록)
    weights = np.array(weights)
    weights = weights / weights.mean()
    
    print(f"\nNormalized weights (mean=1.0):")
    for label, weight in zip(Config.LABELS, weights):
        print(f"  {label:8s}: {weight:.2f}")
    print(f"{'='*60}\n")
    
    return torch.FloatTensor(weights)


# ============================================================================
# 6. Training & Evaluation Functions
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
    """평가 및 상세 메트릭 출력"""
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
        all_labels, all_preds, average=None, labels=range(len(Config.LABELS)), zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(Config.LABELS)))
    
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
    
    for idx, label in enumerate(Config.LABELS):
        print(f"{label:<10} {support[idx]:<10} {precision[idx]:<12.4f} {recall[idx]:<12.4f} {f1[idx]:<12.4f}")
    
    print(f"\n{'-'*80}")
    print(f"Confusion Matrix:")
    print(f"{'-'*80}")
    print("Predicted →")
    header = "True ↓".ljust(12) + "  ".join([f"{label:>8}" for label in Config.LABELS])
    print(header)
    for idx, label in enumerate(Config.LABELS):
        row = f"{label:<12}" + "  ".join([f"{cm[idx][j]:8d}" for j in range(len(Config.LABELS))])
        print(row)
    print(f"{'='*80}\n")
    
    # V5 Recall (핵심 지표)
    v5_idx = Config.LABEL2ID['V5']
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
# 7. Main Training Loop
# ============================================================================

def main():
    # 0. Setup
    print("\n" + "="*80)
    print("Violation Classifier Training")
    print("="*80 + "\n")
    
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Max Length: {Config.MAX_LENGTH}")
    print(f"Pooling: {Config.POOLING}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Seed: {Config.SEED}\n")
    
    # 1. Load Data
    print("Step 1: Loading Data...")
    train_data = load_json_data(os.path.join(Config.DATA_DIR, Config.TRAIN_FILE))
    test_data = load_json_data(os.path.join(Config.DATA_DIR, Config.TEST_FILE))
    
    print_data_statistics(train_data, "Train")
    print_data_statistics(test_data, "Test Gold")
    
    # 2. Initialize Tokenizer (special tokens 추가)
    print("Step 2: Initializing Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            Config.SEEKER_TOKEN,
            Config.SUPPORTER_TOKEN,
            Config.SUPPORTER_TARGET_TOKEN
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens: {special_tokens['additional_special_tokens']}\n")
    
    # 3. Create Datasets
    print("Step 3: Creating Datasets...")
    train_dataset = ViolationDataset(train_data, tokenizer, Config.MAX_LENGTH)
    test_dataset = ViolationDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # 4. Create DataLoaders
    print("Step 4: Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print("DataLoaders created.\n")
    
    # 5. Calculate Class Weights
    print("Step 5: Calculating Class Weights...")
    class_weights = calculate_class_weights(train_data, method="sqrt")
    class_weights = class_weights.to(device)
    
    # 6. Initialize Model
    print("Step 6: Initializing Model...")
    model = ViolationClassifier(
        model_name=Config.MODEL_NAME,
        num_labels=len(Config.LABELS),
        dropout=Config.DROPOUT,
        pooling=Config.POOLING
    )
    
    # Resize token embeddings for special tokens
    model.roberta.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # 7. Setup Loss & Optimizer
    print("Step 7: Setting up Loss & Optimizer...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    # 8. Training Loop
    print("Step 8: Starting Training...")
    print("="*80 + "\n")
    
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device, split_name="Test Gold")
        
        # Check for best model
        current_metric = test_metrics[Config.METRIC_FOR_BEST]
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': vars(Config)
            }, save_path)
            print(f"✅ Best model saved! ({Config.METRIC_FOR_BEST}: {best_metric:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break
    
    # 9. Final Evaluation with Best Model
    print(f"\n{'='*80}")
    print("Step 9: Final Evaluation with Best Model")
    print(f"{'='*80}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {Config.METRIC_FOR_BEST}: {best_metric:.4f}\n")
    
    # Load best model
    checkpoint = torch.load(os.path.join(Config.OUTPUT_DIR, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    final_metrics = evaluate(model, test_loader, criterion, device, split_name="Final Test Gold")
    
    # Save metrics
    metrics_path = os.path.join(Config.OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'final_metrics': final_metrics,
            'config': vars(Config)
        }, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Best model saved to: {Config.OUTPUT_DIR}/best_model.pt")
    print(f"📁 Metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    main()
