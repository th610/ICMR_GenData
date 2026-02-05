"""
모델로 ESConv 1300 prefixes 예측
GPU 사용, 학습 때와 동일한 전처리 적용
"""
import json
import torch
from transformers import RobertaTokenizer
from models.model import ViolationClassifier
from models.data_utils import format_conversation_with_truncation
from tqdm import tqdm
import os

# Label mapping
LABEL_MAP = {
    "Normal": 0,
    "V1": 1,
    "V2": 2,
    "V3": 3,
    "V4": 4,
    "V5": 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_model(model_path, device='cuda'):
    """모델 로드"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Tokenizer 로드
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    special_tokens = ['<SEEKER>', '<SUPPORTER>', '<SUPPORTER_TARGET>']
    tokenizer.add_tokens(special_tokens)
    
    # Model 로드
    model = ViolationClassifier(
        model_name='roberta-base',
        num_labels=6,
        pooling='cls',
        dropout=0.1
    )
    model.roberta.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']})")
    return model, tokenizer

def split_dialog_into_prefix_and_generated(dialog):
    """
    ESConv dialog를 학습 형식으로 분리
    마지막 4턴을 generated_dialog로 취급
    
    Args:
        dialog: ESConv prefix의 전체 대화 턴
    
    Returns:
        prefix_dialog: 마지막 4턴을 제외한 앞부분
        generated_dialog: 마지막 4턴
    """
    if len(dialog) < 4:
        # 4턴 미만이면 빈 prefix + 전체를 generated로
        return [], dialog
    
    # 마지막 4턴을 generated_dialog로
    generated_dialog = dialog[-4:]
    prefix_dialog = dialog[:-4]
    
    return prefix_dialog, generated_dialog

def convert_esconv_turn_format(turn):
    """
    ESConv 턴 형식을 학습 데이터 형식으로 변환
    ESConv: {'speaker': 'seeker', 'content': '...', 'annotation': {...}}
    학습: {'speaker': 'seeker', 'content': '...', ...}
    """
    return {
        'speaker': turn['speaker'],
        'content': turn['content']
    }

def predict_batch(model, tokenizer, texts, device='cuda', batch_size=16):
    """배치 예측"""
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    return predictions

def main():
    print("="*80)
    print("Model Prediction for ESConv 1300 Prefixes")
    print("="*80)
    
    # Check GPU - Force CPU (RTX 5060 not compatible with PyTorch 2.6)
    device = 'cpu'
    print(f"\nDevice: {device}")
    print("Note: RTX 5060 (sm_120) not supported by current PyTorch")
    
    # Load prefixes
    print("\n1. Loading ESConv prefixes...")
    with open('generation/outputs/assigned/ESConv_1300_prefixes.json', 'r', encoding='utf-8') as f:
        prefixes = json.load(f)
    print(f"   Total prefixes: {len(prefixes)}")
    
    # Load model
    print("\n2. Loading model...")
    model, tokenizer = load_model('models/best_model/best_model.pt', device)
    
    # Format conversations (학습 때와 동일한 방식)
    print("\n3. Formatting conversations...")
    texts = []
    skipped = 0
    
    for prefix in prefixes:
        dialog = prefix['dialog']
        
        # 마지막 4턴을 generated_dialog로 분리
        prefix_dialog, generated_dialog = split_dialog_into_prefix_and_generated(dialog)
        
        # 4턴이 안되면 스킵 (학습 데이터는 무조건 4턴)
        if len(generated_dialog) < 4:
            skipped += 1
            texts.append("")  # 빈 문자열로 자리 유지
            continue
        
        # ESConv 형식을 학습 형식으로 변환
        prefix_dialog_converted = [convert_esconv_turn_format(t) for t in prefix_dialog]
        generated_dialog_converted = [convert_esconv_turn_format(t) for t in generated_dialog]
        
        # 학습 때와 동일한 전처리 함수 사용
        formatted_text = format_conversation_with_truncation(
            prefix_dialog_converted,
            generated_dialog_converted,
            tokenizer,
            max_length=512
        )
        texts.append(formatted_text)
    
    print(f"   Formatted {len(texts)} conversations (skipped: {skipped})")
    
    # Predict
    print("\n4. Running predictions...")
    print(f"   Batch size: 16")
    
    model_preds = []
    for i in tqdm(range(0, len(texts), 16), desc="Predicting"):
        batch_texts = texts[i:i+16]
        # 빈 문자열 필터링
        valid_texts = [t for t in batch_texts if t]
        if valid_texts:
            batch_preds = predict_batch(model, tokenizer, valid_texts, device, batch_size=len(valid_texts))
            
            # 원래 순서 유지 (빈 문자열은 -1로 표시)
            pred_idx = 0
            for t in batch_texts:
                if t:
                    model_preds.append(batch_preds[pred_idx])
                    pred_idx += 1
                else:
                    model_preds.append(-1)  # 스킵된 샘플
        else:
            model_preds.extend([-1] * len(batch_texts))
    
    print(f"\n   Completed: {len(model_preds)} predictions")
    
    # Convert to labels
    pred_labels = [ID_TO_LABEL[p] if p >= 0 else "SKIPPED" for p in model_preds]
    
    # Statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    from collections import Counter
    label_counts = Counter(pred_labels)
    
    valid_preds = [p for p in pred_labels if p != "SKIPPED"]
    total_valid = len(valid_preds)
    
    print(f"\nTotal: {len(pred_labels)}")
    print(f"Valid: {total_valid}")
    print(f"Skipped: {label_counts.get('SKIPPED', 0)}")
    
    print("\nLabel distribution:")
    for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
        count = label_counts.get(label, 0)
        pct = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Save results
    output_path = 'generation/outputs/evaluated/model_predictions_1300.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = []
    for i, prefix in enumerate(prefixes):
        results.append({
            'esconv_session_id': prefix['esconv_session_id'],
            'situation': prefix['situation'],
            'predicted_label': pred_labels[i],
            'predicted_id': int(model_preds[i])  # Convert numpy int64 to Python int
        })
    
    output = {
        'metadata': {
            'total': len(results),
            'valid': total_valid,
            'skipped': label_counts.get('SKIPPED', 0),
            'model': 'best_model.pt',
            'device': device,
            'label_distribution': {k: v for k, v in label_counts.items() if k != 'SKIPPED'}
        },
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
