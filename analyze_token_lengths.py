import json
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

print("=" * 70)
print("학습 데이터셋 토큰 길이 분석")
print("=" * 70)

# Train/Valid/Test 각각 분석
for split in ['train', 'valid', 'test']:
    print(f"\n{'='*70}")
    print(f"{split.upper()} 데이터셋 토큰 길이")
    print(f"{'='*70}")
    
    data = json.load(open(f'data/final/{split}.json', encoding='utf-8'))
    
    all_lengths = []
    class_lengths = {
        'normal': [], 'v1': [], 'v2': [], 'v3': [], 'v4': [], 'v5': []
    }
    
    for session in data:
        # 원본 데이터 형식으로 토큰 계산 (전체 concat)
        dialog = session.get('dialog', [])
        text_parts = []
        for turn in dialog:
            speaker = turn.get('speaker', 'unknown')
            text = turn.get('content', turn.get('text', ''))
            text_parts.append(f"{speaker}: {text}")
        
        full_text = " ".join(text_parts)
        tokens = tokenizer(full_text, truncation=True, max_length=512)
        token_len = len(tokens['input_ids'])
        
        all_lengths.append(token_len)
        label = session.get('label', 'unknown')
        if label in class_lengths:
            class_lengths[label].append(token_len)
    
    # 전체 통계
    print(f"\n전체 통계 (N={len(all_lengths)})")
    print(f"  평균:  {np.mean(all_lengths):.1f} tokens")
    print(f"  중앙값: {np.median(all_lengths):.1f} tokens")
    print(f"  최소:  {np.min(all_lengths)} tokens")
    print(f"  최대:  {np.max(all_lengths)} tokens")
    print(f"  표준편차: {np.std(all_lengths):.1f} tokens")
    
    # 512 토큰 초과 비율
    truncated = sum(1 for l in all_lengths if l == 512)
    print(f"  512 토큰 도달: {truncated}/{len(all_lengths)} ({truncated/len(all_lengths)*100:.1f}%)")
    
    # 클래스별 통계
    print(f"\n클래스별 평균 토큰 길이")
    print(f"  {'Class':<10} {'Count':<8} {'평균':<10} {'중앙값':<10} {'최대':<10}")
    print(f"  {'-'*60}")
    
    for cls in ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']:
        if class_lengths[cls]:
            lengths = class_lengths[cls]
            print(f"  {cls:<10} {len(lengths):<8} {np.mean(lengths):<10.1f} {np.median(lengths):<10.1f} {np.max(lengths):<10}")

# 전처리 데이터 분석 (요약+윈도우)
print(f"\n{'='*70}")
print("전처리 데이터 (요약+윈도우6) 토큰 길이")
print(f"{'='*70}")

files = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']
all_processed = []
class_processed = {cls: [] for cls in files}

for f in files:
    data = json.load(open(f'data/processed/{f}_processed.json', encoding='utf-8'))
    
    for session in data:
        # 전처리 형식: summary + context + response
        summary_text = "\n".join(session.get('summary', []))
        context_turns = session.get('context_turns', [])
        context_text = " ".join([f"{t['speaker']}: {t['text']}" for t in context_turns])
        response_text = session.get('response', '')
        
        full_text = f"[SUMMARY]\n{summary_text}\n\n[CONTEXT]\n{context_text}\n\n[RESPONSE]\n{response_text}"
        
        tokens = tokenizer(full_text, truncation=True, max_length=512)
        token_len = len(tokens['input_ids'])
        
        all_processed.append(token_len)
        class_processed[f].append(token_len)

print(f"\n전체 통계 (N={len(all_processed)})")
print(f"  평균:  {np.mean(all_processed):.1f} tokens")
print(f"  중앙값: {np.median(all_processed):.1f} tokens")
print(f"  최소:  {np.min(all_processed)} tokens")
print(f"  최대:  {np.max(all_processed)} tokens")
print(f"  표준편차: {np.std(all_processed):.1f} tokens")

truncated = sum(1 for l in all_processed if l == 512)
print(f"  512 토큰 도달: {truncated}/{len(all_processed)} ({truncated/len(all_processed)*100:.1f}%)")

print(f"\n클래스별 평균 토큰 길이")
print(f"  {'Class':<10} {'Count':<8} {'평균':<10} {'중앙값':<10} {'최대':<10}")
print(f"  {'-'*60}")

for cls in files:
    lengths = class_processed[cls]
    print(f"  {cls:<10} {len(lengths):<8} {np.mean(lengths):<10.1f} {np.median(lengths):<10.1f} {np.max(lengths):<10}")
